#! /usr/bin/env python
import pickle
import sh
import os
import glob
import shutil
import argparse
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
import random
import pdb
import copy


#########
#   implements the SAT GNN.
#   TODO:
#   - do we need self.T?
#   - need to start messages from the input layer
#   - include the current primal solution as input
#   \/ only return embedding of input layer
#   - change mu_embedding vector to correspond to primal rather than dual
#   - maybe change feature vectors
#   - once it runs check that the message passing really goes all the way from the first to the last layer
#   - current hack: set dual for last layer to 0 cause they don't exist. maybe have a different thetat
#   - maybe have 3 types of thetas for in-layer, hidden layers, and out-layers?
#########

class EmbedLayerUpdate(nn.Module):
    '''
    this class updates embeded layer one time
    '''
    def __init__(self, T, p, args):
        super(EmbedLayerUpdate, self).__init__()
        '''
        p_1: input embedding size
        p_2: output embedding size
        '''
        self.mode_linear = True

        self.message_passing = True

        self.p = p
        self.T = T
        self.cpu = args['cpu']

        # self.feature_grad = 'feature_grad' in args.keys()
        self.feature_grad = args['feature_grad']
        self.lp_primal = args['lp_primal']

        self.normalize_conv = False

        feature_fwd = 3

        # weights_fwd = 2
        bias_fwd = True

        theta1 = nn.Linear(feature_fwd, p, bias=bias_fwd)

        self.thetas = nn.ModuleList()
        self.thetas.append(theta1)
        for i in range(1, self.T):
            theta_i = nn.Linear(p, p, bias=bias_fwd)
            self.thetas.append(theta_i)

        # if self.feature_grad:
        #     feature_fwd_input = 4
        # else:
        #     feature_fwd_input = 3
        feature_fwd_input = 3
        if self.feature_grad:
            feature_fwd_input += 1
        if self.lp_primal:
            feature_fwd_input += 1

        theta1_input = nn.Linear(feature_fwd_input, p, bias=bias_fwd)
        self.thetas_input = nn.ModuleList()
        self.thetas_input.append(theta1_input)
        for i in range(1, self.T):
            theta_i = nn.Linear(p, p, bias=bias_fwd)
            self.thetas_input.append(theta_i)

        self.theta0 = nn.Linear(3, p, bias=bias_fwd)

        self.option_relu = True
        self.relu_variant = F.relu

        self.theta_test1_back = nn.Parameter(th.rand(1))
        self.theta_test2_back = nn.Parameter(th.rand(1))
        self.theta_test3_back = nn.Parameter(th.rand(1))
        self.theta_test1_for = nn.Parameter(th.rand(1))
        self.theta_test2_for = nn.Parameter(th.rand(1))
        self.theta_test3_for = nn.Parameter(th.rand(1))

        self.theta_temp1 = nn.Linear(p, p, bias=True)
        self.theta_temp2 = nn.Linear(p, p, bias=True)
        self.theta_temp3 = nn.Linear(p, p, bias=True)
        self.theta_temp4 = nn.Linear(p, p, bias=True)

        self.normalize_backward = False

    def modified_forward(self, linear_layer, vector):
        for i in (range(len(vector.size())-1)):
            vector = vector.transpose(i, i+1)
        new_vector = linear_layer(vector)
        for i in reversed(range(len(new_vector.size())-1)):
            new_vector = new_vector.transpose(i, i+1)
        return new_vector

    def stack_feat(self, rho_prev, zahat, zbhat):
        feature_vector = th.stack((rho_prev, zahat, zbhat, zbhat-zahat), dim=-1)
        return feature_vector

    def update_layers(self, layers):
        self.layers_new = []
        for layer_i in layers:
            if isinstance(layer_i, (nn.ReLU, nn.Linear, nn.Conv2d)):
                self.layers_new.append(layer_i)

    def compute_D(self, bounds):
        # compute D where D_i = \sum_j A_{i,j}
        layers = self.layers_new

        forward_idxes = []
        for idx in range(0, len(bounds)-1):
            forward_idxes.append((idx, 2*(idx)))

        backward_idxes = []
        for idx in range(0, len(bounds)-1):
            backward_idxes.append((idx+1, 2*(idx)))
        backward_idxes.reverse()

        self.forward_sum_vec = [None] * len(forward_idxes)
        self.backward_sum_vec = [None] * len(backward_idxes)

        for rho_idx, layer_idx in forward_idxes:
            cur_layer = layers[layer_idx]
            if isinstance(cur_layer, nn.Conv2d):
                size_ = list(bounds[rho_idx].size())
                size_[0] *= self.p
                tensor_ = th.ones(size_, device=bounds[0].device)
                sum_vec = th.nn.functional.conv2d(tensor_, th.ones_like(cur_layer.weight), bias=None,
                                                  stride=cur_layer.stride, padding=cur_layer.padding,
                                                  groups=cur_layer.groups, dilation=cur_layer.dilation)
                self.forward_sum_vec[rho_idx] = sum_vec

        for rho_idx, layer_idx in backward_idxes:
            cur_layer = layers[layer_idx]
            if isinstance(cur_layer, nn.Conv2d):
                size_ = list(bounds[rho_idx].size())
                size_[0] *= self.p
                tensor_ = th.ones(size_, device=bounds[0].device)
                sum_vec = th.nn.functional.conv_transpose2d(input=tensor_,
                                                            weight=th.ones_like(cur_layer.weight),
                                                            stride=cur_layer.stride, padding=cur_layer.padding,
                                                            output_padding=cur_layer.output_padding,
                                                            groups=cur_layer.groups, dilation=cur_layer.dilation)
                sum_vec = sum_vec.squeeze()
                self.backward_sum_vec[rho_idx-1] = sum_vec
        # print("finished computing D")

    def forward(self, image, lbs, ubs, layers, mu_rho, rhos_prev, zahats, zbhats,
                return_embedding=False, grad_=None, lp_primal=None):

        batch_size = lbs[0].size(0)  # changed

        layers_new = []
        for layer_i in layers:
            if isinstance(layer_i, (nn.ReLU, nn.Linear, nn.Conv2d)):
                layers_new.append(layer_i)
        layers = layers_new

        num_layers = len(layers)
        num_rhos = len(rhos_prev)

        embedding_size = mu_rho[0].size()[-1]

        cpu = self.cpu
        T = self.T
        # eta = 100

        # get the initial embedding vector for the first layer
        if self.feature_grad:
            if self.lp_primal:
                feature_vector = th.stack((image, lbs[0], ubs[0], grad_, lp_primal), dim=-1)
            else:
                feature_vector = th.stack((image, lbs[0], ubs[0], grad_), dim=-1)
        else:
            feature_vector = th.stack((image, lbs[0], ubs[0]), dim=-1)
        # Ninfo (="node info") includes only information about the node
        Ninfo = feature_vector
        for theta_i in self.thetas_input:
            if self.option_relu:
                Ninfo = self.relu_variant(theta_i(Ninfo))
            else:
                Ninfo = theta_i(Ninfo)
        # changing size of Ninfo from batch X layer_dim X embedding_size
        # to  embedding_size X batch X layer_dim
        for i in reversed(range(len(Ninfo.size())-1)):
            Ninfo = Ninfo.transpose(i-1, i)
        mu_rho[0] = Ninfo

        # get initial embedding vectors for all other layers
        for var_idx in range(1, len(mu_rho)):
            # zahat = zahats[var_idx]
            # zbhat = zbhats[var_idx]
            if var_idx == len(mu_rho)-1:
                rho_prev = th.zeros_like(ubs[var_idx])
            else:
                # rho_prev = rhos_prev[var_idx-1][0]
                # TODO check whether it should be 0 or 1 (take the one that corresponds to upper bounds
                rho_prev = rhos_prev[var_idx-1][:, 0]

            # feature_vector = self.stack_feat(rho_prev, zahat, zbhat)
            feature_vector = th.stack((lbs[var_idx], ubs[var_idx], rho_prev), dim=-1)

            Ninfo_new = feature_vector
            for theta_i in self.thetas:
                if self.option_relu:
                    Ninfo_new = self.relu_variant(theta_i(Ninfo_new))
                else:
                    Ninfo_new = theta_i(Ninfo_new)

            # changing size of Ninfo_new from batch X layer_dim X embedding_size
            # to  embedding_size X batch X layer_dim
            for i in reversed(range(len(Ninfo_new.size())-1)):
                Ninfo_new = Ninfo_new.transpose(i-1, i)

            mu_rho[var_idx] = Ninfo_new

        if self.message_passing:
            forward_idxes = []
            for idx in range(0, len(mu_rho)-1):
                # forward_idxes.append((idx, 2*(idx+1)))
                forward_idxes.append((idx, 2*(idx)))  # new

            backward_idxes = []
            for idx in range(0, len(mu_rho)-1):
                # backward_idxes.append((idx+1, 2*(idx+1)))
                backward_idxes.append((idx+1, 2*(idx)))  # new
                # TODO backward_idxes.append((idx+1, idx+1))
            backward_idxes.reverse()

            for rho_idx, layer_idx in forward_idxes:
                layer_mu_prev = mu_rho[rho_idx]

                size_ = layer_mu_prev.size()
                size2_ = mu_rho[rho_idx+1].size()

                # combine the first two layers corresponding to embedding_dimension and batch_dimension
                layer_mu_prev_view = layer_mu_prev.reshape((-1,) + size_[2:])  # new

                cur_layer = layers[layer_idx]
                size = cur_layer.weight.size()

                # Embed_sum is the sum of all embedding vectors in the previous layers
                # that are connected to the current node
                # Embed_pass is the embedding vectors passed through the network
                # Ninfo is the part of the embedding vector that only depends on the node's feature vector
                # sum_vec is the number of nodes in the following layer that every node is connected to -
                # used for normalization
                if isinstance(cur_layer, nn.Linear):
                    sum_mu = th.sum(layer_mu_prev.view(embedding_size, batch_size, -1), -1)  # new
                    sum_mu = sum_mu / (size[0])
                    Embed = self.modified_forward(self.theta_temp1, sum_mu)
                    layer_len = size[0]
                    # Embed_sum = th.stack(layer_len*[Embed], dim=-1)  #old implementation (slow)
                    size_repeat = len(Embed.size())*[1] + [layer_len]
                    Embed_sum = Embed.unsqueeze(-1).repeat(size_repeat)

                    prev_mu_repeated = layer_mu_prev_view.view(embedding_size*batch_size, -1)  # new
                    forward = (th.matmul(cur_layer.weight, prev_mu_repeated.t()).t()
                               + cur_layer.bias.repeat(batch_size*embedding_size, 1))  # new
                    # forward = (th.bmm(cur_layer.weight, prev_mu_repeated.t()).t()
                    #           + cur_layer.bias.repeat(batch_size*embedding_size, 1))  # new
                    Embed_pass = self.modified_forward(self.theta_temp2, forward.view(size2_))

                elif isinstance(cur_layer, nn.Conv2d):
                    mu_prev = layer_mu_prev_view

                    Embed_sum2 = th.nn.functional.conv2d(mu_prev, th.ones_like(cur_layer.weight), bias=None,
                                                         stride=cur_layer.stride, padding=cur_layer.padding,
                                                         groups=cur_layer.groups, dilation=cur_layer.dilation)

                    sum_vec2 = self.forward_sum_vec[rho_idx]

                    # sum_vec2 = th.nn.functional.conv2d(th.ones_like(mu_prev), th.ones_like(cur_layer.weight), bias=None,
                    #                                    stride=cur_layer.stride, padding=cur_layer.padding,
                    #                                    groups=cur_layer.groups, dilation=cur_layer.dilation)
                    # sum_vec_new = self.forward_sum_vec[rho_idx]
                    # assert((sum_vec2 - sum_vec_new).abs().max() < 1e-7), ((sum_vec2 - sum_vec_new).abs().max())

                    Embed_sum = (Embed_sum2 / sum_vec2)

                    Embed_sum = self.modified_forward(self.theta_temp1, Embed_sum.view(size2_))

                    forward = cur_layer.forward(mu_prev)
                    Embed_pass = self.modified_forward(self.theta_temp2, forward.view(size2_))

                else:
                    input("problem in graph_outer_minibatch")
                Ninfo_new = mu_rho[rho_idx+1]

                mu_rho[rho_idx+1] = self.relu_variant(self.theta_test1_for.expand_as(Ninfo_new) * Ninfo_new
                                                      + self.theta_test2_for.expand_as(Ninfo_new) * Embed_pass
                                                      + self.theta_test3_for.expand_as(Ninfo_new) * Embed_sum)

            for rho_idx, layer_idx in backward_idxes:
                layer_mu_prev = mu_rho[rho_idx]

                size_ = layer_mu_prev.size()
                size2_ = mu_rho[rho_idx-1].size()

                layer_mu_prev_view = layer_mu_prev.reshape((-1,) + size_[2:])  # new

                cur_layer = layers[layer_idx]

                # Embed_sum is the sum of all embedding vectors in the previous layers that are connected
                # backward is the embedding vectors passed through the network
                if isinstance(cur_layer, nn.Linear):
                    size = cur_layer.weight.t().size()
                    sum_mu = th.sum(layer_mu_prev.view(embedding_size, batch_size, -1), -1)
                    Embed = self.modified_forward(self.theta_temp3, sum_mu)

                    layer_len = size[0]
                    # Embed_sum = th.stack(layer_len*[Embed], dim=-1) #old implementation (slow)
                    size_repeat = len(Embed.size())*[1] + [layer_len]
                    Embed_sum = Embed.unsqueeze(-1).repeat(size_repeat)

                    Embed_sum = Embed_sum / (size[0])

                    prev_mu_repeated = layer_mu_prev.reshape(embedding_size*batch_size, -1)

                    backward = th.mm(cur_layer.weight.t(), (prev_mu_repeated
                                     - cur_layer.bias.repeat(batch_size * embedding_size, 1)).t()).t()
                    # backward2 = th.mm(prev_mu_repeated - cur_layer.bias.repeat(batch_size * embedding_size, 1),
                    #                   cur_layer.weight)
                    # print((backward-backward2).abs().max())
                    # print((prev_mu_repeated - cur_layer.bias.repeat(batch_size * embedding_size, 1)).size(),
                    #       cur_layer.weight.size())
                    # backward = th.bmm(cur_layer.weight.t(), (prev_mu_repeated
                    #                 - cur_layer.bias.repeat(batch_size * embedding_size, 1)).t()).t()

                    Embed_pass = self.modified_forward(self.theta_temp4, backward.view(size2_))

                elif isinstance(cur_layer, nn.Conv2d):

                    mu_prev = layer_mu_prev_view

                    sum_vec = self.backward_sum_vec[rho_idx - 1]

                    # sum_vec = th.nn.functional.conv_transpose2d(input=th.ones_like(mu_prev),
                    #                                             weight=th.ones_like(cur_layer.weight),
                    #                                             stride=cur_layer.stride, padding=cur_layer.padding,
                    #                                             output_padding=cur_layer.output_padding,
                    #                                             groups=cur_layer.groups, dilation=cur_layer.dilation)
                    # sum_vec = sum_vec.squeeze()
                    # sum_vec_new = self.backward_sum_vec[rho_idx - 1]
                    # assert((sum_vec - sum_vec_new).abs().max() < 1e-7), ((sum_vec - sum_vec_new).abs().max())

                    Embed_sum = th.nn.functional.conv_transpose2d(input=mu_prev, weight=th.ones_like(cur_layer.weight),
                                                                  stride=cur_layer.stride, padding=cur_layer.padding,
                                                                  output_padding=cur_layer.output_padding,
                                                                  groups=cur_layer.groups, dilation=cur_layer.dilation)
                    Embed_sum = Embed_sum.squeeze()
                    Embed_sum = Embed_sum / sum_vec

                    Embed_sum = self.modified_forward(self.theta_temp3, Embed_sum.view(size2_))

                    backward = th.nn.functional.conv_transpose2d(input=mu_prev, weight=cur_layer.weight,
                                                                 stride=cur_layer.stride, padding=cur_layer.padding,
                                                                 output_padding=cur_layer.output_padding,
                                                                 groups=cur_layer.groups, dilation=cur_layer.dilation)
                    backward = backward.squeeze()

                    Embed_pass = self.modified_forward(self.theta_temp4, backward.view(size2_))

                else:
                    input("problem in graph_outer_minibatch")

                Ninfo_new = mu_rho[rho_idx-1]
                size_new = Ninfo_new.size()
                if self.normalize_backward:
                    # coule maybe normalize over embedding dim as well?
                    sum_before = mu_rho[rho_idx-1].transpose(0,1).reshape(batch_size, -1).abs().sum(axis=1)
                mu_rho[rho_idx-1] = (self.relu_variant(self.theta_test1_back.expand_as(Ninfo_new) * Ninfo_new
                                     + self.theta_test2_back.expand_as(Ninfo_new) * Embed_pass.view(size_new)
                                     + self.theta_test3_back.expand_as(Ninfo_new) * Embed_sum.view(size_new)))
                if self.normalize_backward:
                    sum_after = mu_rho[rho_idx-1].transpose(0,1).reshape(batch_size, -1).abs().sum(axis=1)

                    norm_ = sum_before/sum_after
                    for idx_norm in range(batch_size):
                        mu_rho[rho_idx-1][:, idx_norm, :] *= norm_[idx_norm]
                    assert((mu_rho[rho_idx-1]!=mu_rho[rho_idx-1]).sum()==0), ("after", norm_.min(), norm_.max())

        if return_embedding:
            return mu_rho, Ninfo_new
        else:
            return mu_rho


class EmbedUpdates(nn.Module):
    '''
    this class updates embeding vectors from t=1 and t=T
    '''

    def __init__(self, T, p, args):

        super(EmbedUpdates, self).__init__()
        self.T = T
        self.p = p
        self.normalize_mu = False

        self.update = EmbedLayerUpdate(T, p, args)

    def normalize_mu_new(self, mu):
        mu_new = []
        for mu_i in mu:
            if len(mu_i.size()) == 5:
                avg_mu = th.cat(32*[mu_i.sum(dim=-1).unsqueeze(0)]).permute(1, 2, 3, 4, 0)
            elif len(mu_i.size()) == 4:
                avg_mu = th.cat(32*[mu_i.sum(dim=-1).unsqueeze(0)]).permute(1, 2, 3, 0)
            elif len(mu_i.size()) == 3:
                avg_mu = th.cat(32*[mu_i.sum(dim=-1).unsqueeze(0)]).permute(1, 2, 0)
            mu_new.append(mu_i/avg_mu)
        return mu_new

    def forward(self, image, lbs, ubs, layers, rhos, zahats, zbhats,
                return_embedding=False, grad_=None, lp_primal=None):
        mu_rho = init_mu(self.p, lbs)

        mu_rho = self.update(image, lbs, ubs, layers, mu_rho, rhos, zahats, zbhats,
                             return_embedding=return_embedding, grad_=grad_, lp_primal=lp_primal)
        if self.normalize_mu:
            mu_rho = self.normalize_mu_new(mu_rho)
        return mu_rho


class ComputeFinalScore(nn.Module):
    '''
    this class takes the first embedding vector and turns it into a input vector
    corresponding to an image and possible an adversarial example

    p: the dimension of embedding vectors at the final stage
    '''
    def __init__(self, p):
        super(ComputeFinalScore, self).__init__()

        self.theta5 = nn.Linear(p, 1)

    def modified_forward(self, linear_layer, vector):
        for i in (range(len(vector.size())-1)):
            vector = vector.transpose(i, i+1)
        new_vector = linear_layer(vector)
        for i in reversed(range(len(new_vector.size())-1)):
            new_vector = new_vector.transpose(i, i+1)
        return new_vector

    def forward(self, mu):
        new_image = self.modified_forward(self.theta5, mu[0]).squeeze(dim=0)

        return new_image


class GraphNet(nn.Module):
    def __init__(self, T, p, args):
        super(GraphNet, self).__init__()
        self.EmbedUpdates = EmbedUpdates(T, p, args)
        self.ComputeFinalScore = ComputeFinalScore(p)
        self.p = p
        self.mu_norm_rho = 0
        self.mu = 0

    def forward(self, image, lbs_all, ubs_all, layers, prev_rhos, zahats, zbhats, grad_=None, lp_primal=None):
        mu_rho = self.EmbedUpdates(image, lbs_all, ubs_all, layers,
                                   prev_rhos, zahats, zbhats,
                                   grad_=grad_, lp_primal=lp_primal)

        scores_rho = self.ComputeFinalScore(mu_rho)

        return scores_rho

    def embedding_norm(self):
        mu_test = th.zeros(1)
        mu_test[0] = self.mu_norm
        return mu_test

    def return_embedding_vectors(self, lower_bounds_all, upper_bounds_all, layers,
                                 prev_rhos, zahats, zbhats):
        mu_final, mu_mlp = self.EmbedUpdates(lower_bounds_all, upper_bounds_all, layers,
                                             prev_rhos, zahats, zbhats, return_embedding=True)

        return mu_final, mu_mlp

    def update_layers(self, *args):
        self.EmbedUpdates.update.update_layers(*args)

    def compute_D(self, bounds):
        self.EmbedUpdates.update.compute_D(bounds)


def init_mu(p, bounds):
    mu_rhos = []
    for lb_i in bounds:
        # newfull returns a new tensor that has the same dtype and device as bounds[0]
        mu_current_layer = bounds[0].new_full((lb_i.size()+(p,)), fill_value=0.0)
        mu_rhos.append(mu_current_layer)
    return mu_rhos
