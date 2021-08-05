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


########################################################################
#   implements the GNN for AdvGNN
#   implements Equations (8)-(13)
########################################################################

class EmbedLayerUpdate(nn.Module):
    '''
    this class updates embeded layer one time
    '''
    def __init__(self, T, p, args):
        super(EmbedLayerUpdate, self).__init__()
        '''
        p: embedding size
        T: T1 in the paper is the depth of the MLP g
        '''
        self.mode_linear = True
        self.message_passing = True

        self.p = p
        self.T = T
        self.cpu = args['cpu']

        self.feature_grad = args['feature_grad']
        self.lp_primal = args['lp_primal']

        self.normalize_conv = False

        feature_fwd = 3
        bias_fwd = True

        theta1 = nn.Linear(feature_fwd, p, bias=bias_fwd)

        # self.thetas implements the function g: R^p -> R^d
        self.thetas = nn.ModuleList()
        self.thetas.append(theta1)
        for i in range(1, self.T):
            theta_i = nn.Linear(p, p, bias=bias_fwd)
            self.thetas.append(theta_i)

        feature_fwd_input = 3
        if self.feature_grad:
            feature_fwd_input += 1
        if self.lp_primal:
            feature_fwd_input += 1

        # self.thetas_input implements the function g: R^p -> R^d used for the input layer
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
        # implements a normal forward pass but including temporary resizing
        for i in (range(len(vector.size())-1)):
            vector = vector.transpose(i, i+1)
        new_vector = linear_layer(vector)
        for i in reversed(range(len(new_vector.size())-1)):
            new_vector = new_vector.transpose(i, i+1)
        return new_vector

    def update_layers(self, layers):
        # filters the layers, only keeps ReLU, linear, and Convolutional layers
        self.layers_new = []
        for layer_i in layers:
            if isinstance(layer_i, (nn.ReLU, nn.Linear, nn.Conv2d)):
                self.layers_new.append(layer_i)

    def compute_D(self, bounds):
        # computes the normalization matrices Q,and  Q' used in equations (11) and (12).
        # Q, and Q' are matrices whose elements are the number of neighbouring nodes in the
        #   previous and following layer respectively for each node.
        layers = self.layers_new

        forward_idxes = []
        for idx in range(0, len(bounds)-1):
            forward_idxes.append((idx, 2*(idx)))

        backward_idxes = []
        for idx in range(0, len(bounds)-1):
            backward_idxes.append((idx+1, 2*(idx)))
        backward_idxes.reverse()

        self.forward_sum_vec = [None] * len(forward_idxes)  # called Q in the paper
        self.backward_sum_vec = [None] * len(backward_idxes)  # called Q' in the paper

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

    def forward(self, image, lbs, ubs, layers, mu_rho, rhos_prev, zahats, zbhats,
                return_embedding=False, grad_=None, lp_primal=None):

        batch_size = lbs[0].size(0)

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

        # compute the feature vector for the input layer (f_0)
        if self.feature_grad:
            if self.lp_primal:
                feature_vector = th.stack((image, lbs[0], ubs[0], grad_, lp_primal), dim=-1)
            else:
                feature_vector = th.stack((image, lbs[0], ubs[0], grad_), dim=-1)
        else:
            feature_vector = th.stack((image, lbs[0], ubs[0]), dim=-1)

        # compute the initial embedding vector for the input layer (Equation (8))
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

        # get feature vectors for all other layers (f_1, ..., f_{L-1})
        for var_idx in range(1, len(mu_rho)):
            if var_idx == len(mu_rho)-1:
                rho_prev = th.zeros_like(ubs[var_idx])
            else:
                rho_prev = rhos_prev[var_idx-1][:, 0]

            feature_vector = th.stack((lbs[var_idx], ubs[var_idx], rho_prev), dim=-1)

            # Compute the initial embedding vector for all other layers (Equation (8))
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
            # implement the forward and backward passes (Equations (11) and (12))
            forward_idxes = []
            for idx in range(0, len(mu_rho)-1):
                forward_idxes.append((idx, 2*(idx)))

            backward_idxes = []
            for idx in range(0, len(mu_rho)-1):
                backward_idxes.append((idx+1, 2*(idx)))
            backward_idxes.reverse()

            # Forward pass (Equation (11))
            for rho_idx, layer_idx in forward_idxes:
                layer_mu_prev = mu_rho[rho_idx]

                size_ = layer_mu_prev.size()
                size2_ = mu_rho[rho_idx+1].size()

                # combine the first two layers corresponding to embedding_dimension and batch_dimension
                layer_mu_prev_view = layer_mu_prev.reshape((-1,) + size_[2:])

                cur_layer = layers[layer_idx]
                size = cur_layer.weight.size()

                # Embed_sum is the sum of all embedding vectors in the previous layers
                # that are connected to the current node
                # Embed_pass is the embedding vectors passed through the network
                # Ninfo is the part of the embedding vector that only depends on the node's feature vector
                # sum_vec is the number of nodes in the following layer that every node is connected to -
                # used for normalization
                if isinstance(cur_layer, nn.Linear):
                    sum_mu = th.sum(layer_mu_prev.view(embedding_size, batch_size, -1), -1)
                    sum_mu = sum_mu / (size[0])
                    Embed = self.modified_forward(self.theta_temp1, sum_mu)
                    layer_len = size[0]
                    size_repeat = len(Embed.size())*[1] + [layer_len]
                    Embed_sum = Embed.unsqueeze(-1).repeat(size_repeat)

                    prev_mu_repeated = layer_mu_prev_view.view(embedding_size*batch_size, -1)
                    forward = (th.matmul(cur_layer.weight, prev_mu_repeated.t()).t()
                               + cur_layer.bias.repeat(batch_size*embedding_size, 1))

                    Embed_pass = self.modified_forward(self.theta_temp2, forward.view(size2_))

                elif isinstance(cur_layer, nn.Conv2d):
                    mu_prev = layer_mu_prev_view

                    Embed_sum2 = th.nn.functional.conv2d(mu_prev, th.ones_like(cur_layer.weight), bias=None,
                                                         stride=cur_layer.stride, padding=cur_layer.padding,
                                                         groups=cur_layer.groups, dilation=cur_layer.dilation)

                    sum_vec2 = self.forward_sum_vec[rho_idx]

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

            # Forward pass (Equation (12))
            for rho_idx, layer_idx in backward_idxes:
                layer_mu_prev = mu_rho[rho_idx]

                size_ = layer_mu_prev.size()
                size2_ = mu_rho[rho_idx-1].size()

                layer_mu_prev_view = layer_mu_prev.reshape((-1,) + size_[2:])

                cur_layer = layers[layer_idx]

                # Embed_sum is the sum of all embedding vectors in the previous layers that are connected
                # backward is the embedding vectors passed through the network
                if isinstance(cur_layer, nn.Linear):
                    size = cur_layer.weight.t().size()
                    sum_mu = th.sum(layer_mu_prev.view(embedding_size, batch_size, -1), -1)
                    Embed = self.modified_forward(self.theta_temp3, sum_mu)

                    layer_len = size[0]
                    size_repeat = len(Embed.size())*[1] + [layer_len]
                    Embed_sum = Embed.unsqueeze(-1).repeat(size_repeat)

                    Embed_sum = Embed_sum / (size[0])

                    prev_mu_repeated = layer_mu_prev.reshape(embedding_size*batch_size, -1)

                    backward = th.mm(cur_layer.weight.t(), (prev_mu_repeated
                                     - cur_layer.bias.repeat(batch_size * embedding_size, 1)).t()).t()

                    Embed_pass = self.modified_forward(self.theta_temp4, backward.view(size2_))

                elif isinstance(cur_layer, nn.Conv2d):

                    mu_prev = layer_mu_prev_view

                    sum_vec = self.backward_sum_vec[rho_idx - 1]

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
                    sum_before = mu_rho[rho_idx-1].transpose(0, 1).reshape(batch_size, -1).abs().sum(axis=1)
                mu_rho[rho_idx-1] = (self.relu_variant(self.theta_test1_back.expand_as(Ninfo_new) * Ninfo_new
                                     + self.theta_test2_back.expand_as(Ninfo_new) * Embed_pass.view(size_new)
                                     + self.theta_test3_back.expand_as(Ninfo_new) * Embed_sum.view(size_new)))
                if self.normalize_backward:
                    sum_after = mu_rho[rho_idx-1].transpose(0, 1).reshape(batch_size, -1).abs().sum(axis=1)

                    norm_ = sum_before/sum_after
                    for idx_norm in range(batch_size):
                        mu_rho[rho_idx-1][:, idx_norm, :] *= norm_[idx_norm]
                    assert((mu_rho[rho_idx-1] != mu_rho[rho_idx-1]).sum() == 0), ("after", norm_.min(), norm_.max())

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
        self.p = p
        self.update = EmbedLayerUpdate(T, p, args)

    def forward(self, image, lbs, ubs, layers, rhos, zahats, zbhats,
                return_embedding=False, grad_=None, lp_primal=None):
        mu_rho = init_mu(self.p, lbs)

        mu_rho = self.update(image, lbs, ubs, layers, mu_rho, rhos, zahats, zbhats,
                             return_embedding=return_embedding, grad_=grad_, lp_primal=lp_primal)

        return mu_rho


class ComputeFinalScore(nn.Module):
    '''
    this class takes the embedding vectors of the input layer (mu[0]) and turns them into a new direction of movement
        thus implementing Equation (13)

    p: the dimension of embedding vectors at the final stage
    '''
    def __init__(self, p):
        super(ComputeFinalScore, self).__init__()
        # this is called theta_out in the paper
        self.theta5 = nn.Linear(p, 1)

    def modified_forward(self, linear_layer, vector):
        for i in (range(len(vector.size())-1)):
            vector = vector.transpose(i, i+1)
        new_vector = linear_layer(vector)
        for i in reversed(range(len(new_vector.size())-1)):
            new_vector = new_vector.transpose(i, i+1)
        return new_vector

    def forward(self, mu):
        # implements Equation (13)
        new_image = self.modified_forward(self.theta5, mu[0]).squeeze(dim=0)

        return new_image


class GraphNet(nn.Module):
    def __init__(self, T, p, args):
        super(GraphNet, self).__init__()
        self.EmbedUpdates = EmbedUpdates(T, p, args)
        self.ComputeFinalScore = ComputeFinalScore(p)

    def forward(self, image, lbs_all, ubs_all, layers, prev_rhos, zahats, zbhats, grad_=None, lp_primal=None):
        # image: current best estimate of an adversarial example
        # lbs_all, ubs_all: bounds for the neural network we are attacking based on the input domain
        # layers: neural network we are attacking
        # prev_rhos: dual variables returned an lp-solver
        # zahats, and zbhats are None for all UAI experiments
        # grad_: current gradient
        # lp_primal: primal solution of the lp
        mu_rho = self.EmbedUpdates(image, lbs_all, ubs_all, layers,
                                   prev_rhos, zahats, zbhats,
                                   grad_=grad_, lp_primal=lp_primal)

        scores_rho = self.ComputeFinalScore(mu_rho)

        return scores_rho

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
