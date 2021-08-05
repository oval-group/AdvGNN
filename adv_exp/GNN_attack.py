import os
import torch
import copy
from torch import nn
from lp_solver.by_pairs import ByPairsDecomposition
import torch.distributions as dist

from adv_exp.attack_class import Attack_Class
from adv_exp.GNN_SAT_optimized import GraphNet as GNN_SAT_optimized
from adv_exp.utils import load_model_only
from adv_exp.adam import Adam_Manual, AdamOptimizer

########################################################################
#   implements AdvGNN, the GNN-based attack.
#   The code for the GNN itself is in GNN_SAT_optimized.py
#   here we call the GNN and use the values it returns
########################################################################

default_params = {
    'nb_steps': 100,
    'T': 1,
    'p': 32,
    'cpu': False,
    'initial_step_size': 1e-2,
    'final_step_size': 1e-3,
    'num_adv_ex': 10,
    'feature_grad': False,
    'lp_primal': True,
    'pick_inits': None,
    'lin_decay': True,
    'momentum': None,
    'adam': False,
    'GNN_optimized': True,
}


class GNN_attack(Attack_Class):

    def __init__(self, layers=None, params=None, cpu=False, store_loss_progress=False):
        self.__name__ = 'GNN_attack'

        if layers:
            self.set_layers(layers)

        self.params = dict(default_params, **params) if params is not None else default_params
        print(f"init step {self.params['initial_step_size']}, final: {self.params['final_step_size']}")

        self.store_loss_progress = store_loss_progress
        # Store dict of lists of tensors containing the progress in the bounds with the inner iters.
        self.bounds_progress_per_layer = {}

        print("update self.cpu in GNN_Bounding use params?")
        self.cpu = cpu

        self.decomposition = ByPairsDecomposition('KW')

        if self.params['GNN_optimized']:
            self.GNN = GNN_SAT_optimized(self.params['T'], self.params['p'], self.params)
        else:
            self.GNN = GNN_SAT(self.params['T'], self.params['p'], self.params)

        if (torch.cuda.is_available() and self.cpu is False):
            self.GNN.cuda()
        folder_ = './adv_exp/GNNs/'
        if self.params['load_GNN']:
            print(f"{self.params['load_GNN']} has to be not None")
            load_model_only(self.GNN, os.path.join(folder_, self.params['load_GNN']), self.cpu)

    def _reduce_batch_size(self, list_, inds):
        if isinstance(list_, list):
            return [l[inds] for l in list_]
        else:
            return list_[inds]

    def eval_mode(self):
        for params in self.GNN.parameters():
            params.requires_grad = False

    def set_layers(self, layers, lbs_all=None):
        super().set_layers(layers)
        if self.params['GNN_optimized']:
            self.GNN.update_layers(self.layers)
            self.GNN.compute_D(lbs_all)

    def create_adv_examples(self, data, model, return_criterion="all", init_tensor=None, target=None, gpu=False,
                            lbs_all=None, ubs_all=None, dual_vars=None, return_iters=False, lp_primal=None,
                            return_if_no_progress=False):
        assert return_criterion in ["one", "half", "all", "not_early"]
        self.targeted_attack = not isinstance(target, type(None))

        x, y, x_lbs, x_ubs = data
        if gpu and torch.cuda.is_available():
            x = x.cuda()
            x_lbs = x_lbs.cuda()
            x_ubs = x_ubs.cuda()
            model.cuda()
        device = x.device

        iters = self.params['iters']
        num_adv = self.params['num_adv_ex']
        initial_step_size = self.params['initial_step_size']
        final_step_size = self.params['final_step_size']
        alpha = 1e-5

        if device.type == 'cpu':
            labels = torch.LongTensor([y]*num_adv, device=device)
        else:
            labels = torch.cuda.LongTensor([y]*num_adv, device=device)

        # Calculate the mean of the normal distribution in logit space
        prior = dist.Uniform(low=x_lbs, high=x_ubs)
        images = prior.sample(torch.Size([num_adv]))

        if not isinstance(init_tensor, type(None)):
            if images[0].size() == init_tensor.size():
                images[0] = init_tensor
            elif images[0].size() == init_tensor[0].size():
                images = init_tensor
            else:
                print("image size", images.size(), images[0].size())
                print("init tensor size", init_tensor.size(), init_tensor[0].size())
                input("images and init tensor not compatible")

        # Reduce the batch size - pick the 10 best init tensors
        if self.params['pick_inits']:
            outputs = model(images)
            initial_score = -outputs[:, labels[0]] + outputs[:, target]
            idces = torch.sort(initial_score, descending=True).indices[:self.params['pick_inits']]
            images = self._reduce_batch_size(images, idces)
            lbs_all = self._reduce_batch_size(lbs_all, idces)
            ubs_all = self._reduce_batch_size(ubs_all, idces)
            dual_vars = self._reduce_batch_size(dual_vars, idces)

        if not isinstance(target, type(None)):
            self.loss_type = 'targeted_loss'
        else:
            self.loss_type = 'CE_loss'
            self.CE_loss = nn.CrossEntropyLoss()
        loss = nn.CrossEntropyLoss()

        self.loss_progress = []
        if return_if_no_progress:
            self.loss_progress_vec = []
            would_have_stopped = []
            no_progress_last_iter = False

        outputs = model(images)

        if self.params['adam']:
            adam_opt2 = Adam_Manual(images)

        for i in range(iters):
            if not self.params['feature_grad']:
                # AdvGNN version that doesn't use gradients.
                # This is not inclduded in the paper as it doesn't work well
                # descent direction
                desc_dir = self.GNN(images, lbs_all, ubs_all,
                                    self.layers, dual_vars,
                                    None, None)

                # compute decaying stepsize
                step_size = initial_step_size + (i / iters) * (final_step_size - initial_step_size)

                adv_images = images + step_size*desc_dir

                images = torch.max(torch.min(adv_images, x_ubs), x_lbs)

                outputs = model(images)
                cost = self._loss(outputs, labels, target).to(device)

            elif self.params['feature_grad']:
                # compute the current gradient
                # don't need to clone as i'm not changing images while i need images_grad
                images_grad = images.clone().detach()
                with torch.enable_grad():
                    images_grad.requires_grad = True
                    outputs = model(images_grad)

                    model.zero_grad()
                    cost = self._loss(outputs, labels, target).to(device)
                    cost.backward()

                    grad_ = images_grad.grad
                    images_grad = images_grad.detach()

                # call the GNN to compute a new direction of movement
                if self.params['lp_primal']:
                    desc_dir = self.GNN(images, lbs_all, ubs_all,
                                        self.layers, dual_vars,
                                        # primal_vars.zahats, primal_vars.zbhats)
                                        None, None, grad_=grad_.sign(), lp_primal=lp_primal)
                else:
                    desc_dir = self.GNN(images, lbs_all, ubs_all,
                                        self.layers, dual_vars,
                                        # primal_vars.zahats, primal_vars.zbhats)
                                        None, None, grad_=grad_.sign())

                # compute decaying stepsize
                if self.params['lin_decay']:
                    # linearly decaying stepsize
                    step_size = initial_step_size + (i / iters) * (final_step_size - initial_step_size)
                else:
                    # exponentially decaying stepsize
                    step_size = initial_step_size * ((final_step_size/initial_step_size)**(i/iters))

                if self.params['momentum']:
                    # implement momentum
                    if i == 0:
                        velocity = desc_dir
                    else:
                        mom = self.params['momentum']
                        velocity = mom * velocity + (1-mom) * desc_dir
                    desc_dir = velocity

                # take a step using desc_dir (EQUATION (14))
                if self.params['adam']:
                    adam_opt2(images, -desc_dir, step_size)
                    adv_images = images
                else:
                    adv_images = images + step_size*desc_dir

                # project current point onto feasible region (EQUATION (14))
                images = torch.max(torch.min(adv_images, x_ubs), x_lbs)

                # return early if we haven't made enough progress (only if the flag is set to True)
                if return_if_no_progress:
                    # don't need the next 2 lines
                    outputs = model(images)
                    cost, cost_vec = self._loss(outputs, labels, target, return_vector=True)
                    if i < 2:
                        self.loss_progress_vec.append(cost_vec)
                    else:
                        max_idx = cost_vec.argmax()
                        loss_now = cost_vec[max_idx]
                        prev_loss = self.loss_progress_vec[-1][max_idx]
                        prev_prev_loss = self.loss_progress_vec[-2][max_idx]
                        iters_left = iters - i - 1
                        eps = 1  # 0.9 0.99
                        # check whether the improvement made is more than the required rate of improvement needed
                        not_enough_progress = (loss_now - prev_loss < (-prev_loss/(iters_left + 1))*eps
                                               and
                                               loss_now - prev_prev_loss < (-prev_prev_loss/((iters_left + 2)/2))*eps)
                        if not_enough_progress and no_progress_last_iter:
                            break
                            # would_have_stopped.append(i)
                        self.loss_progress_vec[-2] = self.loss_progress_vec[-1]
                        self.loss_progress_vec[-1] = cost_vec
                        no_progress_last_iter = not_enough_progress

                else:
                    # don't need the next 2 lines
                    outputs = model(images)
                    cost = self._loss(outputs, labels, target).to(device)

            # store the all points, mainly needed during training
            if self.store_loss_progress:
                self.loss_progress.append(cost)

            if i % 1 == 0 and return_criterion != 'not_early':
                succ, sum_, mean_ = self.success_tensor(outputs, y, target)
                if return_criterion == "all" and mean_ == 1:
                    break
                elif return_criterion == "one" and mean_ > 0:
                    # we have found an adversarial example so we stop
                    if return_if_no_progress and len(would_have_stopped):
                        print(f"\n\nwould would_have_stopped {would_have_stopped} \n\n")
                    break
                elif return_criterion == "half" and mean_ >= 0.5:
                    break

        succ, sum_, mean_ = self.success_tensor(outputs, y, target)

        if return_iters:
            return images, succ, i
        else:
            return images, succ
