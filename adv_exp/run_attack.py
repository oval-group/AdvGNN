# MIT License
#
# Copyright (c) 2018, University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torchvision
import argparse
import os
import time
import math
import copy
import numpy as np
import pandas as pd

from tools.model_utils import load_cifar_1to1_exp, load_1to1_eth
from adv_exp.cli import get_args
from adv_exp.utils import set_seed
from adv_exp.GNN_attack import GNN_attack
from adv_exp.pgd_attack import Pgd_Attack
from adv_exp.mi_fgsm_attack import MI_FGSM_Attack
from lp_solver.solver import SaddleLP


#######################################################################################################
#   Main code to run adversarial attacks. It supports AdvGNN, PGD, MI-FGSM, and CW
#   TODO:
#       add a few more comments
#######################################################################################################


def _run_lp_as_init(verif_layers, domain, args):
    decision_bound = 0

    if not args.cpu and torch.cuda.is_available():
        cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
        domain = domain.cuda()
    else:
        cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

    # use best of naive interval propagation and KW as intermediate bounds
    intermediate_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
    intermediate_net.set_solution_optimizer('best_naive_kw', None)

    bounds_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True,
                          max_batch=args.max_solver_batch, store_duals=True)
    bounds_net.set_decomposition('pairs', 'KW')
    adam_params = {
        'nb_steps': 100,
        'initial_step_size': 1e-2,
        'final_step_size': 1e-3,
        'betas': (0.9, 0.999),
        'outer_cutoff': 0,
        'log_values': False
    }
    bounds_net.set_solution_optimizer('adam', adam_params)
    # bounds_net.set_solution_optimizer('best_naive_kw', None)
    if args.lp_type == "adam":
        bounds_net.set_solution_optimizer('adam', adam_params)
    elif args.lp_type == "naive_KW":
        bounds_net.set_solution_optimizer('best_naive_kw', None)
    else:
        input("lp type not implemented")

    # do initial computation for the network as it is (batch of size 1: there is only one domain)
    # get intermediate bounds
    intermediate_net.define_linear_approximation(domain.unsqueeze(0))
    intermediate_lbs = copy.deepcopy(intermediate_net.lower_bounds)
    intermediate_ubs = copy.deepcopy(intermediate_net.upper_bounds)

    # compute last layer bounds with a more expensive network
    bounds_net.build_model_using_bounds(domain.unsqueeze(0), (intermediate_lbs, intermediate_ubs))

    global_lb, global_ub = bounds_net.compute_lower_bound(counterexample_verification=True)
    intermediate_lbs[-1] = global_lb
    intermediate_ubs[-1] = global_ub
    bounds_net_device = global_lb.device
    intermediate_net_device = domain.device

    # retrieve bounds info from the bounds network
    global_ub_point = bounds_net.get_lower_bound_network_input()
    global_ub = bounds_net.net(global_ub_point)

    duals_ = bounds_net.get_dual_vars().rhos

    return global_ub_point, global_ub, intermediate_lbs, intermediate_ubs, duals_


def run_pgd_attack(args):

    # # initialize a file to record all results, record should be a pandas dataframe
    path = './datasets/CIFAR10/SAT/'
    result_path = './cifar_results/'

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    record_name = result_path + args.table_name

    if not os.path.exists(os.path.dirname(record_name)):
        os.makedirs(os.path.dirname(record_name))

    # # load all properties
    gt_results = pd.read_pickle(path + args.pdprops)
    bnb_ids = gt_results.index
    batch_ids = bnb_ids
    enum_batch_ids = enumerate(batch_ids)

    if os.path.isfile(record_name):
        graph_df = pd.read_pickle(record_name)
    else:
        _columns = ['Idx', 'Eps', 'prop', 'BSAT', 'BTime_PGD', 'method', 'PGD_lr', 'PGD_steps',
                    'restarts', 'batch_size', 'PGD_max_steps', 'run_lp']
        graph_df = pd.DataFrame(index=bnb_ids, columns=_columns)
        graph_df.experiment_details = args

    if args.adv_method == 'pgd_attack':
        # load the pgd model
        adv_params = {
            'iters': args.pgd_iters,
            'optimizer': args.pgd_optimizer,
            'lr': args.pgd_optimizer_lr,
            'num_adv_ex': args.count_particles,
            'check_adv': int(args.check_adv),
        }
        adv_model = Pgd_Attack(adv_params)
    if args.adv_method == 'a_pgd_attack':
        from adv_exp.apgd_attack import A_Pgd_Attack
        # load the pgd model
        adv_params = {
            'iters': args.pgd_iters,
            'optimizer': args.pgd_optimizer,
            'lr': args.pgd_optimizer_lr,
            'num_adv_ex': args.count_particles,
            'check_adv': int(args.check_adv),
        }
        adv_model = A_Pgd_Attack(adv_params)
    if args.adv_method == 'mi_fgsm_attack':
        # load the MI-FGSM model
        adv_params = {
            'iters': args.pgd_iters,
            'optimizer': args.pgd_optimizer,
            'lr': args.pgd_optimizer_lr,
            'num_adv_ex': args.count_particles,
            'check_adv': int(args.check_adv),
            'mu': args.pgd_momentum,
        }
        if args.mi_fgsm_set_alpha:
            adv_params['original_alpha'] = False
        if args.mi_fgsm_decay:
            adv_params['decay_alpha'] = True
        adv_model = MI_FGSM_Attack(adv_params)
    elif args.adv_method == 'GNN':
        assert(args.run_lp)
        adv_params = {
            'iters': args.GNN_iters,
            'initial_step_size': args.GNN_lr_init,
            'final_step_size': args.GNN_lr_fin,
            'num_adv_ex': args.count_particles,
            'load_GNN': args.SAT_GNN_name,
            # 'pick_inits': args.pick_inits,
            'momentum': args.GNN_momentum
        }
        if args.GNN_grad_feat:
            adv_params['feature_grad'] = True
        if args.GNN_rel_decay:
            adv_params['lin_decay'] = False
        if args.GNN_adam:
            adv_params['adam'] = True
        if args.GNN_optimized:
            adv_params['GNN_optimized'] = True

        adv_model = GNN_attack(params=adv_params)
        adv_model.eval_mode()
    print('params', adv_model.params)

    live_stats = {'num_props': 0, 'total_time': 0, 'num_solved': 0}

    for new_idx, idx in enum_batch_ids:
        # loop over properties to generate new epsilon
        if new_idx > args.num_props:
            break

        imag_idx = gt_results.loc[idx]["Idx"]
        prop_idx = gt_results.loc[idx]['prop']
        eps_temp = gt_results.loc[idx]["Eps"]

        # skip over the current property if already done
        if pd.isna(graph_df.loc[idx]['Eps']) is False:
            print(f'the {new_idx}th element is done')
            continue

        # skip the nan prop_idx or eps_temp (happens in wide.pkl, jodie's mistake, I guess)
        if (math.isnan(imag_idx) or math.isnan(prop_idx) or math.isnan(eps_temp)):
            continue

        if args.printing:
            print(f"index_{new_idx}_imag_idx_{imag_idx}_prop_idx_{prop_idx}_epx_temp_{eps_temp}")

        if args.change_eps_const:
            if args.printing:
                print("eps before", eps_temp)
            eps_temp += args.change_eps_const
            if args.printing:
                print("eps after", eps_temp)
                print(f'\n ATTENTION: epsilon increased by {args.change_eps_const}\n')

        x, verif_layers, test, y, model = load_cifar_1to1_exp(args.nn_name, int(imag_idx),
                                                              int(prop_idx), return_true_class=True)
        if x is None:
            continue

        assert test == prop_idx

        domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
        if not args.cpu and torch.cuda.is_available():
            x = x.cuda()
            model.cuda()

        model.eval()

        if args.adv_method == "GNN":
            if not args.cpu and torch.cuda.is_available():
                cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
                domain = domain.cuda()
            else:
                cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

        # target = (torch.ones(x.size()[0], device=x.device)*prop_idx).long()
        # doublec check
        target = prop_idx

        data = (x.squeeze(), y, x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp)

        start_ = time.time()

        if args.run_lp:
            init_tensor, ub, lbs_all, ubs_all, dual_vars = _run_lp_as_init(verif_layers, domain, args)
            init_tensor.squeeze_(0)
            if args.adv_method == 'GNN':
                def _repeat_list(list_, args):
                    ctd = args.count_particles
                    return [l.repeat([ctd] + (len(l.size())-1)*[1]) for l in list_]
                lbs_all = _repeat_list(lbs_all, args)
                ubs_all = _repeat_list(ubs_all, args)
                dual_vars = _repeat_list(dual_vars, args)
                batch_size = args.count_particles
                lp_primal = init_tensor.unsqueeze(0).repeat([batch_size] + (len(init_tensor.size()))*[1])
        else:
            init_tensor = None
            ub = float('inf')

        # print("!!uncomment setting init_tensor to None!!")
        if args.adv_method == "GNN":
            adv_model.set_layers(cuda_verif_layers, lbs_all=lbs_all)

        if ub > 0:
            if not args.lp_init:
                init_tensor = None
            for ctd in range(args.random_restarts + 1):
                if ctd > 0:
                    init_tensor = None
                if args.adv_method in ['pgd_attack', 'mi_fgsm_attack', 'a_pgd_attack']:
                    adv_examples, is_adv, num_iters = adv_model.create_adv_examples(data, model, return_criterion='one',
                                                                                    target=target, gpu=True,
                                                                                    return_iters=True,
                                                                                    init_tensor=init_tensor)
                elif args.adv_method == 'GNN':
                    adv_examples, is_adv, num_iters = adv_model.create_adv_examples(data, model, return_criterion='one',
                                                                                    target=target, init_tensor=init_tensor,
                                                                                    lbs_all=lbs_all, ubs_all=ubs_all, lp_primal=lp_primal,
                                                                                    dual_vars=dual_vars, gpu=True, return_iters=True)

                else:
                    input("method not implemented")
                if is_adv.sum() > 0:
                    break

                if ((time.time() - start_) > args.timeout):
                    break
        else:
            adv_examples = init_tensor.unsqueeze(0)
            is_adv = torch.FloatTensor([True])
            num_iters = 0
            ctd = -1
            result_ = True

        time_taken = time.time() - start_

        if is_adv.sum() > 0:
            result_ = True
        else:
            result_ = 'timeout'

        if result_ is True:
            # pick adv example out of all examples returned by the adv method
            adv_example = adv_examples[list(is_adv).index(True)].unsqueeze(0)
            # check that the adv example returned lies within the bounds
            assert(eps_temp + 1e-4 >= (adv_example - x).abs().max()), (eps_temp, (adv_example - x).abs().max())
            # check that the images returned are actually adversarial
            # scores_ = model(adv_examples[list(is_adv).index(True)].unsqueeze(0)).squeeze()
            scores_ = model(adv_example).squeeze()
            assert(scores_[target] > scores_[y]), (scores_[target], scores_[y])
            if args.printing:
                print(f"score target {scores_[target]}, score true class {scores_[y]}")

        live_stats['num_props'] += 1
        live_stats['total_time'] += time_taken
        if result_ is True:
            live_stats['num_solved'] += 1
        if args.printing:
            print(time_taken)
            print(f"avg time {live_stats['total_time']/live_stats['num_props']} per solved {live_stats['num_solved']/live_stats['num_props']}")

        # update the result table
        graph_df.loc[idx]["Idx"] = imag_idx
        graph_df.loc[idx]["prop"] = prop_idx
        graph_df.loc[idx]["Eps"] = eps_temp
        graph_df.loc[idx]["method"] = args.adv_method
        graph_df.loc[idx]["BTime_PGD"] = time_taken
        graph_df.loc[idx]["BSAT"] = result_
        graph_df.loc[idx]["PGD_lr"] = args.pgd_optimizer_lr
        graph_df.loc[idx]["PGD_steps"] = num_iters
        graph_df.loc[idx]["batch_size"] = args.count_particles
        graph_df.loc[idx]["restarts"] = ctd
        graph_df.loc[idx]["PGD_max_steps"] = args.pgd_iters
        graph_df.loc[idx]["run_lp"] = args.run_lp
        if new_idx < 5 and args.printing:
            print(graph_df)
        graph_df.to_pickle(record_name)

    if live_stats['num_props'] > 0:
        print(f"\navg time {live_stats['total_time']/live_stats['num_props']} per solved {live_stats['num_solved']/live_stats['num_props']}\n")
    return True


def main():

    args = get_args()

    if args.seed:
        set_seed(args)

    run_pgd_attack(args)


if __name__ == "__main__":
    main()
