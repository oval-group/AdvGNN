import os
import torch
import torch as th
import torchvision
import argparse
import time

import adv_exp.foolbox.foolbox as fb
import adv_exp.foolbox.foolbox.attacks as fa

import pandas as pd
import math
from lp_solver.model_utils import load_cifar_1to1_exp, load_1to1_eth
import numpy as np


#######################################################################################################
#   run the foolbox attack methods on our datasets
#   This has only been used for the Carlini Wagner attack which performs
#       significantly worse than the other baselines used
#######################################################################################################


def _load_attack(attack_type, args, **kargs):
    default_params = {
        'PGD': {'steps': args.steps, 'abs_stepsize': args.pgd_lr},
        'DeepFool': {'steps': args.steps, 'overshoot': args.deepfool_overshoot, 'candidates': 2},
        'BasicIterative': {'steps': args.steps, 'abs_stepsize': args.BI_lr, 'random_start': True},
        'FGSM': {},
        'Additive_noise': {},
        'Repeated_noise': {'repeats': args.repeats},
        'CW': {'steps': args.CW_steps, 'reduce_const': True, 'initial_const': args.CW_init_const, 'largest_const': args.CW_largest_const,
               'decrease_factor': args.CW_decrease_factor, 'stepsize': args.CW_lr, 'const_factor': args.CW_const_factor},
        }
    # new_params = dict(default_params[attack_type], **kargs) if kargs is not None else default_params[attack_type]
    new_params = default_params[attack_type]
    print(attack_type, " with: ", new_params)
    if attack_type == 'PGD':
        return fa.PGD(**new_params)
    elif attack_type == 'DeepFool':
        return fa.LinfDeepFoolAttack(**new_params)
    elif attack_type == 'BasicIterative':
        return fa.LinfBasicIterativeAttack(**new_params)
    elif attack_type == 'FGSM':  # apparently only works for untargeted attacks
        return fa.FGSM(**new_params)
    elif attack_type == 'Additive_noise':
        return fa.LinfAdditiveUniformNoiseAttack(**new_params)
    elif attack_type == 'Repeated_noise':
        return fa.LinfRepeatedAdditiveUniformNoiseAttack(**new_params)
    elif attack_type == 'Brendel':
        return fa.LinfinityBrendelBethgeAttack(**new_params)
    elif attack_type == 'CW':
        return fa.LinfCarliniWagnerAttack(**new_params)
    else:
        raise NotImplementedError


def foolbox(args):

    batch_size = args.batch_size

    # # initialize a file to record all results, record should be a pandas dataframe
    if args.data == 'cifar' or args.data == 'cifar10':
        path = './batch_verification_results/'
        result_path = './cifar_results/adv_results/'

    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)

    # # load all properties
    if args.data == 'mnist' or args.data == 'cifar10':
        csvfile = open('././data/%s_test.csv' % (args.data), 'r')
        tests = list(csv.reader(csvfile, delimiter=','))
        batch_ids = range(100)
        enum_batch_ids = [(bid, None) for bid in batch_ids]
    elif args.data == 'cifar':
        gt_results = pd.read_pickle(path + args.pdprops)
        bnb_ids = gt_results.index
        batch_ids = bnb_ids
        enum_batch_ids = enumerate(batch_ids)

    if args.record:
        result_path = './cifar_results/adv_results/'
        record_name = result_path + args.record_name
        if os.path.isfile(record_name):
            graph_df = pd.read_pickle(record_name)
        else:
            _columns = ['Idx', 'Eps', 'prop', 'BSAT', 'BTime', 'method',  # 'PGD_lr', 'PGD_steps',
                        'restarts', 'batch_size']
            if args.adv_method == 'CW':
                _columns.append('best_eps_found')
                _columns.append('eps_difference')
            graph_df = pd.DataFrame(index=bnb_ids, columns=_columns)

        graph_df.experiment_details = args

    # print(graph_df.experiment_details)
    # print(graph_df.__dict__)

    for new_idx, idx in enum_batch_ids:

        if new_idx > args.num_props:
            break

        if args.data == 'cifar':

            imag_idx = gt_results.loc[idx]["Idx"]
            prop_idx = gt_results.loc[idx]['prop']
            eps_temp = gt_results.loc[idx]["Eps"]

            # skip the nan prop_idx or eps_temp (happens in wide.pkl, jodie's mistake, I guess)
            if (math.isnan(imag_idx) or math.isnan(prop_idx) or math.isnan(eps_temp)):
                continue

            # skip over the current property if already done
            if args.record and pd.isna(graph_df.loc[idx]['Eps']) is False:
                print(f'the {new_idx}th element is done')
                continue

            x, verif_layers, test, y, model = load_cifar_1to1_exp(args.nn_name, int(imag_idx), int(prop_idx),
                                                                  return_true_class=True)
            # since we normalise cifar data set, it is unbounded now
            bounded = False
            assert test == prop_idx
            # eps_temp += 0.5
            domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
            linear = False
            if not args.cpu:
                x = x.cuda()
                model.cuda()
            data = (x.squeeze(), y, x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp)

            targeted_attack = True
        elif args.data == 'mnist' or args.data == 'cifar10':
            imag_idx = new_idx
            eps_temp = float(args.nn_name[6:]) if args.data == 'mnist' else float(args.nn_name.split('_')[1])/float(args.nn_name.split('_')[2])

            x, verif_layers, test, domain, y, model = load_1to1_eth(args.data, args.nn_name, idx=imag_idx, test=tests,
                                                                    eps_temp=eps_temp,
                                                                    max_solver_batch=args.max_solver_batch,
                                                                    return_true_class=True)
            if x is None:
                # handle misclassified images
                continue
            # since we normalise cifar data set, it is unbounded now
            bounded = False
            prop_idx = test
            linear = False
            if not args.cpu:
                verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
                x = x.cuda()
                domain = domain.cuda()
                model.cuda()
            data = (x.squeeze(), y, domain[:, :, :, 0], domain[:, :, :, 1])

            targeted_attack = False

        model.eval()
        bounds = (-10, 10)
        fmodel = fb.PyTorchModel(model, bounds=bounds)

        target = (torch.ones(x.size()[0], device=x.device)*prop_idx).long()
        images = x

        images = images.repeat(batch_size, 1, 1, 1)
        target = target.repeat(batch_size)

        if targeted_attack:
            criterion = fb.criteria.TargetedMisclassification(target)
        else:
            labels = model(images).argmax(dim=1)
            criterion = fb.criteria.Misclassification(labels)

        attack = _load_attack(args.adv_method, args)

        # epsilon_lists = np.linspace(0.0, 0.001, num=2)+eps_temp
        # epsilon_lists = np.linspace(0.5, 1, num=20)+eps_temp
        epsilon_lists = [eps_temp]

        time_start = time.time()
        for ctd in range(args.random_restarts + 1):
            print("Restart number: ", ctd, end="\r")
            raw, clipped, is_adv = attack(fmodel, images, criterion, epsilons=epsilon_lists, eps=eps_temp)
            # print(is_adv)
            # assert(False)

            if is_adv.sum() > 0:
                break

            if ((time.time() - time_start) > args.timeout):
                break

        time_taken = time.time() - time_start

        if args.adv_method == 'CW':
            best_eps = (raw[0] - images).abs().max()
            print("best eps_found", float(best_eps))

        print(th.sum(is_adv, dim=1))
        is_adv2 = (th.sum(is_adv, dim=1) >= 1)
        if True in is_adv2:
            idx_first_adv = list(is_adv2).index(True)
        else:
            idx_first_adv = -1

        num_false = len(is_adv2) - sum(is_adv2)
        num_true = sum(is_adv2)

        for idx_ in range(len(epsilon_lists)):
            # assert(epsilon_lists[idx_] + 1e-6 >= (raw[idx_] - images).abs().max()), (idx_, epsilon_lists[idx_], (raw[idx_] - images).abs().max())
            assert(epsilon_lists[idx_] + 1e-6 >= (clipped[idx_] - images).abs().max()), (idx_, epsilon_lists[idx_], (clipped[idx_] - images).abs().max())

        if num_true > 0:
            print(f"previous unsuccesful: {epsilon_lists[idx_first_adv-1]}, smallest successful: {epsilon_lists[idx_first_adv]} -- original eps(robust): {eps_temp}")
            result_ = True
        else:
            print("all unsuccessful")
            result_ = 'timeout'

        print('time', time_taken)

        if result_ is True:
            adv_examples = clipped[0]
            is_adv = is_adv[0]
            # pick adv example out of all examples returned by the adv method
            adv_example = adv_examples[list(is_adv).index(True)].unsqueeze(0)
            # check that the adv example returned lies within the bounds
            assert(eps_temp + 1e-4 >= (adv_example - x).abs().max()), (eps_temp, (adv_example - x).abs().max())
            # check that the images returned are actually adversarial
            # scores_ = model(adv_examples[list(is_adv).index(True)].unsqueeze(0)).squeeze()
            scores_ = model(adv_example).squeeze()
            target = prop_idx
            assert(scores_[target] > scores_[y]), (scores_[target], scores_[y])
            print(f"score target {scores_[target]}, score true class {scores_[y]}")
            # assert(False)

        if args.record:
            # update the result table
            graph_df.loc[idx]["Idx"] = imag_idx
            graph_df.loc[idx]["prop"] = prop_idx
            graph_df.loc[idx]["Eps"] = eps_temp
            graph_df.loc[idx]["method"] = args.adv_method
            graph_df.loc[idx]["BTime"] = time_taken
            graph_df.loc[idx]["BSAT"] = result_
            # graph_df.loc[idx]["PGD_lr"] = args.pgd_optimizer_lr
            # graph_df.loc[idx]["PGD_steps"] = num_iters
            graph_df.loc[idx]["batch_size"] = args.batch_size
            graph_df.loc[idx]["restarts"] = ctd
            if args.adv_method == 'CW':
                graph_df.loc[idx]["best_eps_found"] = float(best_eps)
                graph_df.loc[idx]["eps_difference"] = eps_temp - float(best_eps)
            if eps_temp - float(best_eps) > -0.01:
                print("\n\nfound one with less than easy", eps_temp - float(best_eps), "\n")
            print("diff, ", eps_temp - float(best_eps))
            # graph_df.loc[idx]["PGD_max_steps"] = args.pgd_iters
            if new_idx < 5:
                print(graph_df)
            # print(record_name)
            graph_df.to_pickle(record_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true', help='file to save results')
    parser.add_argument('--record_name', type=str, help='file to save results')
    parser.add_argument('--pdprops', type=str, help='pandas table with all props we are interested in', default='train_SAT_bugfixed_intermediate.pkl')
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--cpus_total', type=int, help='total number of cpus used')
    parser.add_argument('--cpu_id', type=int, help='the index of the cpu from 0 to cpus_total')
    parser.add_argument('--cpu', action='store_true', help='run experiments on cpus rather than CUDA')
    parser.add_argument('--nn_name', type=str, help='network architecture name', default='cifar_base_kw')
    parser.add_argument('--adv_method', type=str, help='attacking method', default='PGD',
                        choices=['PGD', 'DeepFool', 'BasicIterative', 'Additive_noise', 'Repeated_noise', 'Brendel', 'FGSM', 'CW'])
    parser.add_argument('--data', type=str, default='cifar')
    parser.add_argument('--random_restarts', type=int, default=0,
                        help='number of times we restart PGD if unsuccesful')
    parser.add_argument('--batch_size', type=int, help='count runs', default=100)

    parser.add_argument('--num_props', type=int, default=10000)

    parser.add_argument('--steps', type=int, help='learning rate pgd attack', default=1000)
    # pgd
    parser.add_argument('--pgd_lr', type=float, help='learning rate pgd attack', default=1e-2)
    # deepfool
    parser.add_argument('--deepfool_overshoot', type=float, help='learning rate pgd attack', default=0.0)
    # Basic Iterative
    parser.add_argument('--BI_lr', type=float, help='learning rate Basic Iterative attack', default=1e-3)
    # Repeated Noise
    parser.add_argument('--repeats', type=int, help='repeated noise repeats', default=1000)
    # Carlini Wagner
    if False:
        # round 1 of hparam analysis
        d_steps = 1000
        d_lr = 1e-2
        d_init_cstd = 1e-5
        d_larg_cstd = 2e+1
        d_factor = 2.0
        d_decrease = 0.9
    elif False:
        # round 2 of hparam analysis
        d_steps = 100
        d_lr = 1e-3
        d_init_cstd = 1e-3
        d_larg_cstd = 100
        d_factor = 2.0
        d_decrease = 0.9
    elif False:
        # round 3 of hparam analysis
        d_steps = 100
        d_lr = 1e-3
        d_init_cstd = 1e-4
        d_larg_cstd = 1000
        d_factor = 1.5
        d_decrease = 0.99
    elif True:
        # round 3 of hparam analysis
        d_steps = 100
        d_lr = 1e-4
        d_init_cstd = 1e-5
        d_larg_cstd = 1000
        d_factor = 1.5
        d_decrease = 0.99

    parser.add_argument('--CW_steps', type=int, help='learning rate pgd attack', default=d_steps)
    parser.add_argument('--CW_bin_search_steps', type=int, help='CW binary search steps', default=9)
    parser.add_argument('--CW_lr', type=float, help='learning rate CW attack', default=d_lr)
    parser.add_argument('--CW_init_const', type=float, help='learning rate CW attack', default=d_init_cstd)
    parser.add_argument('--CW_largest_const', type=float, help='learning rate CW attack', default=d_larg_cstd)
    parser.add_argument('--CW_const_factor', type=float, help='learning rate CW attack', default=d_factor)
    parser.add_argument('--CW_decrease_factor', type=float, help='learning rate CW attack', default=d_decrease)

    args = parser.parse_args()

    foolbox(args)


if __name__ == "__main__":
    main()
