import torch
import torchvision
import argparse
import os
import pandas as pd
import math
from lp_solver.model_utils import load_cifar_1to1_exp, load_1to1_eth
import numpy as np
import time
from adv_exp.pgd_attack import Pgd_Attack
from adv_exp.analysis.load_mnist import load_mnist_wide_net


####################################################################
#     Creates the OVAL SAT dataset
####################################################################


def _create_dataset(args):

    def load_attack(attack_type, **kargs):
        # return a foolbox attack
        if attack_type == 'PGD':
            return fb.attacks.PGD(**kargs)
        elif attack_type == 'DeepFool':
            return fb.attacks.LinfDeepFoolAttack(**kargs)
        elif attack_type == 'CW':
            input("note: CW only works for l2 norm")
            return fb.attacks.L2CarliniWagnerAttack(**kargs)
        elif attack_type == 'FGSM':
            return fb.attacks.FGSM(**kargs)
        else:
            raise NotImplementedError

    # # initialize a file to record all results, record should be a pandas dataframe
    path = './batch_verification_results/'
    result_path = './cifar_results/adv_results/'

    if args.table_name:
        new_table = args.table_name
    elif args.pdprops == 'jodie-base_easy.pkl':
        new_table = 'base_easy_SAT.pkl'
    elif args.pdprops == 'jodie-base_med.pkl':
        new_table = 'base_med_SAT.pkl'
    elif args.pdprops == 'jodie-base_hard.pkl':
        new_table = 'base_hard_SAT.pkl'
    elif args.pdprops == 'jodie-deep.pkl':
        new_table = 'deep_SAT.pkl'
    elif args.pdprops == 'jodie-wide.pkl':
        new_table = 'wide_SAT.pkl'
    record_name = result_path + new_table

    # # load all properties
    gt_results = pd.read_pickle(path + args.pdprops)
    bnb_ids = gt_results.index
    batch_ids = bnb_ids
    enum_batch_ids = enumerate(batch_ids)

    if os.path.isfile(record_name):
        graph_df = pd.read_pickle(record_name)
    else:
        _columns = ['Idx', 'Eps', 'prop', 'Eps_low', 'PGD_time', 'PGD_lr', 'PGD_steps', 'bin_eps']
        graph_df = pd.DataFrame(index=bnb_ids, columns=_columns)

    for new_idx, idx in enum_batch_ids:
        # loop over properties to generate new epsilon

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

        x, verif_layers, test, y, model = load_cifar_1to1_exp(args.nn_name, int(imag_idx), int(prop_idx),
                                                              return_true_class=True)
        # since we normalise cifar data set, it is unbounded now
        bounded = False
        assert test == prop_idx

        domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
        linear = False
        if not args.cpu:
            x = x.cuda()
            model.cuda()
        data = (x.squeeze(), y, x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp)

        targeted_attack = True

        model.eval()
        bounds = (-10, 10)
        fmodel = fb.PyTorchModel(model, bounds=bounds)

        target = (torch.ones(x.size()[0], device=x.device)*prop_idx).long()
        images = x

        if targeted_attack:
            criterion = fb.criteria.TargetedMisclassification(target)
        else:
            labels = model(images).argmax(dim=1)
            criterion = fb.criteria.Misclassification(labels)

        if args.pgd_abs_lr:
            attack = load_attack('PGD', steps=int(args.pgd_steps), abs_stepsize=args.pgd_abs_lr)
        else:
            attack = load_attack('PGD', steps=int(args.pgd_steps))

        # the network is robust for eps_temp & pgd is unable to find a adv_example for lower_bound
        # (note: this done not prove robustness)
        lower_bound = eps_temp
        # there exists an adversarial example for eps = upper_bound
        upper_bound = eps_temp + 0.5

        time_last_adv = 0

        print("start binary search")
        print(lower_bound, upper_bound, args.bin_search_eps)
        # start binary search
        while upper_bound - lower_bound > args.bin_search_eps:
            cur_epsilon = (upper_bound + lower_bound)/2

            attack_count = 0  # number of times we've run PGD on the current epsilon

            start_ = time.time()
            for ctd in range(args.random_restarts + 1):
                raw, clipped, is_adv = attack(fmodel, images, criterion, epsilons=cur_epsilon)
                if is_adv:
                    break
            time_taken = time.time() - start_

            print(is_adv)
            if is_adv:
                # we found an adversarial example for cur_epsilon
                upper_bound = cur_epsilon
                time_last_adv = time_taken
            else:
                lower_bound = cur_epsilon

            # check that clipped is within the bounds
            assert(cur_epsilon + 1e-4 >= (clipped - images).abs().max())

            print(time_taken)
            print(f"current bounds: lb: {lower_bound}, ub: {upper_bound}")

        # update the result table
        graph_df.loc[idx]["Idx"] = imag_idx
        graph_df.loc[idx]["Eps"] = cur_epsilon
        graph_df.loc[idx]["prop"] = prop_idx
        graph_df.loc[idx]["Eps_low"] = lower_bound
        graph_df.loc[idx]["PGD_time"] = time_last_adv
        graph_df.loc[idx]["PGD_lr"] = args.pgd_abs_lr
        graph_df.loc[idx]["PGD_steps"] = args.pgd_steps
        graph_df.loc[idx]["bin_eps"] = args.bin_search_eps
        print(graph_df)
        graph_df.to_pickle(record_name)


def create_dataset_manual(args):

    # # initialize a file to record all results, record should be a pandas dataframe
    path = './batch_verification_results/'
    result_path = './cifar_results/adv_results/'

    if args.table_name:
        new_table = args.table_name
    elif args.pdprops == 'jodie-base_easy.pkl':
        new_table = 'base_easy_SAT.pkl'
    elif args.pdprops == 'jodie-base_med.pkl':
        new_table = 'base_med_SAT.pkl'
    elif args.pdprops == 'jodie-base_hard.pkl':
        new_table = 'base_hard_SAT.pkl'
    elif args.pdprops == 'jodie-deep.pkl':
        new_table = 'deep_SAT.pkl'
    elif args.pdprops == 'jodie-wide.pkl':
        new_table = 'wide_SAT.pkl'
    record_name = result_path + new_table

    # # load all properties
    gt_results = pd.read_pickle(path + args.pdprops)
    bnb_ids = gt_results.index
    batch_ids = bnb_ids
    enum_batch_ids = enumerate(batch_ids)

    if os.path.isfile(record_name):
        graph_df = pd.read_pickle(record_name)
    else:
        _columns = ['Idx', 'Eps', 'prop', 'Eps_low', 'PGD_time', 'PGD_lr', 'PGD_steps', 'bin_eps', 'restarts']
        graph_df = pd.DataFrame(index=bnb_ids, columns=_columns)

    # load the pgd model
    adv_params = {
        'iters': args.pgd_iters,
        'optimizer': args.pgd_optimizer,
        'lr': args.pgd_optimizer_lr,
        'num_adv_ex': args.count_particles,
    }
    adv_model = Pgd_Attack(adv_params)

    for new_idx, idx in enum_batch_ids:
        # loop over properties to generate new epsilon

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

        if args.mnist:
            x, verif_layers, test, y, model = load_mnist_wide_net(int(imag_idx), network='wide', test=int(prop_idx))
        else:
            x, verif_layers, test, y, model = load_cifar_1to1_exp(args.nn_name, int(imag_idx), int(prop_idx),
                                                                  return_true_class=True)

        # since we normalise cifar data set, it is unbounded now
        bounded = False
        assert test == prop_idx

        domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
        linear = False
        if not args.cpu:
            x = x.cuda()
            model.cuda()

        targeted_attack = True

        model.eval()

        target = prop_idx
        images = x

        # the network is robust for eps_temp & pgd is unable to find a adv_example for lower_bound
        # (note: this does not prove robustness)
        if args.start_at_eps0:
            lower_bound = 0.0
        else:
            lower_bound = eps_temp
        # there exists an adversarial example for eps = upper_bound
        upper_bound = eps_temp + 0.5

        time_last_adv = 0

        print("start binary search")
        print(lower_bound, upper_bound, args.bin_search_eps)
        # start binary search
        while upper_bound - lower_bound > args.bin_search_eps:
            cur_epsilon = (upper_bound + lower_bound)/2

            data = (x.squeeze(), y, x.squeeze(0) - cur_epsilon, x.squeeze(0) + cur_epsilon)

            start_ = time.time()
            for ctd in range(args.random_restarts + 1):
                adv_examples, is_adv = adv_model.create_adv_examples(data, model, return_criterion='one',
                                                                     target=target, gpu=True)

                if is_adv.sum() > 0:
                    break
            time_taken = time.time() - start_

            if is_adv.sum() > 0:
                print("True")
                # we found an adversarial example for cur_epsilon
                upper_bound = cur_epsilon
                time_last_adv = time_taken
            else:
                print("False")
                lower_bound = cur_epsilon

            # check that clipped is within the bounds
            assert(cur_epsilon + 1e-4 >= (adv_examples - images).abs().max())

            print(time_taken)
            print(f"current bounds: lb: {lower_bound}, ub: {upper_bound}")

        # update the result table
        graph_df.loc[idx]["Idx"] = imag_idx
        graph_df.loc[idx]["Eps"] = upper_bound
        graph_df.loc[idx]["prop"] = prop_idx
        graph_df.loc[idx]["Eps_low"] = lower_bound
        graph_df.loc[idx]["PGD_time"] = time_last_adv
        graph_df.loc[idx]["PGD_lr"] = args.pgd_optimizer_lr
        graph_df.loc[idx]["PGD_steps"] = args.pgd_iters
        graph_df.loc[idx]["bin_eps"] = args.bin_search_eps
        graph_df.loc[idx]["restarts"] = ctd
        print(graph_df)
        graph_df.to_pickle(record_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_name', type=str, help='file to save results')
    parser.add_argument('--pdprops', type=str, default='jodie-base_hard.pkl',
                        help='pandas table with all props we are interested in')
    parser.add_argument('--cpu', action='store_true', help='run experiments on cpus rather than CUDA')
    parser.add_argument('--nn_name', type=str, help='network architecture name', default='cifar_base_kw')
    parser.add_argument('--bin_search_eps', type=float, default=1e-3, help='epsilon for binary search')
    parser.add_argument('--pgd_steps', type=float, default=1e4, help='number of steps for pgd attack')
    parser.add_argument('--pgd_abs_lr', type=float, help='abs lr for pgd attack')
    parser.add_argument('--random_restarts', type=int, default=0,
                        help='number of times we restart PGD if unsuccesful')
    parser.add_argument('--table_name', type=str, help='optional name of the result table')

    parser.add_argument('--pgd_iters', type=int, help='pgd_iters', default=10000)
    parser.add_argument('--pgd_optimizer', type=str, default='default',
                        choices=["adam", "sgd", "default", "SGLD", "GNN"])
    parser.add_argument('--pgd_optimizer_lr', type=float, help='learning rate pgd attack', default=1e-3)
    parser.add_argument('--count_particles', type=int, help='count runs', default=100)

    parser.add_argument('--manual_impl', action='store_true', help='use my version of the pgd attack')
    parser.add_argument('--start_at_eps0', action='store_true', help='start with a lower bound of 0 rather than an UNSAT eps')

    parser.add_argument('--mnist', action='store_true', help='run on mnist rather than cifar')
    args = parser.parse_args()

    if args.manual_impl:
        create_dataset_manual(args)
    else:
        _create_dataset(args)


if __name__ == "__main__":
    main()
