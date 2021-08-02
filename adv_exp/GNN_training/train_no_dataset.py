import torch as th
from datetime import datetime
import time
import os
import pandas as pd
import copy

from adv_exp.GNN_training.cli import get_args
from adv_exp.GNN_training.utils import (load_dataset, set_optimizer, process_files, save_state,
                                        load_model_and_optimizer, initialize_logger, set_seed,
                                        run_lp, load_data)
from adv_exp.GNN_attack import GNN_attack
from adv_exp.GNN_training.baselines import compute_baselines_from_table as compute_baselines
from adv_exp.GNN_training.validation import validation_from_table as validation


########################################################################
#   we train a new GNN. Unlike with train_GNN.py we don't have a training dataset
#   instead we take a table with a list of images, pros, and epsilons.
#   Each epoch we iteratively go through the images, and for each image we
#   run the lp and create a random set of initial starting points.
#   TODO:
#   - update baseline computation
#   - update plotting
#   - treat lp as features
#   - do 3 validation exp, val, deep, wide
#   - 
########################################################################


def _decay_lr(args, optimizer, model, epoch):
    # decay lr if required
    # TODO decay if no progress?
    if args.lr_decay and epoch in args.lr_decay:
        args.lr = args.lr * args.lr_decay_factor

        optimizer = set_optimizer(model, args)
        print("\nfixed lr-decay reset optimizer\n")
    return optimizer


def _logger_reset(xp, epoch_num):
    for metric in xp.loss.metrics():
        metric.reset()
    for metric in xp.lossiteration.metrics():
        metric.reset()
    for metric in xp.isadv.metrics():
        metric.reset()
    for metric in xp.opt.metrics():
        metric.reset()
    xp.timer.train.reset()
    xp.epoch.update(epoch_num)


def _logger_log(xp, args, baselines_dict):
    xp.timer.val.update()
    for string_ in baselines_dict.keys():
        getattr(xp.loss, string_).update(baselines_dict[string_]['loss'])
        getattr(xp.isadv, "subdoms - "+string_).update(baselines_dict[string_]['per_adv'])
        getattr(xp.isadv, "props - "+string_).update(baselines_dict[string_]['_per_adv_props'])

    xp.opt.optimizer.update(th.log10(th.ones(1) * args.lr))

    for metric in xp.loss.metrics():
        metric.log()
    for metric in xp.lossiteration.metrics():
        metric.log()
    for metric in xp.isadv.metrics():
        metric.log()
    for metric in xp.opt.metrics():
        metric.log()
    for metric in xp.timer.metrics():
        metric.log()

    xp.save_to(f'{args.save_path}/state.json')


# def _load_data(gt_results, idx, args):
#     imag_idx = gt_results.loc[idx]["Idx"]
#     prop_idx = gt_results.loc[idx]['prop']
#     eps_temp = gt_results.loc[idx]["Eps"]

#     x, verif_layers, test, y, model = load_cifar_1to1_exp(args.nn_name, int(imag_idx),
#                                                           int(prop_idx), return_true_class=True)

#     domain = th.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
#     if not args.cpu:
#         x = x.cuda()
#         model.cuda()

#     model.eval()

#     if not args.cpu and th.cuda.is_available():
#         cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
#         domain = domain.cuda()
#     else:
#         cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

#     data = (x.squeeze(), y, x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp)

#     return data, model, prop_idx, domain, cuda_verif_layers


def train(args, xp, baselines_dict, baselines_dict_val):

    # initialize GNN
    adv_params = {
        'iters': args.horizon,
        'optimizer': args.GNN_optimizer,
        # 'lr': args.step_size,
        'initial_step_size': args.GNN_lr_init,
        'final_step_size': args.GNN_lr_fin,
        # 'GNN_name': args.SAT_GNN_name,
        'GNN_name': args.exp_name,
        'T': args.T,
        'p': args.p,
        'num_adv_ex': 10,
        'load_GNN': '',
        'num_adv_ex': args.batch_size,
    }
    if args.feature_grad:
        adv_params['feature_grad'] = True
    if args.feature_lp_primal:
        adv_params['lp_primal'] = True

    print(adv_params)

    adv_model = GNN_attack(params=adv_params, store_loss_progress=True)
    adv_model.GNN.train()
    if (th.cuda.is_available() and args.cpu is False):
        adv_model.GNN.cuda()

    # initialize optimizer
    optimizer = set_optimizer(adv_model.GNN, args)

    if args.load_model:
        # load_best_model(model, optimizer, os.path.join(args.save_path, args.load_model), args.load_epoch)
        load_model_and_optimizer(adv_model.GNN, optimizer, '', args.load_model, args.cpu)
        if args.reset_optimizer:
            optimizer = set_optimizer(adv_model.GNN, args)
            print("load model but reset optimizer")

    # load all properties
    path = './cifar_results/adv_results/'
    path = './batch_verification_results/jade/'
    gt_results = pd.read_pickle(path + args.pdprops).dropna(how='all')[:args.max_num_prop]
    factor = 1. / len(gt_results.index)

    # best encountered loss
    best_loss = float('inf')

    # total number of files
    size_dataset = len(gt_results.index) * args.batch_size

    with th.no_grad():
        val_set_madry = validation(adv_model, args, xp, [], -1, baselines_dict_val['madry'], val_type='madry')
        val_set_val = validation(adv_model, args, xp, [], -1, baselines_dict_val['val'], val_type='val')
        val_set_wide = validation(adv_model, args, xp, [], -1, baselines_dict_val['wide'], val_type='wide')
        val_set_deep = validation(adv_model, args, xp, [], -1, baselines_dict_val['deep'], val_type='deep')

    # loop over training epochs
    for epoch_num in range(args.epoch):
        # print("epoch num", epoch_num)

        loss_epoch = 0

        if args.logger:
            _logger_reset(xp, epoch_num)

        # for loop over images
        for new_idx, idx in enumerate(gt_results.index):
            # if new_idx > args.max_num_prop:
            #     continue

            # load image
            data, model, target, domain, cuda_verif_layers = load_data(gt_results, args.nn_name, idx, args)
            adv_model.set_layers(cuda_verif_layers)

            try:
                # run the lp
                init_tensor, lbs_all, ubs_all, dual_vars, lp_primal = run_lp(cuda_verif_layers, domain, args.batch_size, args)
            except Exception:
                print(f"FAIL - new_idx, {new_idx}, {gt_results.iloc[new_idx]}")
                continue

            if not args.lp_init:
                init_tensor = None

            #     run GNN
            adv_examples, is_adv = adv_model.create_adv_examples(data, model,
                                                                 return_criterion='not_early',
                                                                 target=target,
                                                                 init_tensor=init_tensor,
                                                                 lbs_all=lbs_all, ubs_all=ubs_all,
                                                                 dual_vars=dual_vars, lp_primal=lp_primal,
                                                                 gpu=(not args.cpu))

            loss_progress = adv_model.loss_progress
            loss = 0
            for idx_ in range(len(loss_progress)):
                loss -= (args.gamma ** idx_) * loss_progress[idx_]

            #     take an optimizing step
            loss.backward()

            grad_records = [ii.grad for ii in adv_model.GNN.parameters()]
            # print(grad_records)
            req_grad_records = [ii.requires_grad for ii in adv_model.GNN.parameters()]
            # print(req_grad_records)
            # input("Wat")
            # print([float(ii - loss_progress[0]) for ii in loss_progress])

            optimizer.step()
            optimizer.zero_grad()

            loss_epoch += float(loss)

            is_adv_prop = (is_adv.sum() > 0)

            if args.logger:
                xp.loss.loss.update(-loss*factor)
                xp.loss.gnn_final.update(float(loss_progress[-1])*factor)

                xp.isadv.per.update(is_adv.sum()/float(size_dataset))

                for i_plot in range(args.horizon):
                    getattr(xp.lossiteration, "step"+str(i_plot)).update(loss_progress[i_plot] * factor)

                # for string_ in baselines_dict.keys():
                #     getattr(xp.loss, string_).update(baselines_dict[string_]['loss'])
                #     getattr(xp.isadv, string_).update(baselines_dict[string_]['per_adv'])

            if args.logger:
                xp.isadv.props.update(int(is_adv_prop))

        if args.logger:
            xp.timer.train.update()
            xp.timer.val.reset()

        if args.epoch<=10 or epoch_num % int(float(args.epoch)/10.0) == 0 or epoch_num == args.epoch - 1:
            with th.no_grad():
                val_set_val = validation(adv_model, args, xp, val_set_val, epoch_num, baselines_dict_val['val'], val_type='val')
                val_set_wide = validation(adv_model, args, xp, val_set_wide, epoch_num, baselines_dict_val['wide'], val_type='wide')
                val_set_deep = validation(adv_model, args, xp, val_set_deep, epoch_num, baselines_dict_val['deep'], val_type='deep')
                val_set_madry = validation(adv_model, args, xp, val_set_madry, epoch_num, baselines_dict_val['madry'], val_type='madry')
                print("TODO, add dict for madry")

        if args.logger:
            # for string_ in baselines_dict.keys():
            #     getattr(xp.loss, string_).update(baselines_dict[string_]['loss'])
            #     getattr(xp.isadv, string_).update(baselines_dict[string_]['per_adv'])
            # xp.opt.optimizer.update(th.log10(th.ones(1) * args.lr))
            _logger_log(xp, args, baselines_dict)

        optimizer = _decay_lr(args, optimizer, adv_model.GNN, epoch_num)

        # save GNN
        if (args.save_model and epoch_num % args.save_model == 0 and epoch_num > 0):
            save_state(adv_model.GNN, optimizer, f'{args.save_path}/model-{int(epoch_num)}.pkl')
            print(f"save model epoch {epoch_num}")
        if (args.save_model and loss_epoch < best_loss):
            best_loss = loss_epoch
            save_state(adv_model.GNN, optimizer, f'{args.save_path}/model-best.pkl')
            print(f"save best model, epoch {epoch_num}")


def main():
    print("\n\nstarting train_outer_batch", datetime.now(), '\n\n')
    time_start_main = time.time()

    # get command line arguments
    args = get_args()
    print(args)

    set_seed(args)

    if args.load_model:
        if args.load_model in ('same', 'same1'):
            args.load_model = os.path.join(args.save_path, args.exp_name + '/model-best.pkl')
        else:
            args.load_model = os.path.join(args.save_path, args.load_model)
    args.save_path = os.path.join(args.save_path, args.exp_name)

    if args.logger:
        if (os.path.exists(args.save_path) is False):
            os.makedirs(args.save_path)

    # load training dataset and compute baselines
    # _, _, dict_files, num_props, num_subdoms = load_dataset(args) 
    baselines_dict, num_props = compute_baselines(args, args.nn_name, args.pdprops, val_size=args.max_num_prop)
    num_subdoms = args.batch_size * num_props
    dict_files = {}

    # load validation dataset and compute baselines
    dict_madry, num_madry = compute_baselines(args, 'cifar_madry', 'madry_easy_SAT_jade.pkl', val_size=args.val_size)
    dict_val, num_val = compute_baselines(args, 'cifar_base_kw', 'val_SAT_jade.pkl', val_size=args.val_size)
    dict_wide, num_wide = compute_baselines(args, 'cifar_wide_kw', 'wide_SAT_jade.pkl', val_size=args.val_size)
    dict_deep, num_deep = compute_baselines(args, 'cifar_deep_kw', 'deep_SAT_jade.pkl', val_size=args.val_size)
    baselines_dict_val = {'val': dict_val, 'wide': dict_wide, 'deep': dict_deep, 'madry': dict_madry}

    print(f"datasets sizes: train: {num_props} - val: {num_val} - wide: {num_wide} - deep: {num_deep}")
    # create logger and initialize visdom
    if args.logger:
        xp = initialize_logger(args, baselines_dict.keys(), num_props, num_subdoms)
    else:
        xp = None

    train(args, xp, baselines_dict, baselines_dict_val)


if __name__ == '__main__':
    main()
