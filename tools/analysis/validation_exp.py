import mlogger
import pandas as pd
import torch as th
import argparse
import visdom
import time
import csv
import copy
from adv_exp.GNN_training.utils import process_files, run_lp, load_data
from adv_exp.GNN_attack import GNN_attack
from adv_exp.GNN_momentum import GNN_momentum
from adv_exp.mi_fgsm_attack import MI_FGSM_Attack
from adv_exp.pgd_attack import Pgd_Attack

from tools.bab_tools.model_utils import load_cifar_1to1_exp, load_1to1_eth

########################################################################
#   run validation experiments. Take as input a single trained GNN
#   and run hparam analysis on validation dataset
#   TODO
########################################################################


def load_eth_data(nn_name, idx, args, tests):
    imag_idx = idx
    eps_temp = 8/255 

    max_solver_batch = 2800

    #  x, verif_layers, test, domain, y_pred, model = load_1to1_eth('cifar10', nn_name, idx=imag_idx, test=tests, eps_temp=eps_temp,
    #                                                       max_solver_batch=args.max_solver_batch, return_true_class=True)
    return_tuple = load_1to1_eth('cifar10', nn_name, idx=imag_idx, test=tests, eps_temp=eps_temp,
                                                          max_solver_batch=args.max_solver_batch, return_true_class=True)
    if len(return_tuple) == 4:
        return None, None, None, None, None
    x, verif_layers, test, domain, y_pred, model = return_tuple

    # domain = th.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
    if not args.cpu and th.cuda.is_available():
        x = x.cuda()
        model.cuda()

    model.eval()

    if not args.cpu and th.cuda.is_available():
        cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
        domain = domain.cuda()
    else:
        cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

    data = (x.squeeze(), y_pred, domain[:,:,:,0], domain[:,:,:,1])
    # data = (x.squeeze(), y_pred, x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp)

    print(model)
    input("wait")

    return data, model, None, domain, cuda_verif_layers


def get_exp_params(exp_name):
  if exp_name == 'one':
    # init_step_list = [1e-1, 1e-2, 1e-3]
    # final_step_list = [1e-2, 1e-3, 1e-4]
    init_step_list = [1e-2]
    final_step_list = [1e-3]
    # steps_list = [20, 40]
    steps_list = [20]
    decay_type_list = [True]

    exp_dict = []
    for init_step in init_step_list:
        for final_step in final_step_list:
          for steps in steps_list:
           for decay_type in decay_type_list:
            exp_dict.append(({'initial_step_size': init_step,
                              'final_step_size': final_step,
                              'iters': steps,
                              'lin_decay' : decay_type,
                              },
                            f'init_{init_step}_final_{final_step}_steps_{steps}_lindc_{decay_type}'))
  elif exp_name == 'mom_adam':
        momentum_list = [0, 0.01, 0.1, 0.25,0.5, 0.75, 0.9]
        momentum_list = [None, 0, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2]
        adam_list = [True, False]
        adam_list = [False]
        exp_dict = []
        for mom in momentum_list:
            for adam in adam_list:
                exp_dict.append(({'adam': adam, 'momentum': mom}, 
                                f'momentum_{mom}_adam_{adam}'))
  elif exp_name == 'adam_lr':
    init_step_list = [100, 30, 10, 3.0, 1.0]
    final_step_list = [10, 1, 1e-1, 1e-2, 1e-3]
    steps_list = [100]
    decay_type_list = [True]

    exp_dict = []
    for init_step in init_step_list:
        for final_step in final_step_list:
          for steps in steps_list:
           for decay_type in decay_type_list:
            exp_dict.append(({'initial_step_size': init_step,
                              'final_step_size': final_step,
                              'iters': steps,
                              'lin_decay' : decay_type,
                              'adam': True,
                              },
                            f'init_{init_step}_final_{final_step}_steps_{steps}_lindc_{decay_type}'))
    exp_dict.append(({'initial_step_size': 1e-2,
                              'final_step_size': 1e-3,
                              'iters': 100,
                              'lin_decay' : True,
                              'adam': False,
                              }, 'baselines'))

  elif exp_name == 'compare_adam_impl':
    init_step_list = [3.0]
    final_step_list = [1e-2]
    steps_list = [20]
    decay_type_list = [True]

    exp_dict = []
    for init_step in init_step_list:
        for final_step in final_step_list:
          for steps in steps_list:
           for decay_type in decay_type_list:
            exp_dict.append(({'initial_step_size': init_step,
                              'final_step_size': final_step,
                              'iters': steps,
                              'lin_decay' : decay_type,
                              'adam': True,
                              },
                            f'init_{init_step}_final_{final_step}_steps_{steps}_lindc_{decay_type}'))

    exp_dict.append(({'initial_step_size': 1e-2,
                              'final_step_size': 1e-3,
                              'iters': 20,
                              'lin_decay' : True,
                              'adam': False,
                              }, 'baselines'))

  elif exp_name == 'debug_new_GNN' or exp_name == 'debug_old_GNN':
      exp_dict = [({'iters': int(40)}, 'debug')]
  elif exp_name == 'debug_mi_fgsm':
    exp_dict = []
    # for mu in range(0,10,1):
    #   for lr in [5.0, 1.0, 5e-1]:
    for mu in [0.5]:
        for lr in [5.0]:
            exp_dict.append(({'iters': int(1e2), 'mu':mu/10, 'lr':lr}, f'debug_mu_{mu/10}_lr{lr}'))
  elif exp_name == 'debug_GNN_momentum':
      exp_dict = []
      alpha_list = [5.0,1.0,]
      mu_list = [0.0, 0.25, 0.5, 0.75]
      for alpha in alpha_list:
          for mu in mu_list:
              exp_dict.append(({'alpha': alpha, 'mu': mu}, f'alpha_{alpha}_mu_{mu}'))
  elif exp_name == 'debug_eth':
      exp_dict = [(({}, 'debug'))]

  return exp_dict


def validation_exp(args):
    time_start = time.time()
    val_iters = 80
    val_size = 10
    table_name = args.pdprops
    batch_size = 20
    lp_init = False
    batch_size = 40

    if 'val' in table_name:
        nn_name = 'cifar_base_kw'
        batch_size = 10
    if 'wide' in table_name:
        nn_name = 'cifar_wide_kw'
        batch_size = 10
    if 'deep' in table_name:
        nn_name = 'cifar_deep_kw'
        batch_size = 10
    print("nn_name", nn_name)

    # load all properties
    path = './cifar_results/adv_results/'
    path = './datasets/CIFAR10/SAT/'
    gt_results = pd.read_pickle(path + table_name).dropna(how='all')[:val_size]
    factor = 1. / len(gt_results.index)

    # size_dataset = len([item for sublist in [dict_files[key] for key in dict_files.keys()] for item in sublist])
    size_dataset = min(val_size, len(gt_results.index))

    result_list = []

    adv_method = 'mi_fgsm'
    adv_method = 'GNN_momentum'
    # adv_method = 'GNN'
    if adv_method == 'GNN':
     adv_params = {
        'iters': val_iters,
        # 'initial_step_size': args.adv_lr_init,
        # 'final_step_size': args.adv_lr_fin,
        'load_GNN': args.SAT_GNN_name,
        'lp_primal': True,
        'feature_grad': True,
        # 'pick_inits': args.pick_inits,
        'num_adv_ex': batch_size,
     }
     adv_model = GNN_attack(params=adv_params, store_loss_progress=True)
     adv_model.eval_mode()
    elif adv_method == 'GNN_momentum':
     adv_params = {
        'iters': val_iters,
        # 'initial_step_size': args.adv_lr_init,
        # 'final_step_size': args.adv_lr_fin,
        'load_GNN': args.SAT_GNN_name,
        'lp_primal': True,
        'feature_grad': True,
        # 'pick_inits': args.pick_inits,
        'num_adv_ex': batch_size,
     }
     adv_model = GNN_momentum(params=adv_params, store_loss_progress=True)
     adv_model.eval_mode()
    else:
     adv_params = {}
     adv_model = MI_FGSM_Attack(adv_params,  store_loss_progress=True)
     # adv_model = Pgd_Attack(adv_params,  store_loss_progress=True)
##

    # init_step_list = [1e-1, 1e-2, 1e-3]
    # final_step_list = [1e-2, 1e-3, 1e-4]
    init_step_list = [1e-2]
    final_step_list = [1e-3]
    # steps_list = [20, 40]
    steps_list = [20]
    decay_type_list = [True]

    exp_dict = []
    for init_step in init_step_list:
        for final_step in final_step_list:
          for steps in steps_list:
           for decay_type in decay_type_list:
            exp_dict.append(({'initial_step_size': init_step,
                              'final_step_size': final_step,
                              'iters': steps,
 			      'lin_decay' : decay_type, 
                              },
                            f'init_{init_step}_final_{final_step}_steps_{steps}_lindc_{decay_type}'))
    exp_dict = get_exp_params('mom_adam')
    # exp_dict = get_exp_params('adam_lr')
    # exp_dict = get_exp_params('compare_adam_impl')
    exp_dict = get_exp_params(args.exp_name)

    for exp_params in exp_dict:
        dict_, name_ = exp_params
        for key_ in dict_.keys():
            adv_model.params[key_] = dict_[key_]

        print(adv_model.params)

        loss_list = [0] * val_iters

        # for loop over images
        for new_idx, idx in enumerate(gt_results.index):

            # load image
            data, model, target, domain, cuda_verif_layers = load_data(gt_results, nn_name, idx, args)
            adv_model.set_layers(cuda_verif_layers)

            # run the lp
            init_tensor, lbs_all, ubs_all, dual_vars, lp_primal = run_lp(cuda_verif_layers, domain, batch_size, args)

            if not lp_init:
                init_tensor = None

            #     run GNN
            if adv_method == 'GNN' or adv_method == 'GNN_momentum':
             with th.no_grad():
              adv_examples, is_adv = adv_model.create_adv_examples(data, model,
                                                                 return_criterion='not_early',
                                                                 target=target,
                                                                 init_tensor=init_tensor,
                                                                 lbs_all=lbs_all, ubs_all=ubs_all,
                                                                 dual_vars=dual_vars, lp_primal=lp_primal,
                                                                 gpu=(not args.cpu))
            else:
                    adv_examples, is_adv, num_iters = adv_model.create_adv_examples(data, model, return_criterion='not_early',
                                                                                    target=target, gpu=True,
                                                                                    return_iters=True,
                                                                                    init_tensor=init_tensor)
            loss_progress = adv_model.loss_progress

            loss_progress = [l * factor for l in loss_progress]
            loss_list = [float(sum(x)) for x in zip(loss_list, loss_progress)]
            # loss_list += loss_progress * factor

        print(name_)
        result_list.append((name_, loss_list))

    print(time.time() - time_start)
    plot_val(result_list, val_iters, args, title=f'Validation - {table_name}')


def eth_dataset(args):
    time_start = time.time()
    val_iters = 100
    val_size = 10
    table_name = args.pdprops
    batch_size = 10
    lp_init = False

    nn_name = 'cifar10_8_255'

    # load all properties
    path = './cifar_results/adv_results/'
    path = './datasets/CIFAR10/SAT/'
    # gt_results = pd.read_pickle(path + table_name).dropna(how='all')[:val_size]
    # factor = 1. / len(gt_results.index)
    factor = 1.

    csvfile = open('././data/%s_test.csv' % ('cifar10'), 'r')
    tests = list(csv.reader(csvfile, delimiter=','))
    batch_ids = range(val_size)
    enum_batch_ids = [(bid, None) for bid in batch_ids]

    # size_dataset = len([item for sublist in [dict_files[key] for key in dict_files.keys()] for item in sublist])
    # size_dataset = min(val_size, len(gt_results.index))

    result_list = []

    adv_method = 'mi_fgsm'
    # adv_method = 'GNN_momentum'
    adv_method = 'GNN'
    if adv_method == 'GNN':
     adv_params = {
        'iters': val_iters,
        # 'initial_step_size': args.adv_lr_init,
        # 'final_step_size': args.adv_lr_fin,
        'load_GNN': args.SAT_GNN_name,
        'lp_primal': True,
        'feature_grad': True,
        # 'pick_inits': args.pick_inits,
        'num_adv_ex': batch_size,
     }
     adv_model = GNN_attack(params=adv_params, store_loss_progress=True)
     adv_model.eval_mode()
    elif adv_method == 'GNN_momentum':
     adv_params = {
        'iters': val_iters,
        # 'initial_step_size': args.adv_lr_init,
        # 'final_step_size': args.adv_lr_fin,
        'load_GNN': args.SAT_GNN_name,
        'lp_primal': True,
        'feature_grad': True,
        # 'pick_inits': args.pick_inits,
        'num_adv_ex': batch_size,
     }
     adv_model = GNN_momentum(params=adv_params, store_loss_progress=True)
     adv_model.eval_mode()
    else:
     adv_params = {'original_alpha': False, 'lr' : 5.0, 'decay_alpha': True, 'check_adv': 20, 'mu':0.5}
     adv_model = MI_FGSM_Attack(adv_params,  store_loss_progress=True)
     # adv_model = Pgd_Attack(adv_params,  store_loss_progress=True)
##

    # init_step_list = [1e-1, 1e-2, 1e-3]
    # final_step_list = [1e-2, 1e-3, 1e-4]
    init_step_list = [1e-2]
    final_step_list = [1e-3]
    # steps_list = [20, 40]
    steps_list = [100]
    decay_type_list = [True]

    exp_dict = []
    for init_step in init_step_list:
        for final_step in final_step_list:
          for steps in steps_list:
           for decay_type in decay_type_list:
            exp_dict.append(({'initial_step_size': init_step,
                              'final_step_size': final_step,
                              'iters': steps,
                              'lin_decay' : decay_type,
                              },
                            f'init_{init_step}_final_{final_step}_steps_{steps}_lindc_{decay_type}'))
    # exp_dict = get_exp_params('mom_adam')
    # exp_dict = get_exp_params('adam_lr')
    # exp_dict = get_exp_params('compare_adam_impl')
    # exp_dict = get_exp_params(args.exp_name)

    for exp_params in exp_dict:
        dict_, name_ = exp_params
        for key_ in dict_.keys():
            adv_model.params[key_] = dict_[key_]

        print(adv_model.params)

        loss_list = [0] * val_iters

        enum_batch_ids = [(4,4)]
        # for loop over images
        # for new_idx, idx in enumerate(gt_results.index):
        for new_idx, idx in enum_batch_ids:

            # load image
            data, model, target, domain, cuda_verif_layers = load_eth_data(nn_name, new_idx, args, tests)
            if data is None:
                continue
            adv_model.set_layers(cuda_verif_layers)

            # run the lp
            init_tensor, lbs_all, ubs_all, dual_vars, lp_primal = run_lp(cuda_verif_layers, domain, batch_size, args)

            print(ubs_all[-1], lbs_all[-1])
            input("waot")

            if not lp_init:
                init_tensor = None

            #     run GNN
            if adv_method == 'GNN' or adv_method == 'GNN_momentum':
             with th.no_grad():
              adv_examples, is_adv = adv_model.create_adv_examples(data, model,
                                                                 return_criterion='one',
                                                                 init_tensor=init_tensor,
                                                                 lbs_all=lbs_all, ubs_all=ubs_all,
                                                                 dual_vars=dual_vars, lp_primal=lp_primal,
                                                                 gpu=(not args.cpu))
            else:
                    adv_examples, is_adv, num_iters = adv_model.create_adv_examples(data, model, return_criterion='not_early',
                                                                                    gpu=True,
                                                                                    return_iters=True,
                                                                                    init_tensor=init_tensor, target=4)
            loss_progress = adv_model.loss_progress

            loss_progress = [l * factor for l in loss_progress]
            loss_list = [float(sum(x)) for x in zip(loss_list, loss_progress)]
            # loss_list += loss_progress * factor
            print(new_idx, idx, loss_progress[-1], is_adv)

        print(name_)
        result_list.append((name_, loss_list))

    print(time.time() - time_start)
    plot_val(result_list, val_iters, args, title=f'Validation - {table_name}')


def plot_val(loss_models, num_iter, args, win='prox', title='Validation'):
    x_ = []
    for i in range(num_iter):
        x_.append(i)

    trace = []
    idx = 0
    # colour = ['blue', 'green', 'red', 'yellow', 'purple', 'black', 'pink']
    colour = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'black', 'grey']
    cl_len = len(colour)
    for name_, q_ in loss_models:
        trace_i = dict(x=x_, y=q_, mode="lines", type='custom',
                       marker={'color': colour[idx % cl_len], 'symbol': 104, 'size': "10"},
                       text=["one", "two", "three"], name=name_)
        trace.append(trace_i)
        idx += 1

    layout = dict(title=title, xaxis={'title': 'iterations'}, yaxis={'title': 'loss'})

    visdom_opts = {'server': 'http://atlas.robots.ox.ac.uk',
                   'port': 9016, 'env': '{}'.format(args.exp_name)}
    visdom_opts = {'server': args.visdom_server,  # 'http://atlas.robots.ox.ac.uk',
                   'port': args.visdom_port,  # 9016,
                   'env': '{}'.format(args.exp_name)}
    vis = visdom.Visdom(**visdom_opts)
    vis._send({'data': trace, 'layout': layout, 'win': win})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdprops', type=str, default='val_SAT_jade.pkl',
                        help='pandas table with all props we are interested in')
    parser.add_argument('--cpu', action='store_true', help='run experiments on cpus rather than CUDA')
    parser.add_argument('--SAT_GNN_name', type=str, required=True, help='load a pretrained GNN if not None')

    parser.add_argument('--visdom_server', type=str, default='http://themis.robots.ox.ac.uk',
                        help='turn visdom on (default is off)')
    parser.add_argument('--visdom_port', type=int, default=9018,
                        help='turn visdom on (default is off)')
    parser.add_argument('--exp_name', nargs='?', default='validation_default', type=str,
                        help='name of the experiment for the logger')
    parser.add_argument('--max_solver_batch', type=float, default=10000,
                        help='max batch size for bounding computations')

    args = parser.parse_args()

    # validation_exp(args)
    eth_dataset(args)


if __name__ == "__main__":
    main()
