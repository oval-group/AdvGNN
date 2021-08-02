import torch as th
import pandas as pd
from adv_exp.GNN_training.utils import process_files, load_data
from adv_exp.pgd_attack import Pgd_Attack

########################################################################
#   compute baselines values so that we can make sense of the training performance
#   we run pgd attack over 100, and 1000 steps with stepsizes 1e-3 and 1e-4
#   we return a dict which includes for every baseline the average targeted loss ...
#       and the percentage of subdomains for which it finds an adversarial example
#   TODO
#   - debug
########################################################################


def compute_baselines_from_dataset(args, dict_files):
    # list of parameters ('name', num_steps, lr)

    baselines_dict = {}

    # total number of files
    size_dataset = len([item for sublist in [dict_files[key] for key in dict_files.keys()] for item in sublist])

    for num_steps in [20, 100, 1000]:
        # for _lr in [1, 1e-1, 1e-2, 1e-3]:
        if num_steps in [20, 100]:
            _lr = 1e-1
        else:
            _lr = 1e-2
        # for opt in ['default', 'adam']:
        for opt in ['default']:
            adv_params = {
                'iters': num_steps,
                'optimizer': opt,
                'lr': _lr,
                'num_adv_ex': 2,
            }
            adv_model = Pgd_Attack(adv_params, store_loss_progress=True)

            # average loss
            _loss = 0
            # percentage of subdomains for which the baseline returns an adv example
            _per_adv = 0
            # percentage of properties for which the baseline returns an adv example
            _per_adv_props = 0

            debug_size_dataset = 0

            # for loop over properties
            for prop in dict_files.keys():
                # list of current subdomains
                subdoms = dict_files[prop]

                batch_size = args.batch_size
                mini_batch_num = int((len(subdoms)-2)/batch_size)+1

                # bool indiciating whether the baseline has returned at least 1 counter example
                #     for at least one of the subdomains for this property
                is_adv_prop = False

                # for loop over batches
                for batch_idx in range(mini_batch_num):

                    file_idx = batch_idx*batch_size
                    if batch_size == 1:
                        cur_files = [subdoms[batch_idx]]
                    elif batch_idx == (mini_batch_num-1):
                        cur_files = subdoms[file_idx:]
                    else:
                        cur_files = subdoms[file_idx:file_idx+batch_size]

                    # size of the current batch relative to the total training dataset
                    # we add the mean values for each batch and multipy them with factor
                    # or we add the sum for a batch and divide it by float(size_dataset)
                    factor = len(cur_files) / float(size_dataset)

                    # load_data
                    data_dict = process_files(args, prop, cur_files, not args.cpu)

                    # run attack
                    adv_model.set_layers(data_dict['layers'])
                    adv_examples, is_adv = adv_model.create_adv_examples(data_dict['data'], data_dict['model'],
                                                                         return_criterion='not_early',
                                                                         target=data_dict['target'],
                                                                         init_tensor=data_dict['init_tensor'],
                                                                         gpu=(not args.cpu))

                    _per_adv += float((is_adv.sum() / float(size_dataset)))
                    _loss += float(adv_model.loss_progress[-1]) * factor

                    # print(is_adv)
                    # print(len(is_adv), len(cur_files))
                    # input("Wait")
                    # assert(len(is_adv) == len(cur_files))
                    debug_size_dataset += len(is_adv)

                    is_adv_prop = is_adv_prop or is_adv.sum() > 0

                _per_adv_props += float(is_adv_prop) / float(len(dict_files.keys()))

            baselines_dict[f'PGD_iter_{num_steps}_lr_{_lr}'] = {'loss': _loss, 'per_adv': _per_adv,
                                                                '_per_adv_props': _per_adv_props}

    return baselines_dict


def compute_baselines_from_table(args, nn_name, table_name, val_size=10, batch_size=100):

    baselines_dict = {}

    for num_steps in [20, 100, 1000]:
        # for _lr in [1, 1e-1, 1e-2, 1e-3]:
        if num_steps in [20, 100]:
            _lr = 1e-1
        else:
            _lr = 1e-2
        # for opt in ['default', 'adam']:
        for opt in ['default']:
            adv_params = {
                'iters': num_steps,
                'optimizer': opt,
                'lr': _lr,
                'num_adv_ex': batch_size,
            }
            adv_model = Pgd_Attack(adv_params, store_loss_progress=True)

            if nn_name == 'cifar_madry':
                from adv_exp.mi_fgsm_attack import MI_FGSM_Attack
                adv_params = {
                    'iters': num_steps,
                    'lr': _lr,
                    'num_adv_ex': batch_size,
                    'mu': 0.5,
                }
                adv_params['original_alpha'] = False
                adv_model = MI_FGSM_Attack(adv_params,  store_loss_progress=True)


            # average loss
            _loss = 0
            # percentage of subdomains for which the baseline returns an adv example
            _per_adv = 0
            # percentage of properties for which the baseline returns an adv example
            _per_adv_props = 0

            debug_size_dataset = 0

            path = './cifar_results/adv_results/'
            path = './batch_verification_results/jade/'
            gt_results = pd.read_pickle(path + table_name).dropna(how='all')[:val_size]
            num_props = min(val_size, len(gt_results.index))
            factor = 1. / num_props

            # size_dataset = len([item for sublist in [dict_files[key] for key in dict_files.keys()] for item in sublist])
            size_dataset = batch_size * num_props

            # for loop over images
            for new_idx, idx in enumerate(gt_results.index):

                # load image
                data, model, target, domain, cuda_verif_layers = load_data(gt_results, nn_name, idx, args)
                adv_model.set_layers(cuda_verif_layers)

                # run the lp
                # init_tensor, _, _, _ = run_lp(cuda_verif_layers, domain, batch_size, args)

                #     run GNN
                adv_examples, is_adv = adv_model.create_adv_examples(data, model,
                                                                     return_criterion='not_early',
                                                                     target=target,
                                                                     # init_tensor=init_tensor,
                                                                     gpu=(not args.cpu))

                _per_adv += float((is_adv.sum() / float(size_dataset)))
                _loss += float(adv_model.loss_progress[-1]) * factor

                debug_size_dataset += len(is_adv)

                is_adv_prop = is_adv.sum() > 0

                _per_adv_props += float(is_adv_prop) / float(num_props)

            baselines_dict[f'PGD_iter_{num_steps}_lr_{_lr}'] = {'loss': _loss, 'per_adv': _per_adv,
                                                                '_per_adv_props': _per_adv_props}

    return baselines_dict, num_props
