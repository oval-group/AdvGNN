import os
import sys
import argparse


def run_foolbox_train(exp_name):

    pdprops = "--pdprops train_SAT_bugfixed_intermediate.pkl"
    pdprops = "--pdprops train_SAT_med.pkl"
    pdprops = "--pdprops jade/easy_val_SAT_jade.pkl"
    restarts = '--random_restarts 100'
    timeout = '--timeout 100'
    batch = '--batch_size 300'
    record = f'--record --record_name foolbox/{exp_name}.pkl'
    method = f"--adv_method {exp_name}"
    params = ""

    if exp_name == 'PGD':
        gpu_id = 0
        cpus = "0-2"
    elif exp_name == 'PGD_debug':
        gpu_id = 0
        cpus = "10-12"
        method = "--adv_method PGD"
    elif exp_name == 'DeepFool_debug':
        gpu_id = 5
        cpus = "3-5"
        params = '--cpu --steps 10000'
        method = "--adv_method DeepFool"
        batch = '--batch_size 1'
    elif exp_name == 'BasicIterative':
        gpu_id = 2
        cpus = "6-8"
    elif exp_name == 'FGSM':
        gpu_id = 0
        cpus = "0-2"
    elif exp_name == 'Additive_noise':
        gpu_id = 1
        cpus = "3-5"
        restarts = '--random_restarts 500000'
    elif exp_name == 'Repeated_noise':
        gpu_id = 2
        cpus = "6-8"
    elif exp_name == 'PGD_lr_1':
        gpu_id = 0
        params = "--pgd_lr 1"
        method = "--adv_method PGD"
    elif exp_name == 'PGD_lr_01':
        gpu_id = 1
        params = "--pgd_lr 0.1"
        method = "--adv_method PGD"
    elif exp_name == 'PGD_lr_0001':
        gpu_id = 2
        params = "--pgd_lr 0.001"
        method = "--adv_method PGD"
    elif exp_name == 'BI_lr_1':
        gpu_id = 2
        params = "--BI_lr 1"
        method = "--adv_method BasicIterative"
    elif exp_name == 'BI_lr_01':
        gpu_id = 1
        params = "--BI_lr 0.1"
        method = "--adv_method BasicIterative"
    elif exp_name == 'CW':
        gpu_id = 2
        batch = '--batch_size 1'
        params = '--CW_bin_search_steps 20 --steps 10000 --random_restarts 0 '
    else:
        print("exp name not defined")
        return

    cpus = f'{3*gpu_id}-{3*gpu_id + 2}'
    cpus = '10-19'
    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_foolbox.py "
               f"{pdprops} {restarts} {method} {timeout} {batch} {record} {params}")

    print(command)
    os.system(command)


def foolbox_hparam(round=1):
    # parser.add_argument('--CW_lr', type=float, help='learning rate CW attack', default=1e-2)
    # parser.add_argument('--CW_init_const', type=float, help='learning rate CW attack', default=1e-5)
    # parser.add_argument('--CW_largest_const', type=float, help='learning rate CW attack', default=2e+1)
    # parser.add_argument('--CW_const_factor', type=float, help='learning rate CW attack', default=2.0)
    # parser.add_argument('--CW_decrease_factor', type=float, help='learning rate CW attack', default=0.9)

    pdprops = "--pdprops jade/easy_val_SAT_jade.pkl"
    restarts = '--random_restarts 1'
    timeout = '--timeout 100'
    batch = '--batch_size 1'
    method = f"--adv_method CW"
    gpu_id = 2
    cpus = '10-12'
    props = "--num_props 20"
    print("TODO implement num props")

    params_list = []
    if round==1:
     for steps in [10, 100, 1000]:
        params_list.append((f"steps_{steps}", f" --CW_steps {steps} "))
     for lr in [1e-1, 1e-2, 1e-3]:
        params_list.append((f"lr_{lr}", f"--CW_lr {lr}"))
     for init in [1e-5, 1e-4, 1e-3, 1e-2]:
        params_list.append((f"init_const_{init}", f"--CW_init_const {init}"))
     for largest in [1e+2, 1e+1, 1, 1e-1]:
        params_list.append((f"largest_const_{largest}", f"--CW_largest_const {largest}"))
     for factor in [1.5, 2.0, 5.0]:
        params_list.append((f"factor_{factor}", f"--CW_const_factor {factor}"))
     for factor in [0.5, 0.9, 0.99]:
        params_list.append((f"decrease_factor_{factor}", f"--CW_decrease_factor {factor}"))
    elif round==2:
     for steps in [10, 100, 1000]:
        params_list.append((f"steps_{steps}", f" --CW_steps {steps} "))
     for lr in [1e-2, 1e-3, 1e-4]:
        params_list.append((f"lr_{lr}", f"--CW_lr {lr}"))
     for init in [1e-5, 1e-4, 1e-3, 1e-2]:
        params_list.append((f"init_const_{init}", f"--CW_init_const {init}"))
     for largest in [1e+3, 1e+2, 1e+1]:
        params_list.append((f"largest_const_{largest}", f"--CW_largest_const {largest}"))
     for factor in [1.5, 2.0, 5.0]:
        params_list.append((f"factor_{factor}", f"--CW_const_factor {factor}"))
     for factor in [0.5, 0.9, 0.99]:
        params_list.append((f"decrease_factor_{factor}", f"--CW_decrease_factor {factor}"))
    elif round==3:
     for steps in [10, 100, 1000]:
        params_list.append((f"steps_{steps}", f" --CW_steps {steps} "))
     for lr in [1e-2, 1e-3, 1e-4]:
        params_list.append((f"lr_{lr}", f"--CW_lr {lr}"))
     for init in [1e-5, 1e-4, 1e-3]:
        params_list.append((f"init_const_{init}", f"--CW_init_const {init}"))
     for largest in [1e+4, 1e+3, 1e+2]:
        params_list.append((f"largest_const_{largest}", f"--CW_largest_const {largest}"))
     for factor in [1.25, 1.5, 2.0]:
        params_list.append((f"factor_{factor}", f"--CW_const_factor {factor}"))
     for factor in [0.9, 0.99, 0.999]:
        params_list.append((f"decrease_factor_{factor}", f"--CW_decrease_factor {factor}"))

    for name, params in params_list:
        record = f'--record --record_name foolbox/hparam_CW/round{round}/_CW_{name}.pkl'
        
        command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_foolbox.py "
                   f"{pdprops} {restarts} {method} {timeout} {batch} {record} {params} {props}")

        print("params", params)
        print(command)
        out_ = os.system(command)
        assert(out_ == 0)


def run_experiments(model='base', easy=True):
    pdprops = "--pdprops jade/easy_val_SAT_jade.pkl"
    pdprops = "--pdprops jade/base_easy_SAT_jade.pkl"
    pdprops = "--pdprops jade/easy_base_easy_SAT_jade.pkl"

    if easy:
        easy_ = 'easy_'
    else:
        easy_ = ''

    if model=='base':
        pdprops = f"--pdprops jade/{easy_}base_easy_SAT_jade.pkl" 
        nn = "--nn_name cifar_base_kw"
    elif model=='wide':
        pdprops = f"--pdprops jade/{easy_}wide_SAT_jade.pkl"
        nn = "--nn_name cifar_wide_kw"
    elif model=='deep':
        pdprops = f"--pdprops jade/{easy_}deep_SAT_jade.pkl"
        nn = "--nn_name cifar_deep_kw"


    restarts = '--random_restarts 0 --CW_steps 100 '
    timeout = '--timeout 100'
    batch = '--batch_size 1'
    method = f"--adv_method CW"
    gpu_id = 2
    cpus = '10-12'
    props = "--num_props 1000"

    record = f'--record --record_name foolbox/CW_experiments/{easy_}{model}_CW.pkl'

    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_foolbox.py "
               f"{pdprops} {restarts} {method} {timeout} {batch} {record} {props} {nn}")

    print(command)
    out_ = os.system(command)
    assert(out_ == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='experiment name')
    args = parser.parse_args()

    # run_foolbox_train(args.exp_name)
    # run_foolbox_train('PGD')
    # run_foolbox_train('DeepFool')
    # run_foolbox_train('BasicIterative')
    # foolbox_hparam(round=3)
    # run_experiments(model='base')
    # run_experiments(model='wide')
    # run_experiments(model='deep')

    run_experiments(model='base', easy=False)
    run_experiments(model='wide', easy=False)
    run_experiments(model='deep', easy=False)

if __name__ == "__main__":
    main()
