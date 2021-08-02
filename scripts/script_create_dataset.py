import os
import sys


def create_dataset(exp_name):
    gpu_id = 2
    cpus = "6-9"

    name = restarts = lr = steps = ""

    pdprops = "--pdprops jodie-base_easy.pkl"
    nn = "cifar_base_kw"

    if exp_name == 'base_easy_SAT_1212':
        # base_easy_SAT_1212.pkl
        lr = "--pgd_abs_lr 1e-3"
        steps = "--pgd_steps 1e4"
        bin_search = "--bin_search_eps 1e-3"

    elif exp_name == 'base_easy_SAT_1412_1':
        # base_easy_SAT_1412_1.pkl
        name = "--table_name base_easy_SAT_1412_1.pkl"
        lr = "--pgd_abs_lr 1e-3"
        steps = "--pgd_steps 1e4"
        bin_search = "--bin_search_eps 1e-3"
        restarts = "--random_restarts 20"

    elif exp_name == 'base_easy_SAT_1412_2':
        # base_easy_SAT_1412_2.pkl
        name = "--table_name base_easy_SAT_1412_2.pkl"
        steps = "--pgd_steps 1e4"
        bin_search = "--bin_search_eps 1e-3"

    elif exp_name == 'base_deep':
        nn = "cifar_deep_kw"
        pdprops = "--pdprops jodie-deep.pkl"
        name = "--table_name deep_SAT_2112_steps_e5.pkl"
        steps = "--pgd_steps 1e5"
        bin_search = "--bin_search_eps 1e-3"

    elif exp_name == 'base_wide':
        nn = "cifar_wide_kw"
        pdprops = "--pdprops jodie-wide.pkl"
        name = "--table_name wide_SAT_2112_steps_e5.pkl"
        steps = "--pgd_steps 1e5"
        bin_search = "--bin_search_eps 1e-3"

    elif exp_name == 'training':
        gpu_id = 3
        cpus = "9-11"
        pdprops = "--pdprops /../cifar_train_pdtables/train_props.pkl"
        name = "--table_name train_props_2112_steps_e5.pkl"
        steps = "--pgd_steps 1e5"
        bin_search = "--bin_search_eps 1e-3"

    elif exp_name == 'training_manual':
        gpu_id = 2
        cpus = "9-11"
        pdprops = "--pdprops /../cifar_train_pdtables/train_props.pkl"
        name = "--table_name train_SAT_bugfixed.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 50000"
        count = "--count_particles 200"
        restarts = "--random_restarts 2"
    elif exp_name == 'easy_manual':
        gpu_id = 1
        cpus = "6-8"
        # pdprops = "--pdprops /../cifar_train_pdtables/train_props.pkl"
        name = "--table_name base_easy_SAT_bugfixed.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 50000"
        count = "--count_particles 200"
        restarts = "--random_restarts 2"

    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/create_dataset.py "
               f"--nn_name {nn} {pdprops} {lr} {steps} {bin_search} "
               f"{name} {restarts} {count} --manual_impl ")

    # command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/create_dataset.py "
    #            f"--nn_name {nn} {pdprops} {lr} {steps} {bin_search} "
    #            f"{name} {restarts}")

    print(command)
    os.system(command)


def main():
    create_dataset('easy_manual')
    # create_dataset('training_manual')


if __name__ == "__main__":
    main()
