import os
import sys
import argparse


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
        gpu_id = 3
        cpus = "12-19"
        # pdprops = "--pdprops /../cifar_train_pdtables/train_props.pkl"
        name = "--table_name base_easy_SAT_bugfixed.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 50000"
        count = "--count_particles 200"
        restarts = "--random_restarts 2 --cpu "

    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/create_dataset.py "
               f"--nn_name {nn} {pdprops} {lr} {steps} {bin_search} "
               f"{name} {restarts} {count} --manual_impl ")

    # command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/create_dataset.py "
    #            f"--nn_name {nn} {pdprops} {lr} {steps} {bin_search} "
    #            f"{name} {restarts}")

    print(command)
    os.system(command)


def create_dataset_jade(exp_name):

    if exp_name == 'easy':
        pdprops = "--pdprops jodie-base_easy.pkl"
        nn = "cifar_base_kw"
        name = "--table_name easy_SAT_jade.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 2000"
        count = "--count_particles 200"
        restarts = "--random_restarts 100 "
        lr = "--pgd_optimizer_lr 1e-2"
    elif exp_name == 'wide':
        pdprops = "--pdprops jodie-wide.pkl"
        nn = "cifar_wide_kw"
        name = "--table_name wide_SAT_jade.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 2000"
        count = "--count_particles 200"
        restarts = "--random_restarts 100 "
        lr = "--pgd_optimizer_lr 1e-2"
    elif exp_name == 'deep':
        pdprops = "--pdprops jodie-deep.pkl"
        nn = "cifar_deep_kw"
        name = "--table_name deep_SAT_jade.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 2000"
        count = "--count_particles 200"
        restarts = "--random_restarts 100 "
        lr = "--pgd_optimizer_lr 1e-2"
    elif exp_name == 'train':
        pdprops = "--pdprops /../cifar_train_pdtables/train_props.pkl"
        nn = "cifar_base_kw"
        name = "--table_name train_SAT_jade.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 2000"
        count = "--count_particles 200"
        restarts = "--random_restarts 100 "
        lr = "--pgd_optimizer_lr 1e-2"
    elif exp_name == 'val':
        pdprops = "--pdprops /../cifar_train_pdtables/val_props.pkl"
        nn = "cifar_base_kw"
        name = "--table_name val_SAT_jade.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 2000"
        count = "--count_particles 200"
        restarts = "--random_restarts 100 "
        lr = "--pgd_optimizer_lr 1e-2"
    elif exp_name == 'train_quick':
        pdprops = "--pdprops /../cifar_train_pdtables/val_props.pkl"
        pdprops = " --pdprops /../cifar_results/adv_results/train_eps02.pkl"
        nn = "cifar_base_kw"
        name = "--table_name large_quick_train.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 1000"
        count = "--count_particles 100"
        restarts = "--random_restarts 0 --start_at_eps0"
        lr = "--pgd_optimizer_lr 1e-2"
    elif exp_name == 'rebuttal':
        pdprops = "--pdprops madry_finetuning_0.2.pkl"
        nn = "cifar_madry"
        name = "--table_name madry_SAT_jade.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 2000"
        count = "--count_particles 200"
        restarts = "--random_restarts 100  --start_at_eps0 "
        lr = "--pgd_optimizer_lr 1e-2"
    elif exp_name == 'rebuttal_easy':
        pdprops = "--pdprops madry_finetuning_0.2.pkl"
        nn = "cifar_madry"
        name = "--table_name madry_easy_SAT_jade.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 1000"
        count = "--count_particles 100"
        restarts = "--random_restarts 0  --start_at_eps0 "
        lr = "--pgd_optimizer_lr 1e-2"
    elif exp_name == 'rebuttal_med':
        pdprops = "--pdprops madry_finetuning_0.2.pkl"
        nn = "cifar_madry"
        name = "--table_name madry_med_SAT_jade.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 1000"
        count = "--count_particles 200"
        restarts = "--random_restarts 10  --start_at_eps0 "
        lr = "--pgd_optimizer_lr 1e-2"
    elif exp_name == 'rebuttal_mnist_easy':
        pdprops = "--pdprops ../cifar_results/adv_results/mnist_wide_finetuning_0.2.pkl"
        nn = "mnist_base_kw.pth"
        name = "--table_name mnist_base_easy_SAT.pkl"
        bin_search = "--bin_search_eps 1e-3"
        steps = "--pgd_iters 1000"
        count = "--count_particles 100"
        restarts = "--random_restarts 0  --start_at_eps0 --mnist"
        lr = "--pgd_optimizer_lr 1e-2"


    command = (f"python adv_exp/create_dataset.py "
               f"--nn_name {nn} {pdprops} {lr} {steps} {bin_search} "
               f"{name} {restarts} {count} --manual_impl ")

    print(command)
    input("wait")
    os.system(command)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--jade', action='store_true', help='if running on jade')
    args = parser.parse_args()

    print("About to run create_dataset_jade")

    if args.jade:
        create_dataset_jade(args.exp_name)
    else:
        create_dataset('easy_manual')
        # create_dataset('training_manual')


if __name__ == "__main__":
    main()
