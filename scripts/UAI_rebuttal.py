import os
import sys
import argparse


def hparam_mi_fgsm_original(exp_name):
    # FGSM with momentum, where alpha = eps/iters is fixed
    restarts = '--random_restarts 10000'
    timeout = '--timeout 100'
    data_list = [('val', 'val_SAT_jade.pkl')]
    nn = "cifar_base_kw"
    count_ = 1
    adv_method = 'mi_fgsm_attack'

    if 'easy' in exp_name:
        data_list = [('val_easy', 'easy_val_SAT_jade.pkl')]
    else:
        data_list = [('val', 'val_SAT_jade.pkl')]

    if 'first_round' in exp_name:
        num_props = '--num_props 50'
        if 'part0' in exp_name:
            gpu_id = 0
            cpus = "0-2"
            step_list = [50, 100]
        elif 'part1' in exp_name:
            gpu_id = 1
            cpus = "3-5"
            step_list = [250, 500, 1000]
        else:
            gpu_id = 2
            cpus = "6-8"
            step_list = [10, 100, 1000]

        lr_ = 0

        for name_data, d_i in data_list:
            for steps_ in step_list:
                for mu in [0.25, 0.5, 1.0]:
                    pdprops = f"--pdprops {d_i}"
                    name = f"--table_name UAI/hparams_mi_fgsm/original/round1/{name_data}_mi_fgsm_batch_1_steps_{steps_}_lr_{lr_}_mu_{mu}.pkl"
                    steps = f"--pgd_iters {steps_}"
                    count = f"--count_particles {count_}"
                    lr = f"--pgd_optimizer_lr {lr_}"
                    mom = f" --pgd_momentum {mu} "
                    method = f" --adv_method {adv_method}"

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} --check_adv 1 "
                               f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method} {mom}")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)

    if 'second_round' in exp_name:
        num_props = '--num_props 50'
        if 'part0' in exp_name:  # hparam_mi_fgsm_second_round_part0
            gpu_id = 0
            cpus = "0-2"
            step_list = [100]
        if 'part1' in exp_name:  # hparam_mi_fgsm_second_round_part1
            gpu_id = 1
            cpus = "3-5"
            step_list = [200]
        # don't use GPU 2 because it's a titan Xp

        for name_data, d_i in data_list:
            for steps_ in step_list:
                for lr_ in [10.0, 7.5, 5.0]:
                    for mu in [0.3, 0.4, 0.5, 0.6]:
                        pdprops = f"--pdprops {d_i}"
                        name = f"--table_name UAI/hparams_mi_fgsm/round2/val_mi_fgsm_batch_1_steps_{steps_}_lr_{lr_}_mu_{mu}.pkl"
                        steps = f"--pgd_iters {steps_}"
                        count = f"--count_particles {count_}"
                        lr = f"--pgd_optimizer_lr {lr_}"
                        mom = f" --pgd_momentum {mu} "
                        method = f" --adv_method {adv_method}"

                        command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                                   f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} --check_adv 1 "
                                   f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method} {mom}")
                        print(command)
                        out_ = os.system(command)
                        assert(out_ == 0)

        if 'part2' in exp_name:
            gpu_id = 2
            cpus = "6-8"
            step_list = [250]

        for name_data, d_i in data_list:
            for steps_ in step_list:
                for lr_ in [10.0, 5.0, 2.5, 1.0]:
                    for mu in [0.25, 0.5, 1.0]:
                        pdprops = f"--pdprops {d_i}"
                        name = f"--table_name UAI/hparams_mi_fgsm/val_mi_fgsm_batch_1_steps_{steps_}_lr_{lr_}_mu_{mu}.pkl"
                        steps = f"--pgd_iters {steps_}"
                        count = f"--count_particles {count_}"
                        lr = f"--pgd_optimizer_lr {lr_}"
                        mom = f" --pgd_momentum {mu} "
                        method = f" --adv_method {adv_method}"

                        command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                                   f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} --check_adv 1 "
                                   f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method} {mom}")
                        print(command)
                        out_ = os.system(command)
                        assert(out_ == 0)


def hparam_mi_fgsm_set_alpha(exp_name):
    # FGSM with momentum, where alpha = eps/iters is fixed
    restarts = '--random_restarts 10000'
    timeout = '--timeout 100'
    data_list = [('val', 'val_SAT_jade.pkl')]
    nn = "cifar_base_kw"
    count_ = 1
    adv_method = 'mi_fgsm_attack'

    if 'easy' in exp_name:
        data_list = [('val_easy', 'easy_val_SAT_jade.pkl')]
    else:
        data_list = [('val', 'val_SAT_jade.pkl')]

    if 'first_round' in exp_name:
        num_props = '--num_props 35'
        if 'part0' in exp_name:
            gpu_id = 0
            cpus = "0-2"
            step_list = [50, 100]
        elif 'part1' in exp_name:
            gpu_id = 1
            cpus = "3-5"
            step_list = [250, 500, 1000]
        else:
            gpu_id = 0
            cpus = "0-2"
            step_list = [10, 100, 1000]

        step_size_list = [1e-1, 1e-2, 1e-3]

        for name_data, d_i in data_list:
            for steps_ in step_list:
                for lr_ in step_size_list:
                    for mu in [0.25, 0.5, 1.0]:
                        pdprops = f"--pdprops {d_i}"
                        name = f"--table_name UAI/hparams_mi_fgsm/set_alpha/round1/{name_data}_mi_fgsm_batch_1_steps_{steps_}_lr_{lr_}_mu_{mu}.pkl"
                        steps = f"--pgd_iters {steps_}"
                        count = f"--count_particles {count_}"
                        lr = f"--pgd_optimizer_lr {lr_}"
                        mom = f" --pgd_momentum {mu} "
                        method = f" --adv_method {adv_method} --mi_fgsm_set_alpha"

                        command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                                   f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} --check_adv 1 "
                                   f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method} {mom}")
                        print(command)
                        out_ = os.system(command)
                        assert(out_ == 0)


def hparam_mi_fgsm_optimized(exp_name):
    # mi-fgsm FGSM with momentum and decaying stepsize and specific starting point
    restarts = '--random_restarts 10000'
    timeout = '--timeout 100'
    data_list = [('val', 'val_SAT_jade.pkl')]
    nn = "cifar_base_kw"
    count_ = 1
    adv_method = 'mi_fgsm_attack'

    if 'first_round' in exp_name:
        num_props = '--num_props 50'
        if 'part0' in exp_name:
            # gpu_id = 0
            # cpus = "0-2"
            gpu_id = 2
            cpus = "6-8"
            step_list = [1000, 50]
        if 'part1' in exp_name:
            gpu_id = 1
            cpus = "3-5"
            step_list = [100]
        if 'part2' in exp_name:
            gpu_id = 2
            cpus = "6-8"
            step_list = [250]

        for name_data, d_i in data_list:
            for steps_ in step_list:
                for lr_ in [10.0, 5.0, 2.5, 1.0]:
                    for mu in [0.25, 0.5, 1.0]:
                        pdprops = f"--pdprops {d_i}"
                        name = f"--table_name UAI/hparams_mi_fgsm/{name_data}_mi_fgsm_batch_1_steps_{steps_}_lr_{lr_}_mu_{mu}.pkl"
                        steps = f"--pgd_iters {steps_}"
                        count = f"--count_particles {count_}"
                        lr = f"--pgd_optimizer_lr {lr_}"
                        mom = f" --pgd_momentum {mu} "
                        method = f" --adv_method {adv_method}"

                        command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                                   f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} --check_adv 1 "
                                   f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method} {mom}")
                        print(command)
                        out_ = os.system(command)
                        assert(out_ == 0)

    if 'second_round' in exp_name:
        num_props = '--num_props 50'
        if 'part0' in exp_name:  # hparam_mi_fgsm_second_round_part0
            gpu_id = 0
            cpus = "0-2"
            step_list = [100]
        if 'part1' in exp_name:  # hparam_mi_fgsm_second_round_part1
            gpu_id = 1
            cpus = "3-5"
            step_list = [200]

        for name_data, d_i in data_list:
            for steps_ in step_list:
                for lr_ in [10.0, 7.5, 5.0]:
                    for mu in [0.3, 0.4, 0.5, 0.6]:
                        pdprops = f"--pdprops {d_i}"
                        name = f"--table_name UAI/hparams_mi_fgsm/round2/val_mi_fgsm_batch_1_steps_{steps_}_lr_{lr_}_mu_{mu}.pkl"
                        steps = f"--pgd_iters {steps_}"
                        count = f"--count_particles {count_}"
                        lr = f"--pgd_optimizer_lr {lr_}"
                        mom = f" --pgd_momentum {mu} "
                        method = f" --adv_method {adv_method}"

                        command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                                   f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} --check_adv 1 "
                                   f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method} {mom}")
                        print(command)
                        out_ = os.system(command)
                        assert(out_ == 0)

        if 'part2' in exp_name:
            gpu_id = 2
            cpus = "6-8"
            step_list = [250]

        for name_data, d_i in data_list:
            for steps_ in step_list:
                for lr_ in [10.0, 5.0, 2.5, 1.0]:
                    for mu in [0.25, 0.5, 1.0]:
                        pdprops = f"--pdprops {d_i}"
                        name = f"--table_name UAI/hparams_mi_fgsm/val_mi_fgsm_batch_1_steps_{steps_}_lr_{lr_}_mu_{mu}.pkl"
                        steps = f"--pgd_iters {steps_}"
                        count = f"--count_particles {count_}"
                        lr = f"--pgd_optimizer_lr {lr_}"
                        mom = f" --pgd_momentum {mu} "
                        method = f" --adv_method {adv_method}"

                        command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                                   f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} --check_adv 1 "
                                   f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method} {mom}")
                        print(command)
                        out_ = os.system(command)
                        assert(out_ == 0)


def hparam_pgd(exp_name):
    restarts = '--random_restarts 10000'
    timeout = '--timeout 100'
    data_list = [('val', 'val_SAT_jade.pkl')]
    nn = "cifar_base_kw"
    count_ = 1
    adv_method = 'pgd_attack'

    if 'easy' in exp_name:
        data_list = [('val_easy', 'easy_val_SAT_jade.pkl')]
    else:
        data_list = [('val', 'val_SAT_jade.pkl')]

    if 'first_round' in exp_name:
        num_props = '--num_props 50'
        if 'part0' in exp_name:
            gpu_id = 1
            cpus = "3-5"
            step_list = [50, 100]
        elif 'part1' in exp_name:
            gpu_id = 0
            cpus = "0-2"
            step_list = [250, 500, 1000]
        else:
            gpu_id = 2
            cpus = "6-8"
            step_list = [10, 100, 1000]
        lr_list = [1, 1e-1, 1e-2, 1e-3]
        round_ = 'round1'

    if 'second_round' in exp_name:
        num_props = '--num_props 50'
        if 'part0' in exp_name:  # hparam_mi_fgsm_second_round_part0
            gpu_id = 0
            cpus = "0-2"
            step_list = [TODO]
        if 'part1' in exp_name:  # hparam_mi_fgsm_second_round_part1
            gpu_id = 1
            cpus = "3-5"
            step_list = [TODO]
        # don't use GPU 2 because it's a titan Xp
        lr_list = TODO
        round_ = 'round2'

    for name_data, d_i in data_list:
        for steps_ in step_list:
            for lr_ in lr_list:
                pdprops = f"--pdprops {d_i}"
                name = f"--table_name UAI/hparams_pgd/{round_}/{name_data}_pgd_steps_{steps_}_lr_{lr_}.pkl"
                steps = f"--pgd_iters {steps_}"
                count = f"--count_particles {count_}"
                lr = f"--pgd_optimizer_lr {lr_}"
                method = f" --adv_method {adv_method}"

                command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                           f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} --check_adv 1 "
                           f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method}")
                print(command)
                out_ = os.system(command)
                assert(out_ == 0)


def run_paper_experiments(exp_name):
    restarts = '--random_restarts 10000'
    timeout = '--timeout 100'
    count_ = 1
    adv_method = 'mi_fgsm_attack'
    num_props = '--num_props 1000'

    data_list = []
    if 'base' in exp_name:
        if 'easy' in exp_name:
            data_list.append(('base_easy', 'easy_base_easy_SAT_jade.pkl'))
        else:
            data_list.append(('base', 'base_easy_SAT_jade.pkl'))
    if 'wide' in exp_name:
        if 'easy' in exp_name:
            data_list.append(('wide_easy', 'easy_wide_SAT_jade.pkl'))
        else:
            data_list.append(('wide', 'wide_SAT_jade.pkl'))
    if 'deep' in exp_name:
        if 'easy' in exp_name:
            data_list.append(('deep_easy', 'easy_deep_SAT_jade.pkl'))
        else:
            data_list.append(('deep', 'deep_SAT_jade.pkl'))

    seed_list = []
    if 'seed_2222' in exp_name:
        seed_list.append(2222)
    if 'seed_3333' in exp_name:
        seed_list.append(3333)
    if 'seed_4444' in exp_name:
        seed_list.append(4444)
    if 'seed_6666' in exp_name:
        seed_list.append(6666)
    if 'seed_7777' in exp_name:  # just for debugging purposes
        seed_list.append(7777)

    for seed in seed_list:
        for name_data, d_i in data_list:

            if 'base' in name_data:
                nn = "cifar_base_kw"
            elif 'deep' in name_data:
                nn = "cifar_deep_kw"
            elif 'wide' in name_data:
                nn = "cifar_wide_kw"

            if 'mi_fgsm_original' in exp_name:
                gpu_id = 0
                cpus = "0-2"
                steps_ = 100
                lr_ = 5.0
                mu_ = 0.3
                adv_method = 'mi_fgsm_attack'
                params = f" --pgd_iters {steps_} --pgd_momentum {mu_} --pgd_optimizer_lr {lr_} "
                name = f"--table_name UAI/experiments/{name_data}_mi_fgsm_batch_1_steps_{steps_}_lr_{lr_}_mu_{mu_}_seed{seed}.pkl"
            elif 'mi_fgsm_set_alpha' in exp_name:
                gpu_id = 1
                cpus = "3-5"
                steps_ = 100
                lr_ = 0.1
                mu_ = 0.5
                adv_method = 'mi_fgsm_attack'
                params = f" --pgd_iters {steps_} --pgd_momentum {mu_} --pgd_optimizer_lr {lr_}  --mi_fgsm_set_alpha "
                name = f"--table_name UAI/experiments/{name_data}_mi_fgsm_set_alpha_ctd_1_steps_{steps_}_lr_{lr_}_mu_{mu_}_seed{seed}.pkl"
            elif 'pgd' in exp_name:
                gpu_id = 1
                cpus = "3-5"
                steps_ = 100
                lr_ = 0.01
                if 'easy' in exp_name:
                    lr_ = 0.1
                adv_method = 'pgd_attack'
                params = f" --pgd_iters {steps_} --pgd_optimizer_lr {lr_} "
                name = f"--table_name UAI/experiments/{name_data}_pgd_batch_1_steps_{steps_}_lr_{lr_}_seed{seed}.pkl"
            elif 'GNN' in exp_name:
                gpu_id = 0
                cpus = "0-2"
                # gpu_idx = 2
                # cpus = "6-8"
                steps_ = 40
                lr_init = 1e-2
                lr_fin = 1e-3
                lr_fin = 1e-2
                adv_method = "GNN"
                if name_data in ['base', 'base_easy']:
                    gnn = 'train_jade_n25e4_horizon40/model-best.pkl'
                    name_gnn = 'train_n25e4_horizon40'
                    gpu_id = 1
                    cpus = "3-5"
                elif name_data in ['deep', 'deep_easy']:
                    gnn = 'fintuning_deep_large_model_finetuning_15_minute_dc0/model-best.pkl'
                    name_gnn = 'finetuning_15mins'
                    if 'new_GNN' in exp_name:
                        gnn = 'finetuningdeep_gnn30_lr_2/model-best.pkl'
                        name_gnn = 'finetuning_15mins_new'
                        gpu_id = 0
                        cpus = "0-2"
                elif name_data in ['wide', 'wide_easy']:
                    gnn = 'jade/train_jade_n25e4_horizon40/model-best.pkl'
                    name_gnn = 'train_n25e4_horizon40'
                    gpu_id = 0
                    cpus = "0-2"
                params = f"--SAT_GNN_name {gnn} --GNN_grad_feat --GNN_optimized --GNN_iters {steps_} "
                params += f" --run_lp --GNN_lr_init {lr_init} --GNN_lr_fin {lr_fin} "
                name = f"--table_name UAI/experiments/{name_data}_GNN_{name_gnn}_steps_{steps_}_count_{count_}_lr_{lr_init}_{lr_fin}_seed{seed}_nolpadam.pkl"

            pdprops = f"--pdprops {d_i}"
            count = f"--count_particles {count_}"
            method = f" --adv_method {adv_method}"

            command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                       f"--nn_name {nn} {pdprops} {restarts} --check_adv 1 {params} "
                       f"{name} {count} {timeout} --seed {seed} {num_props} {method} ")
            print(command)
            out_ = os.system(command)
            assert(out_ == 0)


def GNN_GAP_comparisons(one_step=False):
    # run GNN experiments on easy base experiments with different amounts of epsilons
    restarts = '--random_restarts 100'
    if one_step:
        restarts = '--random_restarts 1'
    timeout = '--timeout 1'
    count_ = 1
    num_props = '--num_props 100'
    if True:
        seed = 1234
        nn = "cifar_base_kw"
        d_i = 'easy_base_easy_SAT_jade.pkl'
        gpu_id = 0
        cpus = "0-2"
        steps_ = 40
        if one_step:
            steps_ = 1
        lr_init = 1e-1
        lr_fin = 1e-1
        adv_method = "GNN"
        gnn = 'train_jade_n25e4_horizon40/model-best.pkl'
        name_gnn = 'train_n25e4_horizon40'
        params = f"--SAT_GNN_name {gnn} --GNN_grad_feat --GNN_optimized --GNN_iters {steps_} "
        params += f" --run_lp --GNN_lr_init {lr_init} --GNN_lr_fin {lr_fin} "
        # name = f"--table_name UAI/GAP_comparison/debug.pkl"
    if True:
        pdprops = f"--pdprops {d_i}"
        count = f"--count_particles {count_}"
        method = f" --adv_method {adv_method}"

    for eps_ in range(0, 20, 1):
        eps = float(eps_)/100
        params += f' --change_eps_const {eps} '
        name = f"--table_name UAI/GAP_comparison/1_second_eps_{eps}.pkl"
        if one_step:
            name = f"--table_name UAI/GAP_comparison/1_step_lr1_eps_{eps}.pkl"
        command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                   f"--nn_name {nn} {pdprops} {restarts} --check_adv 1 {params} "
                   f"{name} {count} {timeout} --seed {seed} {num_props} {method} ")
        print(command)
        out_ = os.system(command)
        assert(out_ == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='experiment name')
    args = parser.parse_args()

    if 'hparam_mi_fgsm_optimized' in args.exp_name:
        hparam_mi_fgsm_optimized(args.exp_name)
    elif 'hparam_mi_fgsm_original' in args.exp_name:
        hparam_mi_fgsm_original(args.exp_name)
    elif 'hparam_mi_fgsm_set_alpha' in args.exp_name:
        hparam_mi_fgsm_set_alpha(args.exp_name)
    elif 'hparam_pgd' in args.exp_name:
        hparam_pgd(args.exp_name)
    elif 'experiments' in args.exp_name:
        run_paper_experiments(args.exp_name)
    elif 'GAP' in args.exp_name:
        GNN_GAP_comparisons(one_step=True)
    else:
        input("type of experiment is not implemented")


if __name__ == "__main__":
    main()
