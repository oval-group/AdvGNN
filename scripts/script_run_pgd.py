import os
import sys
import argparse


def run_old_exp(exp_name):
    name = restarts = lr = steps = ""

    pdprops = "--pdprops jodie-base_easy.pkl"
    nn = "cifar_base_kw"
    restarts = '--random_restarts 10'

    if exp_name == 'train_hard_attack_4_2':
        gpu_id = 0
        cpus = "0-2"
        # pdprops = "--pdprops /../cifar_train_pdtables/train_props.pkl"
        pdprops = "--pdprops train_SAT_hard.pkl"
        name = "--table_name training_pgd_iter_5e4_batch_1e2.pkl"
        steps = "--pgd_iters 50000"
        count = "--count_particles 100"
    if exp_name == 'train_hard_attack_3_3':
        gpu_id = 1
        cpus = "3-5"
        # pdprops = "--pdprops /../cifar_train_pdtables/train_props.pkl"
        pdprops = "--pdprops train_SAT_hard.pkl"
        name = "--table_name training_pgd_iter_5e3_batch_1e3.pkl"
        steps = "--pgd_iters 5000"
        count = "--count_particles 1000"
    if exp_name == 'train_hard_attack_4_3':
        gpu_id = 2
        cpus = "6-8"
        # pdprops = "--pdprops /../cifar_train_pdtables/train_props.pkl"
        pdprops = "--pdprops train_SAT_hard.pkl"
        name = "--table_name training_pgd_iter_5e4_batch_1e3.pkl"
        steps = "--pgd_iters 50000"
        count = "--count_particles 1000"
    if exp_name == 'train_hard_attack_3_2':
        gpu_id = 3
        cpus = "9-11"
        # pdprops = "--pdprops /../cifar_train_pdtables/train_props.pkl"
        pdprops = "--pdprops train_SAT_hard.pkl"
        name = "--table_name training_pgd_iter_5e3_batch_1e2.pkl"
        steps = "--pgd_iters 5000"
        count = "--count_particles 100"

    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/create_dataset_manual.py "
               f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} "
               f"{name} {restarts} {count} --run_exp")

    print(command)
    os.system(command)


def run_many_pgd_experiments(exp_name):
    name = restarts = lr = steps = lp = ""

    pdprops = "--pdprops train_SAT_bugfixed_intermediate.pkl"
    nn = "cifar_base_kw"
    restarts = '--random_restarts 100'
    timeout = '--timeout 100'
    num_props = '--num_props 200'
    if exp_name == 'lp_exp':
        for steps_ in [1000, 5000, 20000]:
            for count_ in [100, 500, 1000]:
                for lr_ in [1e-2]:  # do 1e-1 later
                    gpu_id = 0
                    cpus = "0-2"
                    name = f"--table_name pgd_steps_{steps_}_ctd_{count_}_lr_{lr_}_lpinit.pkl"
                    steps = f"--pgd_iters {steps_}"
                    count = f"--count_particles {count_}"
                    lr = f"--pgd_optimizer_lr {lr_}"
                    lp = "--run_lp"

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} {lp} "
                               f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} ")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)
    if exp_name == 'no_lp_exp':
        for steps_ in [1000, 5000, 20000]:
            for count_ in [100, 500, 1000]:
                for lr_ in [1e-2]:  # do 1e-1 later
                    gpu_id = 0
                    cpus = "0-2"
                    name = f"--table_name pgd_steps_{steps_}_ctd_{count_}_lr_{lr_}.pkl"
                    steps = f"--pgd_iters {steps_}"
                    count = f"--count_particles {count_}"
                    lr = f"--pgd_optimizer_lr {lr_}"

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} "
                               f"{name} {restarts} {count} {timeout} --seed 2222 {num_props}")

                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)
    if exp_name == 'pgd_train_eps0005':
        for steps_ in [1000]:
            for count_ in [1000]:
                for lr_ in [1e-2]:  # do 1e-1 later
                    gpu_id = 2
                    cpus = "6-8"
                    name = f"--table_name pgd_steps_{steps_}_ctd_{count_}_lr_{lr_}_lpinit.pkl"
                    name = f"--table_name {exp_name}"
                    steps = f"--pgd_iters {steps_}"
                    count = f"--count_particles {count_}"
                    lr = f"--pgd_optimizer_lr {lr_}"
                    lp = "--run_lp"

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} {lp} --change_eps_const -0.005 "
                               f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} ")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)
    if exp_name == 'pgd_check_every_iter':
        for steps_ in [1000]:
            for count_ in [1000]:
                for lr_ in [1e-2]:  # do 1e-1 later
                    gpu_id = 1
                    cpus = "3-5"
                    name = f"--table_name pgd_steps_{steps_}_ctd_{count_}_lr_{lr_}_lpinit_checkeveryiter.pkl"
                    steps = f"--pgd_iters {steps_}"
                    count = f"--count_particles {count_}"
                    lr = f"--pgd_optimizer_lr {lr_}"
                    lp = "--run_lp"

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} {lp} --check_adv 1 "
                               f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} ")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)


def run_GNN(exp_name):
    name = restarts = lr = steps = lp = ""

    pdprops = "--pdprops jodie-base_easy.pkl"
    nn = "cifar_base_kw"
    restarts = '--random_restarts 1000'
    timeout = "--timeout 100"
    num_props = '--num_props 200'

    steps_list = [20]
    count_list = [20]

    if exp_name == 'GNN_train_atlas':
        for steps_ in steps_list:
            for count_ in [10]:
                gpu_id = 2
                cpus = "6-8"
                pdprops = "--pdprops train_SAT_bugfixed_intermediate.pkl"
                name = f"--table_name {exp_name}"
                steps = f"--pgd_iters {steps_}"
                count = f"--count_particles {count_}"
                GNN = "--SAT_GNN_name 20210118_GNN_with_grad_sign/model-best.pkl --GNN_grad_feat --adv_method GNN"
                lp = "--run_lp"

                command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                           f"--nn_name {nn} {pdprops} {steps} {restarts} {lp} "
                           f"{name} {restarts} {count} {GNN} {num_props} {timeout} --seed 2222")
                print(command)
                out_ = os.system(command)
                assert(out_ == 0)
    elif exp_name == 'GNN_train_eps0005':
        for steps_ in steps_list:
            for count_ in [10]:
                gpu_id = 2
                cpus = "6-8"
                pdprops = "--pdprops train_SAT_bugfixed_intermediate.pkl"
                name = f"--table_name {exp_name}"
                steps = f"--pgd_iters {steps_}"
                count = f"--count_particles {count_}"
                GNN = "--SAT_GNN_name 20210118_GNN_with_grad_sign/model-best.pkl --GNN_grad_feat --adv_method GNN --change_eps_const -0.005"
                lp = "--run_lp"

                command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                           f"--nn_name {nn} {pdprops} {steps} {restarts} {lp} "
                           f"{name} {restarts} {count} {GNN} {num_props} {timeout} --seed 2222")
                print(command)
                out_ = os.system(command)
                assert(out_ == 0)


def run_jade_experiments(exp_name):
    pdprops = "--pdprops jodie-base_easy.pkl"
    nn = "cifar_base_kw"
    restarts = '--random_restarts 10000'
    timeout = "--timeout 300"
    num_props = '--num_props 200'

    steps_list = [20]
    count_list = [20]
    data_list = ['deep_SAT_jade_inter.pkl', 'easy_SAT_jade_inter.pkl', 'train_SAT_jade_inter.pkl', 'val_SAT_jade.pkl', 'wide_SAT_jade_inter.pkl']
    data_list = ['train_SAT_jade_inter.pkl', 'val_SAT_jade.pkl', 'easy_SAT_jade_inter.pkl', 'deep_SAT_jade_inter.pkl', 'wide_SAT_jade_inter.pkl']
    data_list = ['wide_SAT_jade_inter.pkl']
    steps_ = 20

    if exp_name == 'GNN_jade':
        for d_i in data_list:
            for count_ in [10]:
                if 'deep' in d_i:
                    nn = "cifar_deep_kw"
                elif 'wide' in d_i:
                    nn = "cifar_wide_kw"
                gpu_id = 1
                cpus = "3-5"
                pdprops = f"--pdprops jade/{d_i}"
                name = f"--table_name jade_{d_i}_GNN"
                steps = f"--pgd_iters {steps_}"
                count = f"--count_particles {count_}"
                GNN = "--SAT_GNN_name 20210118_GNN_with_grad_sign/model-best.pkl --GNN_grad_feat --adv_method GNN"
                lp = "--run_lp"

                command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                           f"--nn_name {nn} {pdprops} {steps} {restarts} {lp} "
                           f"{name} {restarts} {count} {GNN} {num_props} {timeout} --seed 2222")
                print(command)
                out_ = os.system(command)
                assert(out_ == 0)
    if exp_name == 'pgd_jade':
        steps_ = 1000
        for d_i in data_list:
            if 'deep' in d_i:
                nn = "cifar_deep_kw"
            elif 'wide' in d_i:
                nn = "cifar_wide_kw"
            for count_ in [1000]:
                for lr_ in [1e-2]:  # do 1e-1 later
                    gpu_id = 0
                    cpus = "0-2"
                    pdprops = f"--pdprops jade/{d_i}"
                    name = f"--table_name jade_{d_i}_pgd"
                    steps = f"--pgd_iters {steps_}"
                    count = f"--count_particles {count_}"
                    lr = f"--pgd_optimizer_lr {lr_}"
                    lp = "--run_lp"

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} {lp} "
                               f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} ")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)


def deep_experiments(exp_name):
    restarts = '--random_restarts 10000'
    timeout = "--timeout 100"
    num_props = '--num_props 100'
    nn = "cifar_deep_kw"
    stepsize_list = [(1e-2, 1e-3)]
    data_list = [('deep', 'deep_SAT_jade.pkl')]
    count_list = [40]
    steps_list = [40]

    if 'train_on_deep' in exp_name:
        # exp is something like deep_exp_train_on_deep
        gnn_list = [('train_on_deep', 'train_on_deep/model-best.pkl'),]
        steps_list = [20, 40, 100]
        count_list = [20, 40, 100]
        gnn_type_list = [("", ""), ("_gnn_optimized", " --GNN_optimized ")]
        gpu_id = 0
        cpus = "2-4"
    if 'weight_decay' in exp_name:
        # exp is something like deep_exp_weight_decay
        gnn_list = [('weight_decay_1e_1', 'train_new_train_table_wd1_1_fixed/model-5.pkl')]
        steps_list = [40, 80]
        count_list = [40, 100]
        count_list = [40]
        stepsize_list = [(1e-2, 1e-4), (1e-2, 1e-5)]
        gpu_id = 1
        cpus = "3-5"
        gnn_type_list = [("", ""), ("_gnn_optimized", " --GNN_optimized ")]
        gnn_type_list = [("_gnn_optimized", " --GNN_optimized ")]
    if 'finetuning' in exp_name:
        if 'part1' in exp_name:
            gnn_list = [('finetuning_lr2_quick_decay', 'fintuning_deep_large_model_lr2/model-best.pkl')]
            gnn_list = [('finetuning_n100', 'fintuning_deep_large_model_fine_dataset100/model-best.pkl')]
            gnn_list = [('finetuning_1minute', 'fintuning_deep_large_model_finetuning_1_minute/model-best.pkl'),
                        # ('finetuning_1minute_lr01', 'fintuning_deep_large_model_finetuning_1_minute_lr_01/model-best.pkl')
                       ]
            gpu_id = 0
            cpus = "0-2"
            gnn_type_list = [("_gnn_optimized", " --GNN_optimized ")]
        if 'part2' in exp_name:
            gnn_list = [('finetuning_lr2_slow_decay', 'fintuning_deep_large_model_lr2_slow_decay/model-best.pkl')]
            gnn_list = [('finetuning_n50', 'fintuning_deep_large_model_fine_dataset/model-best.pkl')]
            gnn_list = [('finetuning_5minutes', 'fintuning_deep_large_model_finetuning_5_minute/model-best.pkl'),
                        ('finetuning_5minutes_dc', 'fintuning_deep_large_model_finetuning_5_minute_dc0/model-best.pkl')]
            gpu_id = 1
            cpus = "3-5"
            gnn_type_list = [("_gnn_optimized", " --GNN_optimized ")]
        if 'part3' in exp_name:
            gnn_list = [('finetuning_lr3', 'fintuning_deep_large_model_lr3/model-best.pkl')]
            gnn_list = [('finetuning_15minutes', 'fintuning_deep_large_model_finetuning_15_minute/model-best.pkl'),
                        ('finetuning_15minutes_dc', 'fintuning_deep_large_model_finetuning_15_minute_dc0/model-best.pkl')]
            gpu_id = 2
            cpus = "6-8"
            gnn_type_list = [("_gnn_optimized", " --GNN_optimized ")]

    for name_data, d_i in data_list:
        for name_gnn, gnn in gnn_list:
            for steps_ in steps_list:
              for lr_init, lr_fin in stepsize_list:
                for count_ in count_list:
                  for name_gnn2, gnn_type in gnn_type_list:
                    pdprops = f"--pdprops jade/{d_i}"
                    name = f"--table_name deep_experiments/{name_data}_{name_gnn}_steps_{steps_}_count_{count_}{name_gnn2}_lr_{lr_init}_{lr_fin}_momentum.pkl"
                    steps = f"--GNN_iters {steps_}"
                    count = f"--count_particles {count_}"
                    GNN = f"--SAT_GNN_name {gnn} --GNN_grad_feat --adv_method GNN {gnn_type}"
                    lp = "--run_lp"
                    lr = f"--GNN_lr_init {lr_init} --GNN_lr_fin {lr_fin} "

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {steps} {restarts} {lp} "
                               f"{name} {restarts} {count} {GNN} {num_props} {timeout} --seed 2222")
       	            print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)


def debug_GNN(exp_name):
    pdprops = "--pdprops jodie-base_easy.pkl"
    nn = "cifar_base_kw"
    restarts = '--random_restarts 10000'
    timeout = "--timeout 100"
    num_props = '--num_props 100'

    steps_list = [40]
    count_list = [40]
    data_list = [('deep', 'deep_SAT_jade_inter.pkl'), ('val', 'val_SAT_jade.pkl')]
    data_list = [('deep', 'deep_SAT_jade.pkl')]
    data_list = [('wide', 'wide_SAT_jade.pkl')]
    # data_list = [('base_easy', 'base_easy_SAT_jade.pkl')]
    steps_ = 40
    count_ = 40
    eps_change_list = [0]
    gnn_list = ['20210201_train_deep_long/model-100.pkl', '20210201_train_deep_long/model-best.pkl', '20210201_train_deep_long/model-100.pkl', '20210201_train_deep/model-best.pkl']
    gnn_list = ['20210201_weight_decay_01', '20210201_weight_decay_001', '20210201_weight_decay_0001']
    gnn_list = [('20210202_n4e4_wd_001', '20210202_n4e4_weight_decay_001/model-20.pkl'),
                ]
    gnn_list = [#('train_on_deep', 'train_on_deep/model-best.pkl'),
                # ('GNN_without_norm', 'new_GNN_without_norm/model-best.pkl'),
                ('weight_decay_1e_1', 'train_new_train_table_wd1_1_fixed/model-5.pkl'),
                # ('weight_decay_1e_2', 'train_new_train_table_wd1_2_fixed/model-5.pkl'),
                # ('big_jade_model', 'jade/train_jade_new_train_n25e4/model-best.pkl') 
               ]

    if exp_name == 'debug_GNN_deep':
        for name_data, d_i in data_list:
            for eps_change in eps_change_list:
                for name_gnn, gnn in gnn_list:
                    if 'deep' in d_i:
                        nn = "cifar_deep_kw"
                    elif 'wide' in d_i:
                        nn = "cifar_wide_kw"
                    gpu_id = 1
                    cpus = "3-5"
                    pdprops = f"--pdprops jade/{d_i}"
                    name = f"--table_name compare_GNNs_{name_data}_{name_gnn}_steps_{steps_}_count_{count_}_"
                    steps = f"--pgd_iters {steps_}"
                    steps = f"--GNN_iters {steps_}"
                    count = f"--count_particles {count_}"
                    GNN = f"--SAT_GNN_name {gnn} --GNN_grad_feat --adv_method GNN"
                    lp = "--run_lp"

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {steps} {restarts} {lp} "
                               f"{name} {restarts} {count} {GNN} {num_props} {timeout} --seed 2222")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)

    elif 'debug_momentum_adam' in exp_name:
        steps_ = 40
        count_ = 40
        gnn_list = [('new_gnn', 'train_new_train_table_wd1_2_fixed/model-best.pkl')]
        stepsize_list = [(1e-2, 1e-3)]
        decay_list = [('', '')]
        adam_list = [('', ''),
                     # ('adam', ' --GNN_adam ')
                    ]
        momentum_list = [0, 0.001, 0.01, 0.1, 0.5, 0.9]
        num_props = '--num_props 150'
        pdprops = "--pdprops jade/val_SAT_jade.pkl"
        timeout = "--timeout 100"  # 300 for eps - 0.005
        if 'part0' in exp_name:
            momentum_list = momentum_list[0:2]
            gpu_id = 0
            cpus = "0-2"
        if 'part1' in exp_name:
            momentum_list = momentum_list[2:4]
            gpu_id = 1
            cpus = "3-5"
        if 'part2' in exp_name:
            momentum_list = momentum_list[4:6]
            gpu_id = 2
            cpus = "6-8"
        for mom in momentum_list:
            for adam_name, adam_ in adam_list:
                for lr_init, lr_fin in stepsize_list:
                    for dc_name, decay in decay_list:
                        for _, gnn in gnn_list:
                            name = f"--table_name momentum/eps005_mom_{mom}_{adam_name}_nolpinit.pkl"
                            steps = f"--GNN_iters {steps_}"
                            count = f"--count_particles {count_}"
                            GNN = f"--SAT_GNN_name {gnn} --GNN_grad_feat --adv_method GNN {decay} "
                            lr = f"--GNN_lr_init {lr_init} --GNN_lr_fin {lr_fin} "
                            lp = "--run_lp "
                            params = f' --GNN_momentum {mom} {adam_}'
                            # params += ' --change_eps_const -0.005 '
                            command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                                       f"--nn_name {nn} {pdprops} {steps} {restarts} {lp} {params}"
                                       f"{name} {restarts} {count} {GNN} {num_props} {timeout} --seed 2222")
                            print(command)
                            out_ = os.system(command)
                            assert(out_ == 0)


def val_exp(exp_name):
    pdprops = "--pdprops jade/val_SAT_jade.pkl"
    nn = "cifar_base_kw"
    restarts = '--random_restarts 10000'
    timeout = "--timeout 100"
    num_props = '--num_props 150'

    if 'hparam_val' in exp_name:
        steps_list = [20, 40]
        count_list = [10, 20, 40]
        gnn_list = [('standard_gnn', '20210118_GNN_with_grad_sign/model-best.pkl'), ]
        gnn_list = [('new_gnn', 'train_new_train_table_wd1_2_fixed/model-best.pkl')]
        stepsize_list = [(1e-2, 1e-3), (1e-1, 1e-3), (1e-2, 1e-4)]
        decay_list = [('', ''), ('exp_decay', ' --GNN_rel_decay ')]

        if 'part1' in exp_name:
            count_list = [10]
            gpu_id = 0
            cpus = "0-2"
        if 'part2' in exp_name:
            count_list = [20]
            gpu_id = 1
            cpus = "3-5"
        if 'part3' in exp_name:
            count_list = [40]
            gpu_id = 2
            cpus = "6-8"
        for steps_ in steps_list:
            for count_ in count_list:
              for lr_init, lr_fin in stepsize_list:
                for dc_name, decay in decay_list:
                  for _, gnn in gnn_list:
                    name = f"--table_name hparam/steps_{steps_}_count_{count_}_lr_{lr_init}_{lr_fin}_{dc_name}"
                    steps = f"--GNN_iters {steps_}"
                    count = f"--count_particles {count_}"
                    GNN = f"--SAT_GNN_name {gnn} --GNN_grad_feat --adv_method GNN {decay} "
                    lr = f"--GNN_lr_init {lr_init} --GNN_lr_fin {lr_fin} "
                    lp = "--run_lp"

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {steps} {restarts} {lp} "
                               f"{name} {restarts} {count} {GNN} {num_props} {timeout} --seed 2222")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)


def run_baseline_experiments(exp_name):
    name = restarts = lr = steps = lp = ""

    pdprops = "--pdprops jade/val_SAT_jade.pkl"
    nn = "cifar_base_kw"
    restarts = '--random_restarts 1000'
    timeout = '--timeout 100'
    num_props = '--num_props 20'
    adv_method_list = ['mi_fgsm_attack']
    # adv_method_list = ['pgd_attack', 'mi_fgsm_attack']
    print("run baselines")
    if 'check_mi_fgsm' in exp_name:
        for steps_ in [1000]:
            for count_ in [1000]:
                for lr_ in [5.0, 1.0, 0.5]:  # do 1e-1 later
                 for mu in [0.0, 0.5, 1.0]:
                  for adv_method in adv_method_list:
                    gpu_id = 0
                    cpus = "0-2"
                    name = f"--table_name mi_fgsm_hparam/{adv_method}_steps_{steps_}_ctd_{count_}_lr_{lr_}_mu_{mu}.pkl"
                    steps = f"--pgd_iters {steps_}"
                    count = f"--count_particles {count_}"
                    lr = f"--pgd_optimizer_lr {lr_}"
                    mom = f" --pgd_momentum {mu} "
                    method = f" --adv_method {adv_method}"
                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} {lp} --check_adv 1 "
                               f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method} {mom}")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)
    if 'check_pgd' in exp_name:
        adv_method_list = ['pgd_attack']
        for steps_ in [1000]:
            for count_ in [1000]:
                for lr_ in [1e-2]:  # do 1e-1 later
                  for adv_method in adv_method_list:
                    gpu_id = 1
                    cpus = "3-5"
                    name = f"--table_name mi_fgsm_hparam/{adv_method}_steps_{steps_}_ctd_{count_}_lr_{lr_}.pkl"
                    steps = f"--pgd_iters {steps_}"
                    count = f"--count_particles {count_}"
                    lr = f"--pgd_optimizer_lr {lr_}"
                    method = f" --adv_method {adv_method}"
                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} {lp} --check_adv 1 "
                               f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method}")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)
    if 'deep_mi_fgsm' in exp_name:
        pdprops = "--pdprops jade/deep_SAT_jade.pkl"
        nn = "cifar_deep_kw"
        num_props = '--num_props 100'
        for steps_ in [1000]:
            for count_ in [1000]:
                for lr_ in [5.0]:  # do 1e-1 later
                 for mu in [0.5]:
                  for adv_method in adv_method_list:
                    gpu_id = 2
                    cpus = "6-8"
                    name = f"--table_name deep_experiments/{adv_method}_steps_{steps_}_ctd_{count_}_lr_{lr_}_mu_{mu}.pkl"
                    steps = f"--pgd_iters {steps_}"
                    count = f"--count_particles {count_}"
                    lr = f"--pgd_optimizer_lr {lr_}"
                    mom = f" --pgd_momentum {mu} "
                    method = f" --adv_method {adv_method}"
                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} {lp} --check_adv 1 "
                               f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method} {mom}")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)

def compare_all_methods(exp_name):
    restarts = '--random_restarts 1000'
    timeout = '--timeout 100'
    num_props = '--num_props 50'
    data_list = [('easy', 'base_easy_SAT_jade.pkl'), ('val', 'val_SAT_jade.pkl'),  ('wide', 'wide_SAT_jade.pkl'), ('deep', 'deep_SAT_jade.pkl'), ('easy', 'base_easy_SAT_jade.pkl')]
    data_list = [('val', 'val_SAT_jade.pkl')]
    nn = "cifar_base_kw"
    count_list = [1,32]

    if 'deep_mi_fgsm' in exp_name:
        adv_method_list = ['mi_fgsm_attack']
        for name_data, d_i in data_list:
          for steps_ in [50, 100, 250]:
            for count_ in count_list:
                for lr_ in [10.0, 5.0, 2.5]:  # do 1e-1 later
                 for mu in [0.25, 0.5, 1.0]:
                  for adv_method in adv_method_list:
                    if 'deep' in d_i:
                        nn = "cifar_deep_kw"
                    elif 'wide' in d_i:
                        nn = "cifar_wide_kw"
                    gpu_id = 0
                    cpus = "0-2"
                    pdprops = f"--pdprops jade/{d_i}"
                    name = f"--table_name compare_all_methods/hparams_{adv_method}_{name_data}_steps_{steps_}_ctd_{count_}_lr_{lr_}_mu_{mu}.pkl"
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

    if 'GNN_momentum' in exp_name:
        gnn_list = {'val': ('big_jade_model', 'jade/train_jade_new_train_n25e4/model-best.pkl'),
                    'easy': ('big_jade_model', 'jade/train_jade_new_train_n25e4/model-best.pkl'),
                    'wide': ('big_jade_model', 'jade/train_jade_new_train_n25e4/model-best.pkl'),
                    'deep': ('finetuning_15minutes_dc', 'fintuning_deep_large_model_finetuning_15_minute_dc0/model-best.pkl'),
                   }
        gpu_id = 2
        cpus = "6-8"
        gnn_type_list = [("_gnn_optimized", " --GNN_optimized ")]
        stepsize_list = [(1e-2, 1e-3)]
        steps_list = [40]

        for name_data, d_i in data_list:
            name_gnn, gnn = gnn_list[name_data]
            for steps_ in steps_list:
              for lr_init, lr_fin in stepsize_list:
                for count_ in count_list:
                  for name_gnn2, gnn_type in gnn_type_list:
                    if 'deep' in d_i:
                        nn = "cifar_deep_kw"
                    elif 'wide' in d_i:
                        nn = "cifar_wide_kw"
                    pdprops = f"--pdprops jade/{d_i}"
                    name = f"--table_name compare_all_methods/{name_data}_{name_gnn}_steps_{steps_}_count_{count_}{name_gnn2}_alpha_5_mu_05_momentum.pkl"
                    steps = f"--GNN_iters {steps_}"
                    count = f"--count_particles {count_}"
                    GNN = f"--SAT_GNN_name {gnn} --GNN_grad_feat --adv_method GNN {gnn_type}"
                    lp = "--run_lp"
                    lr = f"--GNN_lr_init {lr_init} --GNN_lr_fin {lr_fin} "

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {steps} {restarts} {lp} "
                               f"{name} {restarts} {count} {GNN} {num_props} {timeout} --seed 2222")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)

    if 'old_GNN' in exp_name:
        gnn_list = {'val': ('big_jade_model', 'jade/train_jade_new_train_n25e4/model-best.pkl'),
                    'easy': ('big_jade_model', 'jade/train_jade_new_train_n25e4/model-best.pkl'),
                    'wide': ('big_jade_model', 'jade/train_jade_new_train_n25e4/model-best.pkl'),
                    'deep': ('finetuning_15minutes_dc', 'fintuning_deep_large_model_finetuning_15_minute_dc0/model-best.pkl'),
                   }
        gpu_id = 1
        cpus = "3-5"
        gnn_type_list = [("_gnn_optimized", " --GNN_optimized ")]
        stepsize_list = [(1e-2, 1e-3)]
        steps_list = [40]

        for name_data, d_i in data_list:
            name_gnn, gnn = gnn_list[name_data]
            for steps_ in steps_list:
              for lr_init, lr_fin in stepsize_list:
                for count_ in count_list:
                  for name_gnn2, gnn_type in gnn_type_list:
                    if 'deep' in d_i:
                        nn = "cifar_deep_kw"
                    elif 'wide' in d_i:
                        nn = "cifar_wide_kw"
                    pdprops = f"--pdprops jade/{d_i}"
                    name = f"--table_name compare_all_methods/{name_data}_{name_gnn}_steps_{steps_}_count_{count_}{name_gnn2}_lr_{lr_init}_{lr_fin}.pkl"
                    steps = f"--GNN_iters {steps_}"
                    count = f"--count_particles {count_}"
                    GNN = f"--SAT_GNN_name {gnn} --GNN_grad_feat --adv_method GNN {gnn_type}"
                    lp = "--run_lp"
                    lr = f"--GNN_lr_init {lr_init} --GNN_lr_fin {lr_fin} "

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {steps} {restarts} {lp} --old_GNN "
                               f"{name} {restarts} {count} {GNN} {num_props} {timeout} --seed 2222")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)


def debug_apgd(exp_name):
    restarts = '--random_restarts 10000'
    timeout = '--timeout 100'
    data_list = [('val', 'val_SAT_jade.pkl')]
    nn = "cifar_base_kw"
    count_ = 1
    adv_method = 'a_pgd_attack'
    # adv_method = 'pgd_attack'

    if 'easy' in exp_name:
        data_list = [('val_easy', 'easy_val_SAT_jade.pkl')]
    else:
        data_list = [('val', 'val_SAT_jade.pkl')]

    num_props = '--num_props 50'
    gpu_id = 1
    cpus = "3-5"
    step_list = [100]
    lr_list = [1e-1]
    round_ = 'round1'

    for name_data, d_i in data_list:
        for steps_ in step_list:
            for lr_ in lr_list:
                    pdprops = f"--pdprops jade/{d_i}"
                    name = f"--table_name debug_apgd/{name_data}_apgd_steps_{steps_}_lr_{lr_}.pkl"
                    steps = f"--pgd_iters {steps_}"
                    count = f"--count_particles {count_}"
                    lr = f"--pgd_optimizer_lr {lr_}"
                    method = f" --adv_method {adv_method} "

                    command = (f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/run_attack.py "
                               f"--nn_name {nn} {pdprops} {lr} {steps} {restarts} --check_adv 1 "
                               f"{name} {restarts} {count} {timeout} --seed 2222 {num_props} {method}")
                    print(command)
                    out_ = os.system(command)
                    assert(out_ == 0)


def main():
    # run_old_exp('train_hard_attack_3_2')
    # run_many_pgd_experiments('lp_exp')
    # run_many_pgd_experiments('no_lp_exp')
    # run_pgd('old_train_dataset')
    # run_GNN('GNN_train_atlas')
    # run_GNN('GNN_train_eps0005')
    # run_many_pgd_experiments('pgd_train_eps0005')
    # run_many_pgd_experiments('pgd_check_every_iter')
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='experiment name')
    args = parser.parse_args()

    if 'jade' in args.exp_name:
        run_jade_experiments(args.exp_name)
    # elif 'debug' in args.exp_name:
    #     debug_GNN(args.exp_name)
    elif 'hparam' in args.exp_name:
        val_exp(args.exp_name)
    elif 'deep_exp' in args.exp_name:
        deep_experiments(args.exp_name)
    elif 'baselines' in args.exp_name:
        run_baseline_experiments(args.exp_name)
    elif 'compare_all_methods' in args.exp_name:
        compare_all_methods(args.exp_name)
    elif 'debug_apgd':
        debug_apgd(args.exp_name)
    else:
        input("TODO")


if __name__ == "__main__":
    main()
