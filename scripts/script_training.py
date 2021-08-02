import os
import sys
import argparse


########################################################################
#   run training experiments
#   TODO:
########################################################################


def run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, dup='', no_data=False):
    cpus = f'{3*gpu_id}-{3*gpu_id + 2}'
    if horizon == 100:
        batch_size = "--batch_size 5"
    elif 'deep' in train_dir:
        batch_size = "--batch_size 15"
    else:
        batch_size = "--batch_size 30"
    if horizon == '--horizon 40':
         batch_size = "--batch_size 15"

    if no_data:
        command = (f"""CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/GNN_training/train_no_dataset.py """
                   f"""{train_dir} {val_dir} {max_train} {max_prop} {horizon} {epoch} {opt} {batch_size} {lr} {save} """
                   f"""{load} --logger --visdom --seed 2222 {gamma} {name} {lr_decay} --pick_data_rdm {dup}""")
    else:
        command = (f"""CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/GNN_training/train_GNN.py """
                   f"""{train_dir} {val_dir} {max_train} {max_prop} {horizon} {epoch} {opt} {batch_size} {lr} {save} """
                   f"""{load} --logger --visdom --seed 2222 {gamma} {name} {lr_decay} --pick_data_rdm {dup}""")
    print(command)
    os.system(command)


def run_SAT_training(exp_type):
    if exp_type == 'debug_val':
        train_dir = '--train_sub_dir 2021_01_14_hard/train/'
        val_dir = '--val_sub_dir 2021_01_14_hard/val/'
        max_train = '--max_train_size 10'
        max_prop = '--max_num_prop 2'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--lr 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        lr_decay = ''
        lr_decay = '--lr_decay 50 100 150'

        gpu_id = 2
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == 'debug_val2':
        train_dir = '--train_sub_dir 2021_01_12_eps0025/train/'
        val_dir = '--val_sub_dir 2021_01_12_eps0025/val/'
        max_train = '--max_train_size 20'
        max_prop = '--max_num_prop 2'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--lr 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        lr_decay = ''
        lr_decay = '--lr_decay 50 100 150'

        gpu_id = 3
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == 'n1000_3decay':
        train_dir = '--train_sub_dir 2021_01_08'
        max_train = '--max_train_size 1000'
        max_prop = '--max_num_prop 100'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--lr 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        lr_decay = ''
        lr_decay = '--lr_decay 50 100 150'

        gpu_id = 3
        run_train_exp(gpu_id, train_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == 'n1000_3decay_GNNlr_decay':
        train_dir = '--train_sub_dir 2021_01_08'
        max_train = '--max_train_size 1000'
        max_prop = '--max_num_prop 100'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--GNN_lr_init 0.01 --GNN_lr_fin 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        # lr_decay = ''
        lr_decay = '--lr_decay 50 100 150'

        gpu_id = 2
        run_train_exp(gpu_id, train_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == 'extended_baseline2':
        train_dir = '--train_sub_dir 2021_01_08'
        max_train = '--max_train_size 1000'
        max_prop = '--max_num_prop 100'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--GNN_lr_init 0.01 --GNN_lr_fin 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        # lr_decay = ''
        lr_decay = '--lr_decay 100 150 180'

        gpu_id = 1
        run_train_exp(gpu_id, train_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)

    if exp_type == 'step_init_1e-3_fin_1e-3':
        train_dir = '--train_sub_dir 2021_01_08'
        max_train = '--max_train_size 1000'
        max_prop = '--max_num_prop 100'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--GNN_lr_init 0.001 --GNN_lr_fin 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        # lr_decay = ''
        lr_decay = '--lr_decay 100 150 180'

        gpu_id = 1
        run_train_exp(gpu_id, train_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == 'step_init_1e-2_fin_1e-4':
        train_dir = '--train_sub_dir 2021_01_08'
        max_train = '--max_train_size 1000'
        max_prop = '--max_num_prop 100'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--GNN_lr_init 0.01 --GNN_lr_fin 0.0001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        # lr_decay = ''
        lr_decay = '--lr_decay 100 150 180'

        gpu_id = 3
        run_train_exp(gpu_id, train_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == 'dataset_eps_minus_0.05_n1e4':
        train_dir = '--train_sub_dir 2021_01_12_eps005'
        max_train = '--max_train_size 2000'
        max_prop = '--max_num_prop 100'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--GNN_lr_init 0.01 --GNN_lr_fin 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        # lr_decay = ''
        lr_decay = '--lr_decay 100 150 180'

        gpu_id = 0
        run_train_exp(gpu_id, train_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == 'dataset_eps_minus_0.025_n1e4':
        train_dir = '--train_sub_dir 2021_01_12_eps0025'
        max_train = '--max_train_size 2000'
        max_prop = '--max_num_prop 100'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--GNN_lr_init 0.01 --GNN_lr_fin 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        # lr_decay = ''
        lr_decay = '--lr_decay 100 150 180'

        gpu_id = 1
        run_train_exp(gpu_id, train_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == '20210113_eps_0.0025_duplicate':
        train_dir = '--train_sub_dir 2021_01_12_eps0025/train/'
        val_dir = '--val_sub_dir 2021_01_12_eps0025/val/'
        max_train = '--max_train_size 4500'
        max_prop = '--max_num_prop 300'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--lr 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        lr_decay = ''
        lr_decay = '--lr_decay 100 150 180'
        dup = '--duplicate'
        gpu_id = 2
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, dup=dup)
    if exp_type == '20210113_eps_0.005_duplicate':
        train_dir = '--train_sub_dir 2021_01_12_eps005/train/'
        val_dir = '--val_sub_dir 2021_01_12_eps005/val/'
        max_train = '--max_train_size 4500'
        max_prop = '--max_num_prop 300'
        horizon = '--horizon 20'
        epoch = '--epoch 150'
        opt = '--opt adam'
        lr = '--lr 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        lr_decay = ''
        lr_decay = '--lr_decay 75 120 140'

        gpu_id = 1
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == '20210113_eps_0.0025_iter100_lr_2_3':
        train_dir = '--train_sub_dir 2021_01_12_eps0025/train/'
        val_dir = '--val_sub_dir 2021_01_12_eps0025/val/'
        max_train = '--max_train_size 4500'
        max_prop = '--max_num_prop 300'
        horizon = '--horizon 100'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--lr 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        lr_decay = ''
        lr_decay = '--lr_decay 100 150 180'

        gpu_id = 0
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == '20210113_eps_0.0025_iter100_lr_2_4':
        train_dir = '--train_sub_dir 2021_01_12_eps0025/train/'
        val_dir = '--val_sub_dir 2021_01_12_eps0025/val/'
        max_train = '--max_train_size 4500'
        max_prop = '--max_num_prop 300'
        horizon = '--horizon 100'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--GNN_lr_init 0.01 --GNN_lr_fin 0.0001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        lr_decay = ''
        lr_decay = '--lr_decay 100 150 180'

        gpu_id = 1
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == '20210115_train_hard':
        train_dir = '--train_sub_dir 2021_01_14_hard/train/'
        val_dir = '--val_sub_dir 2021_01_14_hard/val/'
        max_train = '--max_train_size 4000'
        max_prop = '--max_num_prop 200'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--lr 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type}'
        lr_decay = '--lr_decay 100 150 180'
        gpu_id = 2
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == '20210125_GNN_with_grad_sign_isadv_fixed':
        train_dir = '--train_sub_dir 2021_01_14_hard/train/'
        val_dir = '--val_sub_dir 2021_01_14_hard/val/'
        max_train = '--max_train_size 4000'
        max_prop = '--max_num_prop 200'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--lr 0.01'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type} --feature_grad'
        lr_decay = '--lr_decay 50 110 160 185'
        gpu_id = 3
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == '20210201_weight_decay_0001':
        train_dir = '--train_sub_dir 2021_01_14_hard/train/'
        val_dir = '--val_sub_dir 2021_01_14_hard/train/'
        max_train = '--max_train_size 4000'
        max_prop = '--max_num_prop 200'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type} --feature_grad'
        lr_decay = '--lr_decay 50 110 160 185'
        gpu_id = 1
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == '20210201_weight_decay_001':
        train_dir = '--train_sub_dir 2021_01_14_hard/train/'
        val_dir = '--val_sub_dir 2021_01_14_hard/train/'
        max_train = '--max_train_size 4000'
        max_prop = '--max_num_prop 200'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type} --feature_grad'
        lr_decay = '--lr_decay 50 110 160 185'
        gpu_id = 0
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == '20210202_n4e4_weight_decay_001':
        train_dir = '--train_sub_dir bab_dataset/'
        val_dir = '--val_sub_dir 2021_02_01_deep/'
        max_train = '--max_train_size 40000'
        max_prop = '--max_num_prop 200'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type} --feature_grad'
        lr_decay = '--lr_decay 50 110 160 185'
        gpu_id = 1
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)

def generalization_exp(exp_type):
    if exp_type == '20210201_val_deep':
        train_dir = '--train_sub_dir 2021_01_14_hard/'
        val_dir = '--val_sub_dir 2021_02_01_deep/'
        max_train = '--max_train_size 200'
        max_prop = '--max_num_prop 10'
        horizon = '--horizon 20'
        epoch = '--epoch 20'
        opt = '--opt adam'
        lr = '--lr 0.01'
        save = '--save_model 20'
        load = '--load_model 20210118_GNN_with_grad_sign/model-best.pkl'  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type} --feature_grad '
        lr_decay = '--lr_decay 50 110 160 185'
        gpu_id = 1
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == '20210201_train_deep':
        train_dir = '--train_sub_dir 2021_02_01_deep/'
        # val_dir = '--val_sub_dir 2021_01_14_hard/'
        val_dir = '--val_sub_dir 2021_02_01_deep/'
        max_train = '--max_train_size 200'
        max_prop = '--max_num_prop 10'
        horizon = '--horizon 20'
        epoch = '--epoch 20'
        opt = '--opt adam'
        lr = '--lr 0.01'
        save = '--save_model 20'
        load = '--load_model 20210118_GNN_with_grad_sign/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type} --feature_grad '
        lr_decay = '--lr_decay 10 15 18'
        gpu_id = 1
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)
    if exp_type == '20210201_train_deep_long':
        train_dir = '--train_sub_dir 2021_02_01_deep/'
        # val_dir = '--val_sub_dir 2021_01_14_hard/'
        val_dir = '--val_sub_dir 2021_02_01_deep/'
        max_train = '--max_train_size 300'
        max_prop = '--max_num_prop 15'
        horizon = '--horizon 20'
        epoch = '--epoch 200'
        opt = '--opt adam'
        lr = '--lr 0.01'
        save = '--save_model 20'
        load = '--load_model 20210118_GNN_with_grad_sign/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type} --feature_grad '
        lr_decay = '--lr_decay 50 110 160 185'
        gpu_id = 1
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)

    if exp_type == '20210201_val_wide':
        train_dir = '--train_sub_dir 2021_01_14_hard/'
        val_dir = '--val_sub_dir 2021_02_01_wide/'
        max_train = '--max_train_size 200'
        max_prop = '--max_num_prop 10'
        horizon = '--horizon 20'
        epoch = '--epoch 20'
        opt = '--opt adam'
        lr = '--lr 0.01'
        save = '--save_model 20'
        load = '--load_model 20210118_GNN_with_grad_sign/model-best.pkl'  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_type} --feature_grad '
        lr_decay = '--lr_decay 50 110 160 185'
        gpu_id = 1
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay)


def run_SAT_training_no_data(exp_name):
    if exp_name == 'debug':
        train_dir = '--train_sub_dir bab_dataset/'
        val_dir = '--val_sub_dir 2021_02_01_deep/'
        max_train = '--max_train_size 200'
        max_prop = '--max_num_prop 100'
        horizon = '--horizon 20'
        epoch = '--epoch 20'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 20'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name new_training_debug9 --feature_grad --val_size 10'
        name += ' --feature_lp_primal '
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 0
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_no_dataset':
        max_prop = '--max_num_prop 400'
        horizon = '--horizon 20'
        epoch = '--epoch 50'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name train_no_dataset --feature_grad --val_size 10'
        name += ' --feature_lp_primal '
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 1
        train_dir = val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_new_train_table':
        max_prop = '--max_num_prop 400'
        horizon = '--horizon 20'
        epoch = '--epoch 50'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal '
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 0
        train_dir = '--pdprops train_SAT_jade_inter.pkl'
        val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_new_train_table_wd1_1_fixed':
        max_prop = '--max_num_prop 400'
        horizon = '--horizon 20'
        epoch = '--epoch 50'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.1'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal '
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 0
        train_dir = '--pdprops train_SAT_jade_inter.pkl'
        val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_new_train_table_wd1_2_fixed':
        max_prop = '--max_num_prop 400'
        horizon = '--horizon 20'
        epoch = '--epoch 50'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal '
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 1
        train_dir = '--pdprops train_SAT_jade_inter.pkl'
        val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_new_train_table_eps03':
        max_prop = '--max_num_prop 4'
        horizon = '--horizon 20'
        epoch = '--epoch 50'
        opt = '--opt adam'
        lr = '--lr 0.1 --weight_decay 0.2'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 2'
        name += ' --feature_lp_primal --cpu'
        # name += ' --lp_init '
        lr_decay = '--lr_decay 10 20 30 35'
        gpu_id = 5
        train_dir = '--pdprops train_eps03.pkl'
        # train_dir = '--pdprops train_SAT_jade_inter.pkl'
        val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_on_deep':
        max_prop = '--max_num_prop 200'
        horizon = '--horizon 20'
        epoch = '--epoch 50'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 2'
        name += ' --feature_lp_primal '
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 1
        train_dir = '--pdprops train_eps03.pkl'
        train_dir = '--pdprops deep_SAT_jade.pkl --nn_name cifar_deep_kw'
        # train_dir = '--pdprops train_SAT_jade_inter.pkl'
        val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'new_GNN_with_norm':
        max_prop = '--max_num_prop 400'
        horizon = '--horizon 20'
        epoch = '--epoch 20'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal '
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 1
        train_dir = '--pdprops train_SAT_jade_inter.pkl'
        val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'new_GNN_without_norm':
        max_prop = '--max_num_prop 400'
        horizon = '--horizon 20'
        epoch = '--epoch 20'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal '
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 2
        train_dir = '--pdprops train_SAT_jade_inter.pkl'
        val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)


def finetuning_GNN(exp_name):
    # parameters common for all fintuning experiments
    gamma = '--gamma 0.9'
    opt = '--opt adam'
    horizon = '--horizon 20'
    name = ' --feature_lp_primal --feature_grad '
    val_dir = max_train = ''

    if 'deep' in exp_name:
        train_dir = '--pdprops deep_SAT_jade.pkl --nn_name cifar_deep_kw ' 

    if 'large_model_lr3' in exp_name:
        max_prop = '--max_num_prop 10'
        epoch = '--epoch 20'
        lr = '--lr 0.001 --weight_decay 0.01'
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 15 18'
        gpu_id = 1
    elif 'large_model_lr2_quick_decay' in exp_name:
        max_prop = '--max_num_prop 10'
        epoch = '--epoch 20'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 5 15 18'
        gpu_id = 0
    elif 'large_model_lr2_slow_decay' in exp_name:
        max_prop = '--max_num_prop 10'
        epoch = '--epoch 20'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 14 16 18'
        gpu_id = 2
    elif 'large_model_more_props' in exp_name:
        max_prop = '--max_num_prop 30'
        epoch = '--epoch 6'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 4 5 6'
        gpu_id = 1
    elif 'large_model_fine_dataset50' in exp_name:
        max_prop = '--max_num_prop 50'
        epoch = '--epoch 5'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 3 4'
        gpu_id = 0
        train_dir = '--pdprops ../../cifar_results/adv_results/deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
    elif 'large_model_fine_dataset100' in exp_name:
        max_prop = '--max_num_prop 100'
        epoch = '--epoch 3'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 2 3'
        gpu_id = 1
        train_dir = '--pdprops ../../cifar_results/adv_results/deep_finetuning_025.pkl --nn_name cifar_deep_kw '
    elif 'finetuning_1_minute' in exp_name:
        max_prop = '--max_num_prop 45'
        epoch = '--epoch 1'
        lr = '--lr 0.1 --weight_decay 0.001'
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = ''  # '--lr_decay 2 3'
        gpu_id = 0
        train_dir = '--pdprops ../../cifar_results/adv_results/deep_finetuning_025.pkl --nn_name cifar_deep_kw '
    elif 'finetuning_5_minute' in exp_name:
        max_prop = '--max_num_prop 110'
        epoch = '--epoch 2'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 0'
        gpu_id = 1
        train_dir = '--pdprops ../../cifar_results/adv_results/deep_finetuning_025.pkl --nn_name cifar_deep_kw '
    elif 'finetuning_15_minute' in exp_name:
        max_prop = '--max_num_prop 300'
        epoch = '--epoch 2'
        lr = '--lr 0.01 --weight_decay 0.01'
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 0'
        gpu_id = 2
        train_dir = '--pdprops ../../cifar_results/adv_results/deep_finetuning_025.pkl --nn_name cifar_deep_kw '
    elif 'finetuning_wide_15_minutes' in exp_name:
        max_prop = '--max_num_prop 300'
        epoch = '--epoch 2'
        lr = '--lr 0.01 '
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 0'
        gpu_id = 1
        train_dir = '--pdprops ../../cifar_results/adv_results/wide_finetuning_025.pkl --nn_name cifar_wide_kw '
    elif 'finetuning_wide_15_mins_hor40' in exp_name:
        max_prop = '--max_num_prop 200'
        epoch = '--epoch 2'
        lr = '--lr 0.01 '
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 0'
        gpu_id = 1
        horizon = '--horizon 40'
        train_dir = '--pdprops ../../cifar_results/adv_results/wide_finetuning_025.pkl --nn_name cifar_wide_kw '
    elif 'finetuning_wide_15_mins_eps02' in exp_name:
        max_prop = '--max_num_prop 300'
        epoch = '--epoch 2'
        lr = '--lr 0.01 '
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 0'
        gpu_id = 2
        train_dir = '--pdprops ../../cifar_results/adv_results/wide_finetuning_02.pkl --nn_name cifar_wide_kw '
    elif 'finetuning_wide_15_mins_rel_decay' in exp_name:
        max_prop = '--max_num_prop 300'
        epoch = '--epoch 2'
        lr = '--lr 0.01 '
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 0'
        gpu_id = 2
        train_dir = '--pdprops ../../cifar_results/adv_results/wide_finetuning_02.pkl --nn_name cifar_wide_kw '
    elif 'finetuning_wide_15_mins_new_GNN' in exp_name:
        max_prop = '--max_num_prop 200'
        epoch = '--epoch 2'
        lr = '--lr 0.001 '
        save = '--save_model 2'
        load = '--load_model jade/train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 1'
        gpu_id = 2
        horizon = '--horizon 40'
        train_dir = '--pdprops ../../cifar_results/adv_results/wide_finetuning_025.pkl --nn_name cifar_wide_kw '
    elif 'finetuning_wide_15m_new_GNN_lr2' in exp_name:
        max_prop = '--max_num_prop 200'
        epoch = '--epoch 2'
        lr = '--lr 0.01 '
        save = '--save_model 2'
        load = '--load_model jade/train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        name += f'--exp_name {exp_name} --val_size 20'
        lr_decay = '--lr_decay 0'
        gpu_id = 2
        horizon = '--horizon 40'
        train_dir = '--pdprops ../../cifar_results/adv_results/wide_finetuning_025.pkl --nn_name cifar_wide_kw '
    elif 'finetuning_15mins_adam_lr2_GNN_lr3' in exp_name:
        max_prop = '--max_num_prop 300'
        epoch = '--epoch 2'
        lr = '--lr 0.01'
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        load = '--load_model jade/train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.001 --GNN_lr_fin 0.001 '
        lr_decay = '--lr_decay 0'
        gpu_id = 3
        train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw '
    elif 'finetuning_15mins_adam_lr3_nodc_GNN_lr3' in exp_name:
        max_prop = '--max_num_prop 300'
        epoch = '--epoch 2'
        lr = '--lr 0.001'
        save = '--save_model 2'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        load = '--load_model jade/train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.001 --GNN_lr_fin 0.001 '
        lr_decay = '--lr_decay 2'
        gpu_id = 2
        train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw '
    elif 'finetuning_15mins_30gnn_adam_lr3_nodc_GNN_lr2' in exp_name:
        max_prop = '--max_num_prop 300'
        epoch = '--epoch 2'
        lr = '--lr 0.001'
        save = '--save_model 1'
        load = '--load_model jade/train_jade_new_train_n25e4/model-best.pkl --reset_optimizer True '  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        load = '--load_model jade/train_jade_n25e4_horizon40/model-30.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.01 --GNN_lr_fin 0.01 '
        lr_decay = '--lr_decay 2'
        gpu_id = 1
        train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw '
    elif 'finetuning_deep_4' in exp_name:
        max_prop = '--max_num_prop 200'
        train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
        if 'gnn30' in exp_name:
            load = '--load_model jade/train_jade_n25e4_horizon40/model-30.pkl --reset_optimizer True '
            gnn_name = 'gnn30'
        elif 'wd_gnn' in exp_name:
            load = '--load_model train_new_train_table_wd1_1_fixed/model-5.pkl --reset_optimizer True '
            gnn_name = 'wd_gnn'
        else:
            load = '--load_model jade/train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
            gnn_name = 'gnnbest'
        if 'part0' in exp_name:
            gpu_id = 0
            lr = '--lr 0.001'
            lr_decay = '--lr_decay 2'
            exp_name = f'finetuningdeep_{gnn_name}_lr_3'
            print(gpu_id)
        elif 'part1' in exp_name:
            gpu_id = 1
            lr = '--lr 0.001'
            lr_decay = '--lr_decay 0'
            exp_name = f'finetuningdeep_{gnn_name}_lr_3_dc'
        elif 'part2' in exp_name:
            gpu_id = 2
            lr = '--lr 0.01'
            lr_decay = '--lr_decay 5'
            exp_name = f'finetuningdeep_{gnn_name}_lr_2'
        elif 'part3' in exp_name:
            gpu_id = 3
            lr = '--lr 0.01'
            lr_decay = '--lr_decay 0'
            exp_name = f'finetuningdeep_{gnn_name}_lr_2_dc'
        elif 'part5' in exp_name:
            gpu_id = 0
            lr = '--lr 0.001'
            lr_decay = '--lr_decay 2'
            horizon = '--horizon 40'
            max_prop = '--max_num_prop 100'
            exp_name = f'finetuningdeep_{gnn_name}_lr_3_hor40'
        elif 'part6' in exp_name:
            gpu_id = 1
            lr = '--lr 0.01'
            lr_decay = '--lr_decay 2'
            horizon = '--horizon 40'
            max_prop = '--max_num_prop 100'
            exp_name = f'finetuningdeep_{gnn_name}_lr_2_hor40'
        elif 'part7' in exp_name:
            gpu_id = 2
            lr = '--lr 0.01'
            lr_decay = '--lr_decay 5'
            exp_name = f'finetuningdeep_avgeps_lr2'
            train_dir = '--pdprops /deep_finetuning_avg_eps.pkl --nn_name cifar_deep_kw '
        elif 'part8' in exp_name:
            gpu_id = 3
            lr = '--lr 0.001'
            lr_decay = '--lr_decay 5'
            exp_name = f'finetuningdeep_avgeps_lr2'
            train_dir = '--pdprops /deep_finetuning_avg_eps.pkl --nn_name cifar_deep_kw '
        else:
            input("Failed")

        epoch = '--epoch 2'
        # train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
        save = '--save_model 2'
    run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
 # --weight_decay 0.01


def finetuning_madry(exp_name):
    # parameters common for all fintuning experiments
    gamma = '--gamma 0.9'
    opt = '--opt adam'
    horizon = '--horizon 20'
    name = ' --feature_lp_primal --feature_grad '
    val_dir = max_train = ''

    train_dir = '--pdprops deep_SAT_jade.pkl --nn_name cifar_madry '

    if 'rebuttal_constant_eps' in exp_name:
        max_prop = '--max_num_prop 300'
        lr = '--lr 0.001'
        save = '--save_model 1'
        load = '--load_model train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.01 --GNN_lr_fin 0.01 '
        lr_decay = '--lr_decay 8 15'
        gpu_id = 1
        train_dir = '--pdprops /madry_finetuning_02.pkl --nn_name cifar_madry '
        name += f' --exp_name {exp_name} --val_size 20 --GNN_lr_init 0.01 --GNN_lr_fin 0.01  --feature_grad --feature_lp_primal '
        epoch = '--epoch 20'
        # train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
        save = '--save_model 2'

    elif 'rebuttal_easy_train1' in exp_name:
        max_prop = '--max_num_prop 100'
        lr = '--lr 0.001'
        save = '--save_model 1'
        load = '--load_model train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.01 --GNN_lr_fin 0.01 '
        lr_decay = '--lr_decay 8 15'
        gpu_id = 1
        train_dir = '--pdprops /madry_easy_SAT_jade.pkl  --nn_name cifar_madry '
        name += f' --exp_name {exp_name} --val_size 20 --GNN_lr_init 0.01 --GNN_lr_fin 0.01  --feature_grad --feature_lp_primal '
        epoch = '--epoch 20'
        # train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
        save = '--save_model 2'

    elif 'rebuttal_easy_train2' in exp_name:
        max_prop = '--max_num_prop 200'
        lr = '--lr 0.001'
        save = '--save_model 1'
        load = '--load_model train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.01 --GNN_lr_fin 0.01 '
        lr_decay = '--lr_decay 15 24'
        gpu_id = 1
        train_dir = '--pdprops /madry_easy_SAT_jade.pkl  --nn_name cifar_madry '
        name += f' --exp_name {exp_name} --val_size 20 --GNN_lr_init 0.01 --GNN_lr_fin 0.01  --feature_grad --feature_lp_primal '
        epoch = '--epoch 30'
        # train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
        save = '--save_model 2'

    elif 'rebuttal_debug' in exp_name:
        max_prop = '--max_num_prop 2'
        lr = '--lr 0.001'
        save = '--save_model 1'
        load = '--load_model train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.01 --GNN_lr_fin 0.01 '
        lr_decay = '--lr_decay 8 15'
        gpu_id = 2
        train_dir = '--pdprops /madry_easy_SAT_jade.pkl  --nn_name cifar_madry '
        name += f' --exp_name {exp_name} --val_size 20 --GNN_lr_init 0.01 --GNN_lr_fin 0.01  --feature_grad --feature_lp_primal '
        epoch = '--epoch 20'
        # train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
        save = '--save_model 2'

    elif 'rebuttal_train_easy' in exp_name:
        max_prop = '--max_num_prop 300'
        lr = '--lr 0.001'
        save = '--save_model 1'
        load = '--load_model train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.01 --GNN_lr_fin 0.01 '
        lr_decay = '--lr_decay 15 24'
        gpu_id = 1
        train_dir = '--pdprops ../../cifar_results/adv_results/madry_easy_SAT_training.pkl  --nn_name cifar_madry '
        name += f' --exp_name {exp_name} --val_size 20 --GNN_lr_init 0.01 --GNN_lr_fin 0.01  --feature_grad --feature_lp_primal '
        epoch = '--epoch 30'
        # train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
        save = '--save_model 2'

    elif 'rebuttal_train_med_hor20' in exp_name:
        max_prop = '--max_num_prop 300'
        lr = '--lr 0.001'
        save = '--save_model 1'
        load = '--load_model train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.01 --GNN_lr_fin 0.01 '
        lr_decay = '--lr_decay 15 24'
        lr_decay = '--lr_decay 24 27'
        lr = '--lr 0.01'
        lr_decay = '--lr_decay 5 22 26'
        gpu_id = 1
        train_dir = '--pdprops ../../cifar_results/adv_results/madry_med_SAT_training.pkl  --nn_name cifar_madry '
        name += f' --exp_name {exp_name} --val_size 20 --GNN_lr_init 0.01 --GNN_lr_fin 0.01  --feature_grad --feature_lp_primal '
        epoch = '--epoch 30'
        # train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
        save = '--save_model 2'

    elif 'rebuttal_train_med_hor40_lr2' in exp_name:
        max_prop = '--max_num_prop 300'
        lr = '--lr 0.01'
        save = '--save_model 1'
        load = '--load_model train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.01 --GNN_lr_fin 0.01 '
        horizon = '--horizon 40'
        lr = '--lr 0.01'
        lr_decay = '--lr_decay 10 15 18'
        gpu_id = 1
        train_dir = '--pdprops ../../cifar_results/adv_results/madry_med_SAT_training.pkl  --nn_name cifar_madry '
        name += f' --exp_name {exp_name} --val_size 20 --GNN_lr_init 0.01 --GNN_lr_fin 0.01  --feature_grad --feature_lp_primal '
        epoch = '--epoch 20'
        # train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
        save = '--save_model 2'

    run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)


def finetuning_mnist(exp_name):
    # parameters common for all fintuning experiments
    gamma = '--gamma 0.9'
    opt = '--opt adam'
    horizon = '--horizon 20'
    name = ' --feature_lp_primal --feature_grad '
    val_dir = max_train = ''

    train_dir = '--pdprops deep_SAT_jade.pkl --nn_name cifar_madry '

    if 'rebuttal_train_mnist1' in exp_name:
        max_prop = '--max_num_prop 300'
        save = '--save_model 1'
        load = '--load_model train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.01 --GNN_lr_fin 0.01 '
        lr_decay = '--lr_decay 15 24'
        lr_decay = '--lr_decay 24 27'
        lr = '--lr 0.001'
        lr_decay = '--lr_decay 5 22 26'
        gpu_id = 2
        train_dir = '--pdprops ../../cifar_results/adv_results/mnist_base_easy_SAT.pkl --nn_name wide '
        name += f' --exp_name {exp_name} --val_size 20 --GNN_lr_init 0.01 --GNN_lr_fin 0.01  --feature_grad --feature_lp_primal '
        epoch = '--epoch 30'
        # train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
        save = '--save_model 2'
    elif 'rebuttal_train_mnist_hor10' in exp_name:
        max_prop = '--max_num_prop 500'
        save = '--save_model 1'
        load = '--load_model train_jade_n25e4_horizon40/model-best.pkl --reset_optimizer True '
        name += f'--exp_name {exp_name} --val_size 20'
        name += ' --GNN_lr_init 0.01 --GNN_lr_fin 0.01 '
        lr_decay = '--lr_decay 15 24'
        lr_decay = '--lr_decay 24 27'
        lr = '--lr 0.001'
        horizon = '--horizon 10'
        lr_decay = '--lr_decay 40 46'
        gpu_id = 0
        train_dir = '--pdprops ../../cifar_results/adv_results/mnist_base_easy_SAT.pkl --nn_name wide '
        name += f' --exp_name {exp_name} --val_size 20 --GNN_lr_init 0.01 --GNN_lr_fin 0.01  --feature_grad --feature_lp_primal '
        epoch = '--epoch 50'
        # train_dir = '--pdprops /deep_finetuning_025.pkl --nn_name cifar_deep_kw ' 
        save = '--save_model 2'

    cpus = f'{3*gpu_id}-{3*gpu_id + 2}'
    if horizon == 100:
        batch_size = "--batch_size 5"
    elif 'deep' in train_dir:
        batch_size = "--batch_size 15"
    else:
        batch_size = "--batch_size 30"
    if horizon == '--horizon 40':
         batch_size = "--batch_size 15"

    command = (f"""CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/GNN_training/train_mnist.py """
               f"""{train_dir} {val_dir} {max_train} {max_prop} {horizon} {epoch} {opt} {batch_size} {lr} {save} """
               f"""{load} --logger --visdom --seed 2222 {gamma} {name} {lr_decay} --pick_data_rdm --dataset mnist """)
    print(command)
    os.system(command)


def main():
    # run_SAT_training(exp_type='debug_val2')
    # run_SAT_training(exp_type='dataset_eps_minus_0.025_n1e4')
    # run_SAT_training(exp_type='step_init_1e-2_fin_1e-4')
    # run_SAT_training(exp_type='step_init_1e-3_fin_1e-3')
    # run_SAT_training(exp_type='extended_baseline2')
    # run_SAT_training(exp_type='n1000_3decay')
    # run_SAT_training(exp_type='debug_light')
    # run_SAT_training(exp_type='n1000_3decay_GNNlr_decay')
    # run_SAT_training(exp_type='20210113_eps_0.0025_duplicate')
    # run_SAT_training(exp_type='20210113_eps_0.005_duplicate')
    # run_SAT_training(exp_type='20210113_eps_0.0025_iter100_lr_2_4')
    # run_SAT_training(exp_type='20210113_eps_0.0025_iter100_lr_2_3')
    # run_SAT_training(exp_type='20210115_train_hard')
    # run_SAT_training(exp_type='20210118_GNN_with_grad_sign')
    # run_SAT_training(exp_type='debug_val')
    # run_SAT_training(exp_type='20210125_GNN_with_grad_sign_isadv_fixed')
    # run_SAT_training(exp_type='20210201_weight_decay_0001')
    # run_SAT_training(exp_type='20210202_n4e4_weight_decay_001')

    # run_SAT_training_no_data('train_no_dataset')
    # run_SAT_training_no_data('train_new_train_table')
    # run_SAT_training_no_data('train_new_train_table_lr1-1')
    # run_SAT_training_no_data('train_new_train_table_wd1_2_fixed')
    # run_SAT_training_no_data('train_new_train_table_wd1_1_fixed')
    # run_SAT_training_no_data('new_GNN_with_norm')

    # generalization_exp('20210201_train_deep_long')
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='experiment name')
    args = parser.parse_args()

    if 'finetuning' in args.exp_name:
        finetuning_GNN(args.exp_name)
    elif 'mnist' in args.exp_name:
        finetuning_mnist(args.exp_name)
    elif 'rebuttal' in args.exp_name:
        finetuning_madry(args.exp_name)

if __name__ == "__main__":
    main()
