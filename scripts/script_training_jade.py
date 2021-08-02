import os
import sys
import argparse

########################################################################
#   run training experiments
#   TODO:
########################################################################


# def run_train_exp(gpu_id, train_dir, val_dir='', max_train='', max_prop, horizon, epoch, opt='--opt adam', lr, save, load, gamma='--gamma 0.9', name, lr_decay, dup='', no_data=False):
def run_train_exp(train_dir, max_prop, horizon, epoch, lr, save, name, lr_decay, load='', val_dir='', max_train='', gamma='--gamma 0.9', opt='--opt adam', dup='', no_data=True):
    if horizon == 100:
        batch_size = "--batch_size 5"
    elif 'deep' in train_dir:
        batch_size = "--batch_size 10"
    else:
        batch_size = "--batch_size 30"

    if no_data:
        command = (f"""python adv_exp/GNN_training/train_no_dataset.py """
                   f"""{train_dir} {val_dir} {max_train} {max_prop} {horizon} {epoch} {opt} {batch_size} {lr} {save} """
                   f"""{load} --logger --seed 2222 {gamma} {name} {lr_decay} --pick_data_rdm {dup}""")
    else:
        command = (f"""CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python adv_exp/GNN_training/train_GNN.py """
                   f"""{train_dir} {val_dir} {max_train} {max_prop} {horizon} {epoch} {opt} {batch_size} {lr} {save} """
                   f"""{load} --logger --visdom --seed 2222 {gamma} {name} {lr_decay} --pick_data_rdm {dup}""")
    print(command)
    os.system(command)


def run_SAT_training_no_data(exp_name):
    if exp_name == 'train_table_eps03_jade':
        max_prop = '--max_num_prop 4'
        horizon = '--horizon 20'
        epoch = '--epoch 20'
        opt = '--opt adam'
        lr = '--lr 0.1 --weight_decay 0.2'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 2'
        name += ' --feature_lp_primal'
        # name += ' --lp_init '
        lr_decay = '--lr_decay 10 20 30 35'
        gpu_id = 5
        train_dir = '--pdprops train_eps03.pkl'
        val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_jade_table_eps03_n4e3':
        max_prop = '--max_num_prop 400'
        horizon = '--horizon 20'
        epoch = '--epoch 50'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal'
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 5
        train_dir = '--pdprops train_eps03.pkl'
        val_dir = max_train = ''
        input("wait")
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_jade_table_eps03_n4e4':
        max_prop = '--max_num_prop 4000'
        horizon = '--horizon 20'
        epoch = '--epoch 20'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 2'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal'
        # name += ' --lp_init '
        lr_decay = '--lr_decay 5 10 15'
        gpu_id = 5
        train_dir = '--pdprops train_eps03.pkl'
        val_dir = max_train = ''
        input("wait")
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_jade_table_eps02_n4e3':
        max_prop = '--max_num_prop 400'
        horizon = '--horizon 20'
        epoch = '--epoch 50'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal'
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 5
        train_dir = '--pdprops train_eps02.pkl'
        val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_jade_table_eps025_n4e3':
        max_prop = '--max_num_prop 400'
        horizon = '--horizon 20'
        epoch = '--epoch 50'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal'
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 5
        train_dir = '--pdprops train_eps025.pkl'
        val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_jade_new_train_n25e4':
        max_prop = '--max_num_prop 2500'
        horizon = '--horizon 20'
        epoch = '--epoch 50'
        opt = '--opt adam'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 5'
        load = ''  # '--load_model same'  # '--load_model default_name/model-best.pkl'
        gamma = '--gamma 0.9'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal'
        # name += ' --lp_init '
        lr_decay = '--lr_decay 20 30 35'
        gpu_id = 5
        train_dir = '--pdprops large_quick_train.pkl'
        val_dir = max_train = ''
        run_train_exp(gpu_id, train_dir, val_dir, max_train, max_prop, horizon, epoch, opt, lr, save, load, gamma, name, lr_decay, no_data=True)
    if exp_name == 'train_jade_n25e4_horizon40':
        max_prop = '--max_num_prop 2500'
        horizon = '--horizon 40'
        epoch = '--epoch 50'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 5'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal'
        lr_decay = '--lr_decay 20 30 35'
        train_dir = '--pdprops large_quick_train.pkl'
        run_train_exp(train_dir, max_prop, horizon, epoch, lr, save, name, lr_decay)
    if exp_name == 'train_jade_n25e4_momentum_01':
        max_prop = '--max_num_prop 2500'
        horizon = '--horizon 40'
        epoch = '--epoch 50'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 5'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal'
        name += ' --GNN_momentum 0.1 '
        lr_decay = '--lr_decay 20 30 35'
        train_dir = '--pdprops large_quick_train.pkl'
        run_train_exp(train_dir, max_prop, horizon, epoch, lr, save, name, lr_decay)
    if exp_name == 'train_jade_n25e4_adam':
        max_prop = '--max_num_prop 2500'
        horizon = '--horizon 40'
        epoch = '--epoch 50'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 5'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal'
        name += ' --GNN_adam '
        lr_decay = '--lr_decay 20 30 35'
        train_dir = '--pdprops large_quick_train.pkl'
        run_train_exp(train_dir, max_prop, horizon, epoch, lr, save, name, lr_decay)
    if exp_name == 'train_jade_n25e4_rel_decay':
        max_prop = '--max_num_prop 2500'
        horizon = '--horizon 40'
        epoch = '--epoch 50'
        lr = '--lr 0.01 --weight_decay 0.001'
        save = '--save_model 5'
        name = f'--exp_name {exp_name} --feature_grad --val_size 20'
        name += ' --feature_lp_primal'
        name += ' --GNN_rel_decay '
        lr_decay = '--lr_decay 20 30 35'
        train_dir = '--pdprops large_quick_train.pkl'
        run_train_exp(train_dir, max_prop, horizon, epoch, lr, save, name, lr_decay)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='experiment name')
    args = parser.parse_args()

    run_SAT_training_no_data(args.exp_name)


if __name__ == "__main__":
    main()
