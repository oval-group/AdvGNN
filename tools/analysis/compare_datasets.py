import torch
import torch as th
import pandas as pd
import pickle
import mlogger


def compare_dataset():
    # compare different datasets
    datasets = ['base_easy_SAT_1212.pkl', 'base_easy_SAT_1412_2.pkl', 'base_easy_SAT_1412_1.pkl']

    tables = []

    for d_i in datasets:
        path = './cifar_results/adv_results/'
        table_ = pd.read_pickle(path + d_i)
        tables.append(table_)

    eps_list = []
    for t_i in tables:
        eps_list.append(t_i['Eps'])
        print(t_i.dropna(how='all'))
    input("wait")

    print((eps_list[0]-eps_list[1]).dropna().mean())
    print("if negative  default is better, if positive abs=0.3 is better")
    print((eps_list[0]-eps_list[2]).dropna().mean())
    print("if negative default is better, if positive restart is better")


def compare_exp():
    datasets = ['20210115_PGD_TrainSatHard.pkl', '20210115_train_hard_TrainSatHard_invertedpickout.pkl', '20210115_train_hard_TrainSatHard.pkl']
    datasets = ['20210115_train_hard_TrainSatHard.pkl', '20210115_train_hard_TrainSatHard_10_best_subdoms.pkl', '20210115_train_hard_TrainSatHard_10_subdoms.pkl']
    datasets = ['20210115_train_hard_TrainSatHard.pkl', '20210115_PGD_TrainSatHard.pkl', '20210119_GNN_grad_sign_train_hard.pkl', '20210119_GNN_grad_sign_pick10_train_hard.pkl']
    datasets = ['training_pgd_attack.pkl', 'training_pgd_attack_restart1000.pkl']
    datasets = ['training_pgd_iter_5e4_batch_1e3.pkl', 'training_pgd_iter_5e3_batch_1e2.pkl', 'firstSAT_step1e4_lr1e-3_count11.pkl']
    datasets = ['debug_lp.pkl', 'training_pgd_iter_5e3_batch_1e2.pkl']
    datasets = ['pgd_steps_1000_ctd_100_lr_0.001_lpinit.pkl', 'pgd_steps_1000_ctd_100_lr_0.001.pkl', 'pgd_steps_1000_ctd_100_lr_0.01_lpinit.pkl', 'pgd_steps_1000_ctd_100_lr_0.01.pkl']
    datasets = ['train_SAT_bugfixed.pkl', 'base_easy_SAT_bugfixed.pkl']
    datasets = ['pgd_steps_1000_ctd_500_lr_0.01_lpinit.pkl', 'pgd_steps_1000_ctd_500_lr_0.01.pkl']
    datasets = ['GNN_train_eps001', 'GNN_train_eps0005']
    datasets = ['GNN_train_eps0005', 'pgd_train_eps0005']
    # datasets = ['pgd_steps_1000_ctd_1000_lr_0.01_lpinit.pkl', 'pgd_steps_1000_ctd_1000_lr_0.01_lpinit_checkeveryiter.pkl']
    # datasets = ['GNN_train_atlas', 'GNN_train2', 'pgd_steps_1000_ctd_500_lr_0.01_lpinit.pkl', 'pgd_steps_1000_ctd_1000_lr_0.01_lpinit.pkl']
    print("1. old gNN, 2.pgd, 3. GNN grad, 4. GNN grad, pick10")
    tables = []

    for d_i in datasets:
        path = './cifar_results/adv_results/'
        table_ = pd.read_pickle(path + d_i)
        table_ = table_.dropna(how='any')[0:45]
        tables.append(table_)

    for t_i in tables:
        # print(t_i.dropna(how='any')[0:50])
        # print(t_i)
        time_ = th.FloatTensor(t_i['BTime_PGD'].tolist())
        time_.clamp_(max=100)
        print("mean time", time_.mean())
        print("\nlength", len(t_i['BTime_PGD'].tolist()))
        print("sum time", sum(t_i['BTime_PGD'].tolist()))
        print("average time", sum(t_i['BTime_PGD'].tolist())/len(t_i['BTime_PGD'].tolist()))
        print("sum restarrt", sum(t_i['restarts'].tolist()))
        print("num restarts==0", (t_i['restarts'].tolist()).count(0))
        print("count timeouts", (t_i['BSAT'].tolist()).count("timeout"))
        # print(t_i['BTime_PGD'].tolist())

    # dataset = 'train_manual_pgd.pkl'
    # table_ = pd.read_pickle(path + dataset)
    # print(table_)
    # table_ = pd.read_pickle('cifar_train_pdtables/train_props.pkl')
    # print(table_)


def print_fb():
    datasets = ['GNN_train_eps0005', 'pgd_train_eps0005']
    datasets = ['Additive_noise.pkl', 'BasicIterative.pkl', 'DeepFool.pkl',  'PGD.pkl',  'Repeated_noise.pkl']
    tables = []

    for d_i in datasets:
        path = './cifar_results/adv_results/foolbox/'
        table_ = pd.read_pickle(path + d_i)
        table_ = table_.dropna(how='any')[0:200]
        tables.append(table_)

    for t_i in tables:
        # print(t_i.dropna(how='any')[0:50])
        # print(t_i)
        time_ = th.FloatTensor(t_i['BTime'].tolist())
        time_.clamp_(max=100)
        print("mean time", time_.mean())
        print("\nlength", len(t_i['BTime'].tolist()))
        print("sum time", sum(t_i['BTime'].tolist()))
        print("average time", sum(t_i['BTime'].tolist())/len(t_i['BTime'].tolist()))
        print("sum restarrt", sum(t_i['restarts'].tolist()))
        print("num restarts==0", (t_i['restarts'].tolist()).count(0))
        print("count timeouts", (t_i['BSAT'].tolist()).count("timeout"))
        print("count solved", (t_i['BSAT'].tolist()).count(True))
        # print(t_i['BTime_PGD'].tolist())


def compare_SAT_performance():
    # compare different verification methods
    datasets = ['firstSAT_step1e1_lr1e-3_count11.pkl', 'firstSAT_step1e3_lr1e-3_count11.pkl', 'firstSAT_step1e4_lr1e-3_count11.pkl']
    datasets = ['2021_01_14_GNN_0.0025_eps025.pkl', '2021_01_14_PGD_eps025.pkl']
    datasets = ['2021_01_14_GNN_0.0025_eps025_t600.pkl', '2021_01_14_PGD_eps025_t600.pkl']
    datasets = ['compare_GNNs_deep_train_on_deep', 'compare_GNNs_deep_weight_decay_1e_1_steps_40_count_40', 'jade_deep_SAT_jade_inter.pkl_pgd']

    tables = []

    for d_i in datasets:
        path = './cifar_results/adv_results/'
        table_ = pd.read_pickle(path + d_i)
        tables.append(table_)

    t_1_ = []
    time_list = []
    timeout_list = []
    subdom_list = []
    t_3_ = []
    for t_i in tables:
        t_i = t_i.dropna(how='any')
        t_i = t_i.head(100)

        timeout_ = 100
        table_.loc[table_['BTime_PGD'] > timeout_, 'BSAT'] = 'timeout'
        table_.loc[table_['BSAT'] == 'timeout', 'BTime_PGD'] = timeout_ + 1
        print(len(t_i))

        keys_time = [key_ for key_ in t_i.keys() if 'BTime' in key_][0]
        keys_SAT = [key_ for key_ in t_i.keys() if 'SAT' in key_][0]

        time_list.append(t_i[keys_time].tolist())
        timeout_list.append(t_i[keys_SAT].tolist())

        print('mean time', sum(time_list[-1])/len(time_list[-1]))
        print('percentage timeout', timeout_list[-1].count('timeout')/len(time_list[-1]))


def compare_training_dataset():
    tables = []

    table1 = './cifar_results/adv_results/train_props_2112_steps_e5.pkl'
    table2 = './batch_verification_results/train_SAT.pkl'
    table2 = './cifar_results/adv_results/train_manual_pgd.pkl'
        # print("length", len(time_list[-1]))
    table1 = './cifar_results/adv_results/base_easy_SAT_bugfixed.pkl'
    table2 = './cifar_results/adv_results/train_SAT_bugfixed.pkl'
    table1 = './batch_verification_results/jade/val_SAT_jade.pkl'
    table2 = './cifar_results/adv_results/val_SAT_jade.pkl'
    table3 = './batch_verification_results/val_props.pkl'
    table1 = './batch_verification_results/jade/train_SAT_jade_inter.pkl' 
    table2 = './cifar_results/adv_results/train_SAT_jade.pkl'
	
    t1 = pd.read_pickle(table1)
    t2 = pd.read_pickle(table2)

    print(t1)
    print(t2)

    print(t1.dropna(how='all'))
    print(t2.dropna(how='all'))


def replot_exp():
  list_ = ['step_init_1e-1_fin_1e-3', 'extended_baseline2']
  for exp_name in list_:
    state_path = f'./adv_exp/GNNs/{exp_name}/state.json'
    new_xp = mlogger.load_container(state_path)

    new_visdom_plotter = mlogger.VisdomPlotter(visdom_opts={'env': exp_name, 'server': 'http://atlas.robots.ox.ac.uk',
                                               'port': 9016}, manual_update=True)
    new_xp.plot_on_visdom(new_visdom_plotter)
    new_visdom_plotter.update_plots()


def chance_pickle():
    import pickle
    path_ = f'./batch_verification_results/'
    file_ = f'train_SAT_bugfixed_intermediate2.pkl'
    record_name = path_+file_
    print(record_name)
    graph_df = pd.read_pickle(record_name).dropna(how="any")
    file_new = file_[:-4] + '_pkl4.pkl'
    with open(path_+file_new, 'wb') as pfile:
     print(path_+file_new)
     input("wait")
     pickle.dump(graph_df, pfile, protocol=4)  


def main():
    # compare_exp()
    # compare_training_dataset()
    # compare_dataset()
    compare_SAT_performance()
    # replot_exp()
    # chance_pickle()
    # print_fb()


if __name__ == "__main__":
    main()

