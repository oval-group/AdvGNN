import torch
import os
import torch as th
import pandas as pd
import pickle
import mlogger


def get_table(exp_type='easy_experiments', model='base'):
    path_ = './adv_exp/UAI_experiments/'
    datasets_name = os.listdir(f"{path_}/{exp_type}/{model}/")
    datasets = datasets_name
    datasets.sort()
    datasets_name.sort()

    # datasets = [d for d in datasets if '2222' in d]
    # datasets_name = [d for d in datasets_name if '2222' in d]

    tables = []
    for d_i in datasets:
        table_ = pd.read_pickle(f"{path_}/{exp_type}/{model}/{d_i}")
        tables.append(table_)
        # print(f"\n\n length, {len(table_.dropna(how='any'))}\n\n")

    create_table = True
    if create_table:
        _columns = ['Exp_name', 'number_props', 'Method', 'seed', 'Time(s)', 'Timeout(%)']
        graph_df = pd.DataFrame(columns=_columns)

    t_1_ = []
    time_list = []
    timeout_list = []
    for idx_, t_i in enumerate(tables):
        string_ = datasets_name[idx_] 

        t_i = t_i.dropna(how='any')

        timeout_ = 20
        # t_i.loc[table_['BTime_PGD'] > timeout_, 'BSAT'] = 'timeout'
        # t_i.loc[table_['BSAT'] == 'timeout', 'BTime_PGD'] = timeout_ + 1

        keys_time = [key_ for key_ in t_i.keys() if 'Time' in key_][0]
        keys_SAT = [key_ for key_ in t_i.keys() if 'SAT' in key_][0]
        t_i.loc[t_i[keys_time] > timeout_, keys_SAT] = 'timeout'
        t_i.loc[t_i[keys_SAT] == 'timeout', keys_time] = timeout_ + 1

        time_list.append(t_i[keys_time].tolist())
        timeout_list.append(t_i[keys_SAT].tolist())

        if 'GNN' in datasets_name[idx_]:
            method = 'AdvGNN'
        if 'mi_fgsm' in datasets_name[idx_]:
            method = 'MI-FGSM'
        if 'pgd' in datasets_name[idx_]:
            method = 'PGD Attack'
        if 'CW' in datasets_name[idx_]:
            method = 'C&W'

        try:
            seed = string_.split("seed",1)[1].split(".",1)[0]
        except Exception:
            seed=0

        len_ = len(t_i)
        avg_time = sum(time_list[-1])/len(time_list[-1])
        per_timeout = timeout_list[-1].count('timeout')*100/len(time_list[-1])
        row_ = pd.DataFrame([[datasets_name[idx_], len_, method, seed, avg_time, per_timeout]], columns=_columns)
        graph_df = graph_df.append(row_, ignore_index=True)
    # print(graph_df)
    return graph_df


def print_latex(table_latex):
    if 'Exp_name' in table_latex.keys():
        table_latex.drop(labels='Exp_name', axis=1, inplace=True)
    if 'number_props' in table_latex.keys():
        table_latex.drop(labels='number_props', axis=1, inplace=True)
    print(table_latex.to_latex(index=False))


def print_experiment(exp_type='easy_experiments', model='all'):
    if model == 'all':
     table_combined = get_table(exp_type, 'base')
     for model in ['wide', 'deep']:
        extra_tb = get_table(exp_type, model)
        assert(list(extra_tb.iloc[:, 2]) == list(table_combined.iloc[:, 2])), ((list(extra_tb.iloc[:, 2])), (list(table_combined.iloc[:, 2])))
        table_combined[f'Time(s)_{model}'] = extra_tb['Time(s)']
        table_combined[f'Timeout(%)_{model}'] = extra_tb['Timeout(%)']
    else:
        table_combined = get_table(exp_type, model)
    table_combined_seeds = table_combined.groupby(['Method'], as_index=False).mean()
    table_combined = table_combined.round(decimals=3)
    table_combined_seeds = table_combined_seeds.round(decimals=3)
    print_latex(table_combined)
    print_latex(table_combined_seeds)


def print_hparam_mi_fgsm(round=1):
    datasets_name = os.listdir(f'./cifar_results/adv_results/UAI/hparams_mi_fgsm/round{round}/')
    datasets = [f'UAI/hparams_mi_fgsm/round{round}/'+l for l in datasets_name]
    datasets.sort()
    datasets_name.sort()

    tables = []

    for d_i in datasets:
        path = './cifar_results/adv_results/'
        table_ = pd.read_pickle(path + d_i)
        tables.append(table_)

    create_table = True
    if create_table:
        _columns = ['Exp_name', 'Steps', 'lr', 'mu', 'number_props', 'average_time', 'percentage_timeout']
        graph_df = pd.DataFrame(columns=_columns)

    t_1_ = []
    time_list = []
    timeout_list = []
    for idx_, t_i in enumerate(tables):
        # print(f"\n {datasets_name[idx_]}")
        string_ = datasets_name[idx_] 
        steps = string_.split("steps_",1)[1].split("_",1)[0]
        lr = string_.split("lr_",1)[1].split("_",1)[0]
        mu = string_.split("mu_",1)[1].split(".pkl",1)[0]
        print(steps, lr, mu)

        t_i = t_i.dropna(how='any')
        t_i = t_i.head(50)

        timeout_ = 100
        table_.loc[table_['BTime_PGD'] > timeout_, 'BSAT'] = 'timeout'
        table_.loc[table_['BSAT'] == 'timeout', 'BTime_PGD'] = timeout_ + 1
        # print(len(t_i))

        keys_time = [key_ for key_ in t_i.keys() if 'BTime' in key_][0]
        keys_SAT = [key_ for key_ in t_i.keys() if 'SAT' in key_][0]

        time_list.append(t_i[keys_time].tolist())
        timeout_list.append(t_i[keys_SAT].tolist())

        # print('mean time', sum(time_list[-1])/len(time_list[-1]))
        # print('percentage timeout', timeout_list[-1].count('timeout')/len(time_list[-1]))
        len_ = len(t_i)
        avg_time = sum(time_list[-1])/len(time_list[-1])
        per_timeout = timeout_list[-1].count('timeout')/len(time_list[-1])
        row_ = pd.DataFrame([[datasets_name[idx_], steps, lr, mu, len_, avg_time, per_timeout]], columns=_columns)
        # row_ = pd.DataFrame([[datasets_name[idx_], len(t_i),  sum(time_list[-1])/len(time_list[-1]),  timeout_list[-1].count('timeout')/len(time_list[-1])]], columns=_columns)
        graph_df = graph_df.append(row_, ignore_index=True)
    table_sorted_time = graph_df.sort_values(by='average_time') 
    # print(table_sorted_time)
    table_sorted_timeout = graph_df.sort_values(by='percentage_timeout')
    # print(table_sorted_timeout)
    list_time = graph_df['average_time']
    graph_df['rank_time'] = graph_df['average_time'].rank()
    graph_df['rank_timeout'] = graph_df['percentage_timeout'].rank()
    graph_df['average_rank'] = (graph_df['rank_time'] + graph_df['rank_timeout'])/2
    table_sorted_combined = graph_df.sort_values(by='average_rank')
    print(table_sorted_combined)
    table_latex = table_sorted_combined
    table_latex.drop(labels='Exp_name', axis=1, inplace=True)
    table_latex.drop(labels='number_props', axis=1, inplace=True)
    print(table_latex.to_latex(index=False)) 


def print_hparam_pgd(round=1):
    datasets_name = os.listdir(f'./cifar_results/adv_results/UAI/hparams_pgd/round{round}/')
    datasets = [f'UAI/hparams_pgd/round{round}/'+l for l in datasets_name]
    datasets.sort()
    datasets_name.sort()

    tables = []

    for d_i in datasets:
        path = './cifar_results/adv_results/'
        table_ = pd.read_pickle(path + d_i)
        tables.append(table_)

    create_table = True
    if create_table:
        _columns = ['Exp_name', 'Steps', 'lr', 'number_props', 'average_time', 'percentage_timeout']
        graph_df = pd.DataFrame(columns=_columns)

    t_1_ = []
    time_list = []
    timeout_list = []
    for idx_, t_i in enumerate(tables):
        # print(f"\n {datasets_name[idx_]}")
        string_ = datasets_name[idx_]
        steps = string_.split("steps_",1)[1].split("_",1)[0]
        lr = string_.split("lr_",1)[1].split("_",1)[0]

        t_i = t_i.dropna(how='any')
        t_i = t_i.head(50)

        timeout_ = 100
        table_.loc[table_['BTime_PGD'] > timeout_, 'BSAT'] = 'timeout'
        table_.loc[table_['BSAT'] == 'timeout', 'BTime_PGD'] = timeout_ + 1
        # print(len(t_i))

        keys_time = [key_ for key_ in t_i.keys() if 'BTime' in key_][0]
        keys_SAT = [key_ for key_ in t_i.keys() if 'SAT' in key_][0]

        time_list.append(t_i[keys_time].tolist())
        timeout_list.append(t_i[keys_SAT].tolist())

        # print('mean time', sum(time_list[-1])/len(time_list[-1]))
        # print('percentage timeout', timeout_list[-1].count('timeout')/len(time_list[-1]))
        len_ = len(t_i)
        avg_time = sum(time_list[-1])/len(time_list[-1])
        per_timeout = timeout_list[-1].count('timeout')/len(time_list[-1])
        row_ = pd.DataFrame([[datasets_name[idx_], steps, lr, len_, avg_time, per_timeout]], columns=_columns)
        # row_ = pd.DataFrame([[datasets_name[idx_], len(t_i),  sum(time_list[-1])/len(time_list[-1]),  timeout_list[-1].count('timeout')/len(time_list[-1])]], columns=_columns)
        graph_df = graph_df.append(row_, ignore_index=True)
    table_sorted_time = graph_df.sort_values(by='average_time')
    # print(table_sorted_time)
    table_sorted_timeout = graph_df.sort_values(by='percentage_timeout')
    # print(table_sorted_timeout)
    list_time = graph_df['average_time']
    graph_df['rank_time'] = graph_df['average_time'].rank()
    graph_df['rank_timeout'] = graph_df['percentage_timeout'].rank()
    graph_df['average_rank'] = (graph_df['rank_time'] + graph_df['rank_timeout'])/2
    table_sorted_combined = graph_df.sort_values(by='average_rank')
    print(table_sorted_combined)
    table_latex = table_sorted_combined
    table_latex.drop(labels='Exp_name', axis=1, inplace=True)
    table_latex.drop(labels='number_props', axis=1, inplace=True)
    print(table_latex.to_latex(index=False))


def print_foolbox_hparam(round=3):
    if True:
        datasets_name = os.listdir(f'./cifar_results/adv_results/foolbox/hparam_CW/round{round}/')
        datasets = [f'foolbox/hparam_CW/round{round}/'+l for l in datasets_name]
    else:
        datasets_name = os.listdir(f'./cifar_results/adv_results/foolbox/CW_experiments/')
        datasets = [f'foolbox/CW_experiments/'+l for l in datasets_name]
    datasets.sort()
    datasets_name.sort()

    tables = []

    for d_i in datasets:
        path = './cifar_results/adv_results/'
        table_ = pd.read_pickle(path + d_i)
        tables.append(table_)

    create_table = True
    if create_table:
        _columns = ['Exp_name', 'Steps', 'lr', 'number_props', 'average_time', 'percentage_timeout']
        _columns = ['Exp_name', 'number_props', 'eps_difference']
        graph_df = pd.DataFrame(columns=_columns)

    eps_list = []
    for idx_, t_i in enumerate(tables):

        t_i = t_i.dropna(how='any')
        t_i = t_i.head(50)

        eps_list.append(t_i['eps_difference'])
        len_ = len(t_i)
        avg_eps = sum(eps_list[-1])/len_
        print(avg_eps, eps_list[-1].max())
        row_ = pd.DataFrame([[datasets_name[idx_], len_, avg_eps]], columns=_columns)

        graph_df = graph_df.append(row_, ignore_index=True)

    print(graph_df)
    print(list(graph_df['eps_difference']))

def print_foolbox_exp():
    datasets_name = os.listdir(f'./cifar_results/adv_results/foolbox/CW_experiments/')
    datasets = [f'foolbox/CW_experiments/'+l for l in datasets_name]

    datasets.sort()
    datasets_name.sort()

    print(datasets)
    tables = []

    for d_i in datasets:
        path = './cifar_results/adv_results/'
        table_ = pd.read_pickle(path + d_i)
        tables.append(table_)

    create_table = True
    if create_table:
        _columns = ['Exp_name', 'Steps', 'lr', 'number_props', 'average_time', 'percentage_timeout']
        _columns = ['Exp_name', 'number_props', 'eps_difference']
        graph_df = pd.DataFrame(columns=_columns)

    eps_list = []
    for idx_, t_i in enumerate(tables):

        t_i = t_i.dropna(how='any')

        eps_list.append(t_i['eps_difference'])
        len_ = len(t_i)
        avg_eps = sum(eps_list[-1])/len_
        print(avg_eps, eps_list[-1].max())
        row_ = pd.DataFrame([[datasets_name[idx_], len_, avg_eps]], columns=_columns)

        graph_df = graph_df.append(row_, ignore_index=True)

        keys_time = [key_ for key_ in t_i.keys() if 'BTime' in key_][0]
        keys_SAT = [key_ for key_ in t_i.keys() if 'SAT' in key_][0]

        timeout_ = 100
        time_list = []; timeout_list = []
        t_i.loc[t_i[keys_time] > timeout_, keys_SAT] = 'timeout'
        t_i.loc[t_i[keys_SAT] == 'timeout', keys_time] = timeout_ + 1

        time_list.append(t_i[keys_time].tolist())
        timeout_list.append(t_i[keys_SAT].tolist())

        # print('mean time', sum(time_list[-1])/len(time_list[-1]))
        # print('percentage timeout', timeout_list[-1].count('timeout')/len(time_list[-1]))
        len_ = len(t_i)
        avg_time = sum(time_list[-1])/len(time_list[-1])
        per_timeout = timeout_list[-1].count('timeout')/len(time_list[-1])
        print(len_)
        print(avg_time)
        print(per_timeout)


def split_dataset():
    for diff in ['easy', 'med']:
        path = f'cifar_results/adv_results/madry_{diff}_SAT_jade.pkl'
        table_ = pd.read_pickle(path)
        graph_first = table_[:200]
        graph_second = table_[200:]
        graph_first.to_pickle(f'cifar_results/adv_results/madry_{diff}_SAT_testing.pkl')
        graph_second.to_pickle(f'cifar_results/adv_results/madry_{diff}_SAT_training.pkl')


def analyse_rebuttal():
    path = './cifar_results/adv_results/UAI/rebuttal/'
    table = "GNN_lp_exp_naive_KW_GNN_seed2222.pkl"
    table = pd.read_pickle(path+table)
    print(table)
    table = table.dropna(how='any')

    timeout_ = 100
    t_i = table
    time_list = []
    timeout_list = []

    keys_time = [key_ for key_ in t_i.keys() if 'Time' in key_][0]
    keys_SAT = [key_ for key_ in t_i.keys() if 'SAT' in key_][0]
    t_i.loc[t_i[keys_time] > timeout_, keys_SAT] = 'timeout'
    t_i.loc[t_i[keys_SAT] == 'timeout', keys_time] = timeout_ + 1

    time_list.append(t_i[keys_time].tolist())
    timeout_list.append(t_i[keys_SAT].tolist())

    len_ = len(t_i)
    avg_time = sum(time_list[-1])/len(time_list[-1])
    per_timeout = timeout_list[-1].count('timeout')*100/len(time_list[-1])
    print(f"length {len_}, avg time {avg_time}, per timeout {per_timeout}")


def main():
    # split_dataset()
    # print_hparam_mi_fgsm(round=1)
    # print_hparam_mi_fgsm(round=2)
    # print_experiment()
    # print_experiment('experiments')
    # print_experiment('experiments', 'base')
    # print_experiment('experiments', 'wide')
    # print_experiment('experiments', 'deep')
    # print_hparam_pgd(round=1)
    # print_foolbox_exp()
    # print_foolbox_hparam(round=3)
    analyse_rebuttal()

if __name__ == "__main__":
    main()

