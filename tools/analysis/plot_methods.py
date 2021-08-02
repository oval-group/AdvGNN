import torch
import visdom
import pandas as pd
import argparse
import os


########################################################################
#   plotting experiments on visdom
#   TODO
#   - make sure that all timeout are set to timeout_time
########################################################################

def _get_dataset(exp_name):
    if exp_name == 'train_200':
        datasets = []
        for steps_ in [1000, 5000, 20000]:
            for count_ in [100, 500, 1000]:
                # for lr_ in [1e-1, 1e-2, 1e-3]:
                for lr_ in [1e-2]:
                    for lp_init in ['', '_lpinit']:
                        datasets.append(f'pgd_steps_{steps_}_ctd_{count_}_lr_{lr_}{lp_init}.pkl')
        datasets.append("pgd_steps_1000_ctd_1000_lr_0.01_lpinit_checkeveryiter.pkl")
        datasets.append("GNN_train2")
        datasets.append("GNN_train_atlas")
    if exp_name == 'train_0.005':
        datasets = []
        datasets = ['GNN_train_eps0005', 'pgd_train_eps0005']
    elif exp_name == '20210201_jade_easy':
        datasets = ['jade_easy_SAT_jade_inter.pkl_pgd', 'jade_easy_SAT_jade_inter.pkl_GNN',
                    'compare_GNNs_base_easy_weight_decay_1e_1_steps_40_count_40', 'compare_GNNs_base_easy_big_jade_model_steps_40_count_40']
    elif exp_name == '20210201_jade_deep':
        datasets = ['jade_deep_SAT_jade_inter.pkl_pgd', 'jade_deep_SAT_jade_inter.pkl_GNN', 'debug_GNN_deep_eps0', 'debug_GNN_deep_model-10', 'debug_GNN_deep_odel-bes']
    elif exp_name == 'generalize_to_deep':
        datasets = ['jade_deep_SAT_jade_inter.pkl_pgd', 'jade_deep_SAT_jade_inter.pkl_GNN', 'debug_GNN_deep_20210201_weight_decay_01',
                    'debug_GNN_deep_20210201_weight_decay_001', 'debug_GNN_deep_20210201_weight_decay_0001', 'debug_GNN_deep_20210202_n4e4_wd_001']
    elif exp_name == 'generalize_to_deep2':
        datasets = ['jade_deep_SAT_jade_inter.pkl_pgd', 'compare_GNNs_deep_train_on_deep', 'compare_GNNs_deep_weight_decay_1e_1',
                    # 'compare_GNNs_deep_GNN_without_norm',
                    'compare_GNNs_deep_weight_decay_1e_2', 'compare_GNNs_deep_weight_decay_1e_1_steps_20_count_40',
                    'compare_GNNs_deep_weight_decay_1e_1_steps_40_count_40']
    elif exp_name == '20210201_jade_wide':
        datasets = ['jade_wide_SAT_jade_inter.pkl_pgd', 'jade_wide_SAT_jade_inter.pkl_GNN',
                    'compare_GNNs_wide_weight_decay_1e_1_steps_20_count_40', 'compare_GNNs_wide_weight_decay_1e_1_steps_40_count_40_',
                    'compare_GNNs_wide_big_jade_model_steps_20_count_40', 'compare_GNNs_wide_big_jade_model_steps_40_count_40_']
    elif exp_name == 'attack_validation_momentum':
        datasets = os.listdir('./cifar_results/adv_results/momentum/')
        datasets = ['momentum/'+l for l in datasets]
        datasets.sort()
    elif exp_name =='big_val':
        datasets = os.listdir('./cifar_results/adv_results/hparam/')
        datasets = ['hparam/'+l for l in datasets]
        datasets.sort()
    elif exp_name == 'attack_validation_PGD_hparam_on_train':
        datasets = os.listdir('./cifar_results/adv_results/pgd_hparam/')
        datasets = ['pgd_hparam/'+l for l in datasets]
        datasets.sort()
    elif exp_name == 'attack_validation_deep':
        datasets = os.listdir('./cifar_results/adv_results/deep_experiments/')
        datasets = ['deep_experiments/'+l for l in datasets if 'old' not in l]
        datasets.sort()
    elif exp_name == 'attack_validation_mi_fgsm_on_val':
        datasets = os.listdir('./cifar_results/adv_results/mi_fgsm_hparam/')
        datasets = ['mi_fgsm_hparam/'+l for l in datasets]
        datasets.sort()
    elif exp_name == 'attack_validation_all_methods':
        datasets = os.listdir('./cifar_results/adv_results/compare_all_methods/')
        datasets = ['compare_all_methods/'+l for l in datasets]
        datasets.sort()
    elif exp_name == 'UAI_mi_fgsm_hparam_round1':
        datasets_name = os.listdir('./cifar_results/adv_results/UAI/hparams_mi_fgsm/')
        datasets = ['UAI/hparams_mi_fgsm/'+l for l in datasets_name]
        datasets.sort()
        datasets_name.sort()
    elif exp_name == 'UAI_mi_fgsm_hparam_round2':
        datasets_name = os.listdir('./cifar_results/adv_results/UAI/hparams_mi_fgsm/round2/')
        datasets = ['UAI/hparams_mi_fgsm/round2/'+l for l in datasets_name]
        datasets.sort()
        datasets_name.sort()
    elif exp_name == 'UAI_GNN_hparam':
        datasets_name = os.listdir('./cifar_results/adv_results/UAI/hparams_GNN/')
        datasets = ['UAI/hparams_GNN/'+l for l in datasets_name]
        datasets.sort()
        datasets_name.sort()
    elif exp_name == 'UAI_experiments':
        datasets_name = os.listdir('./cifar_results/adv_results/UAI/experiments/')
        datasets = ['UAI/experiments/'+l for l in datasets_name]
        datasets.sort()
        datasets_name.sort()

    return datasets, datasets_name


def plot_pgd_experiments(name_, args, gpu_=True):
    datasets, datasets_names = _get_dataset(name_)

    tables = []

    x_list = []
    y_list = []

    dataset_worked = []

    timeout_ = 100
    for d_i in datasets:
        try:
            path = './cifar_results/adv_results/'
            table_ = pd.read_pickle(path + d_i)
            table_ = table_.dropna(how='any')[0:190]

            # make sure the time for all timed out properties is set to timeout_
            table_.loc[table_['BSAT'] == 'timeout', 'BTime_PGD'] = timeout_ + 1

            m_timings = table_['BTime_PGD'].tolist()
            m_timings.sort()

            # Make it an actual cactus plot
            axis_min = 0
            y_min = 0
            x_axis_max = 100
            x_axis_max = timeout_

            xs = [axis_min]
            ys = [y_min]
            prev_y = 0
            for i, x in enumerate(m_timings):
                if x <= x_axis_max:
                    # Add the point just before the rise
                    xs.append(x)
                    ys.append(prev_y)
                    # Add the new point after the rise, if it's in the plot
                    xs.append(x)
                    new_y = 100*(i+1)/len(m_timings)
                    ys.append(new_y)
                    prev_y = new_y
            # Add a point at the end to make the graph go the end
            xs.append(x_axis_max)
            ys.append(prev_y)

            x_list.append(xs)
            y_list.append(ys)

            # print("xs", xs)
            # print("ys", ys)
            print(d_i)
            dataset_worked.append(d_i)
        except Exception:
            continue

    if args.sort:
     y_last = [l[-1] for l in y_list]
     x_list = [x for _,x in sorted(zip(y_last,x_list))]
     y_list = [y for _,y in sorted(zip(y_last,y_list))]
     datasets_names = [x for _,x in sorted(zip(y_last, datasets_names))]
     x_list.reverse()
     y_list.reverse()
     datasets_names.reverse()

    _visdom_plotting(x_list, y_list, datasets_names, name_)


def _visdom_plotting(xs, ys, names, exp_name):
    # dotted line, colour
    # visdom_opts = {'server': 'http://atlas.robots.ox.ac.uk',
    #                'port': 9016, 'env': 'PGD_experiments1e2_and_GNN_n200'}
    visdom_opts = {'server': 'http://themis.robots.ox.ac.uk',
                   'port': 9018, 'env': exp_name}
    vis = visdom.Visdom(**visdom_opts)
    # vis = visdom_opts

    trace = []
    idx = 0
    # colour = ['blue', 'green', 'red', 'yellow', 'purple', 'black', 'pink']
    colour = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'black', 'grey']
    cl_len = len(colour)
    for idx in range(len(xs)):
        # print(xs[idx])
        # print(ys[idx])
        print(names[idx])
        print(idx)
        trace_i = dict(x=xs[idx], y=ys[idx], mode="lines", type='custom',
                       marker={'color': colour[idx % cl_len], 'symbol': 104, 'size': "10"},
                       text=["one", "two", "three"], name=names[idx])
        trace.append(trace_i)
        idx += 1

    layout = dict(title='Timings ', xaxis={'title': 'time in seconds'}, yaxis={'title': 'Percentage solved'})
    vis._send({'data': trace, 'layout': layout, 'win': 'prox'+str(torch.rand(1))})


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--sort', action='store_true')
    args = parser.parse_args()

    plot_pgd_experiments(args.exp_name, args, gpu_=True)

if __name__ == "__main__":
    main()
