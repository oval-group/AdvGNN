import os
import torch as th
import pandas as pd
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.titlepad'] = 10


def make_cactus_plot(m_timings, timeout_=100):
        m_timings.sort()

        # Make it an actual cactus plot
        axis_min = 0
        y_min = 0
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

        return xs, ys 

def get_arrays(exp_type='easy_experiments', model='base', timeout_=100, combine_=False):
    path_ = './adv_exp/UAI_experiments/'
    datasets_name = os.listdir(f"{path_}/{exp_type}/{model}/")
    datasets = datasets_name
    datasets.sort()
    datasets_name.sort()

#
    order_ = ['pgd', 'fgsm', 'CW', 'GNN']
    datasets = [d for w in order_ for d in datasets if w in d]
    datasets_name = datasets
#
    tables = []
    for d_i in datasets:
        table_ = pd.read_pickle(f"{path_}/{exp_type}/{model}/{d_i}")
        tables.append(table_)

    create_table = True
    if create_table:
        _columns = ['Exp_name', 'number_props', 'Method', 'seed', 'Time(s)', 'Timeout(%)']
        graph_df = pd.DataFrame(columns=_columns)

    x_list = []
    y_list = []

    timeout_ = 100

    method_list = []

    if combine_:
        combined_list = {}

    for idx_, t_i in enumerate(tables):
        if 'GNN' in datasets_name[idx_]:
            method = 'AdvGNN'
        elif 'mi_fgsm' in datasets_name[idx_]:
            method = 'MI-FGSM+'
        elif 'pgd' in datasets_name[idx_]:
            method = 'PGD Attack'
        elif 'CW' in datasets_name[idx_]:
            method = 'C&W'
        method_list.append(method)

        string_ = datasets_name[idx_] 
        # seed = string_.split("seed", 1)[1].split(".", 1)[0]
        try:
            seed = string_.split("seed",1)[1].split(".",1)[0]
        except Exception:
            seed=0

        t_i = t_i.dropna(how='any')
        table_ = t_i

  #      table_.loc[table_['BTime_PGD'] > timeout_, 'BSAT'] = 'timeout'
 #       table_.loc[table_['BSAT'] == 'timeout', 'BTime_PGD'] = timeout_ + 1
 
        #keys_time = [key_ for key_ in t_i.keys() if 'BTime' in key_][0]

        keys_time = [key_ for key_ in t_i.keys() if 'Time' in key_][0]
        keys_SAT = [key_ for key_ in t_i.keys() if 'SAT' in key_][0]
        t_i.loc[t_i[keys_time] > timeout_, keys_SAT] = 'timeout'
        t_i.loc[t_i[keys_SAT] == 'timeout', keys_time] = timeout_ + 1

        m_timings = table_[keys_time].tolist()

        if combine_:
            if method in combined_list.keys():
                combined_list[method] += m_timings
            else:
                combined_list[method] = m_timings
        else:
         xs, ys = make_cactus_plot(m_timings, timeout_=timeout_)
         x_list.append(xs)
         y_list.append(ys)

    if combine_:
        method_list = combined_list.keys()
        for method in method_list:
            m_timings = combined_list[method]
            xs, ys = make_cactus_plot(m_timings, timeout_=timeout_)
            x_list.append(xs)
            y_list.append(ys)

    return datasets, datasets_name, x_list, y_list, method_list


def plot_UAI(exp_type='easy_experiments', model='base', timeout_=100, fig_name_=None, combine_=False):

    _, _, x_list, y_list, method_list = get_arrays(exp_type=exp_type, model=model, timeout_=timeout_, combine_=combine_)

    plot_names = {
        'GNN': 'AdvGNN',
        'pgd': 'PGD',
        'mi-fgsm': 'MI-FGSM+',
        'CW': 'C&W',
    }

    title_names = {
        'base_easy': "Base model - easy properties",
        'base': "Base model",
        'wide_easy': "Wide large model - easy properties",
        'wide': "Wide large model",
        'deep_easy': "Deep large model - easy properties",
        'deep': "Deep large model",
    }

    fig_path = './adv_exp/UAI_experiments/plots/'
    if exp_type == 'easy_experiments':
        # fig_name = fig_path + f"{model}_easy.pdf"
        title = f"`{model}' model - easy properties".title()
    else:
        # fig_name = fig_path + f"{model}.pdf"
        title = f"`{model}' model".title()

    fig_name = fig_path + f"{model}"
    if exp_type == 'easy_experiments':
        fig_name += '_easy'
    if combine_:
        fig_name += '_combined'
    fig_name += '.pdf'

    list_of_times = y_list

    starting_point = min([t[0] for t in list_of_times])

    method_2_color = {}
    method_2_color['AdvGNN'] = 'red'
    method_2_color['PGD Attack'] = 'green'
    method_2_color['MI-FGSM+'] = 'gold'
    # method_2_color['adam'] = 'gold'
    method_2_color['C&W'] = 'skyblue'
    # method_2_color['GNN_GNN'] = 'darkmagenta'
    # method_2_color['Branching_GNN'] = 'RoyalBlue'
    # method_2_color['eran'] = 'Black'
    fig = plt.figure(figsize=(10, 10))
    ax_value = plt.subplot(1, 1, 1)
    ax_value.axhline(linewidth=3.0, y=100, linestyle='dashed', color='grey')

    y_min = 0
    y_max = 100
    ax_value.set_ylim([y_min, y_max+5])

    axis_min = min([t[0] for t in x_list])
    x_axis_max = timeout_

    ax_value.set_xlim([axis_min, x_axis_max])

    print(axis_min, x_axis_max)
    # Plot all the properties
    linestyle_dict = {
        'PGD Attack': 'dotted',
        'MI-FGSM+': 'dotted',
        'C&W': 'dotted',
        'AdvGNN': 'solid',
    }

    # for x in x_list:
    #     print(x[0], x[-1])
    # for y in y_list:
    #     print(y[0], y[-1])

    if combine_:
        linewidth = 4.0
    else:
        linewidth = 1.0

    for idx_, method in enumerate(method_list):
        ax_value.plot(x_list[idx_], y_list[idx_], color=method_2_color[method],
                      linestyle=linestyle_dict[method], label=method, linewidth=linewidth)

    ax_value.set_ylabel("Properties successfully attacked [%]", fontsize=22)
    ax_value.set_xlabel("Computation time [s]", fontsize=22)
    # plt.xscale('log', nonposx='clip')
    # plt.yscale('log')
    plt.xscale('linear', nonposx='clip')
    ax_value.legend(fontsize=19.5)
    plt.grid(True)
    plt.title(title)
    plt.savefig(fig_name, format='pdf', dpi=300)


def main():
 for exp_type in ['experiments', 'easy_experiments']:
  for model in ['base', 'wide', 'deep']:
   for combine_ in [True, False]:
    timeout_ = 20 if exp_type=='easy_experiments' else 100
    plot_UAI(exp_type=exp_type, model=model, combine_=combine_, timeout_=timeout_)


if __name__ == "__main__":
    main()
