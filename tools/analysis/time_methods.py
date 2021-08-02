import time
import torch
import visdom
from adv_exp.GNN_training.utils import process_files
from adv_exp.pgd_attack import Pgd_Attack
from adv_exp.GNN_attack import GNN_attack

########################################################################
#   time pgd and GNN methods
#   analyse how the batch size influences the time of the different methods
#   TODO:
#   -
########################################################################


def _increase_batch(data_dict, batch_size):
    dict_new = {}
    dict_new['init_tensor'] = torch.repeat_interleave(data_dict['init_tensor'], repeats=batch_size, dim=0)
    dict_new['lbs'] = [torch.repeat_interleave(l, repeats=batch_size, dim=0) for l in data_dict['lbs']]
    dict_new['ubs'] = [torch.repeat_interleave(l, repeats=batch_size, dim=0) for l in data_dict['ubs']]
    dict_new['dual_vars'] = [torch.repeat_interleave(l, repeats=batch_size, dim=0) for l in data_dict['dual_vars']]
    return dict_new


def time_methods(gpu_=True):
    '''
    list of experiments:
    x axis: batch size
    y axis: time taken for 20 steps GNN (with and without gradient), 20/100/1000 steps pgd

    '''

    # load data
    prop = '/data0/florian/SAT_GNN_training_dataset/2021_01_14_hard/train/100_idx_116_prop_5_eps_0.2594921875'
    cur_files = ['/data0/florian/SAT_GNN_training_dataset/2021_01_14_hard/train/100_idx_116_prop_5_eps_0.2594921875/_iter_16_dom_249']
    #  '/data0/florian/SAT_GNN_training_dataset/2021_01_14_hard/train/100_idx_116_prop_5_eps_0.2594921875/_iter_13_dom_242']
    data_dict = process_files(None, prop, cur_files, gpu_)

    # initialize GNN
    adv_params_gnn = {
        'iters': 20,
        'initial_step_size': 1e-2,
        'final_step_size': 1e-4,
        'num_adv_ex': 10,
        'load_GNN': '20210115_train_hard/model-best.pkl',
    }
    adv_model_gnn = GNN_attack(params=adv_params_gnn)
    adv_model_gnn.set_layers(data_dict['layers'])
    if not gpu_:
        adv_model_gnn.GNN.cpu()

    # initialize pgd
    adv_params_pgd = {
        'iters': 20,
        'optimizer': 'default',
        'lr': 1e-2,
        'num_adv_ex': 10,
    }
    adv_model_pgd = Pgd_Attack(adv_params_pgd)

    GNN_20_grad = []
    GNN_20_nograd = []
    pgd_20 = []
    pgd_100 = []
    pgd_1000 = []

    pgd_list = [(20, 1e-1, pgd_20), (100, 1e-1, pgd_100), (1000, 1e-2, pgd_1000)]

    # start for loop over batch size
    # for idx, batch_size in enumerate(range(10, 150, 10)):
    range_ = range(10, 30, 10)
    range_ = range(10, 150, 10)
    for idx, batch_size in enumerate(range_):
        print(idx, batch_size)
        dict_new = _increase_batch(data_dict, batch_size)

        # run GNN
        start_time = time.time()
        adv_model_gnn.params['feature_grad'] = False
        with torch.no_grad():
            adv_examples, is_adv = adv_model_gnn.create_adv_examples(data_dict['data'], data_dict['model'],
                                                                     return_criterion='not_early',
                                                                     target=data_dict['target'],
                                                                     init_tensor=dict_new['init_tensor'],
                                                                     lbs_all=dict_new['lbs'], ubs_all=dict_new['ubs'],
                                                                     dual_vars=dict_new['dual_vars'],
                                                                     gpu=gpu_)
        GNN_20_nograd.append(time.time() - start_time)

        start_time = time.time()
        adv_model_gnn.params['feature_grad'] = True
        with torch.no_grad():
            adv_examples, is_adv = adv_model_gnn.create_adv_examples(data_dict['data'], data_dict['model'],
                                                                     return_criterion='not_early',
                                                                     target=data_dict['target'],
                                                                     init_tensor=dict_new['init_tensor'],
                                                                     lbs_all=dict_new['lbs'], ubs_all=dict_new['ubs'],
                                                                     dual_vars=dict_new['dual_vars'],
                                                                     gpu=gpu_)
        GNN_20_grad.append(time.time() - start_time)

        # run pgd
        for iter_, lr_, list_ in pgd_list:
            adv_model_pgd.params['iters'] = iter_
            adv_model_pgd.params['lr'] = lr_
            start_time = time.time()
            adv_examples, is_adv = adv_model_pgd.create_adv_examples(data_dict['data'], data_dict['model'],
                                                                     return_criterion='not_early',
                                                                     target=data_dict['target'],
                                                                     init_tensor=dict_new['init_tensor'], gpu=gpu_)
            list_.append(time.time() - start_time)

    timings = [('GNN grad', GNN_20_grad), ('GNN no grad', GNN_20_nograd),
               ('pgd20', pgd_20), ('pgd100', pgd_100), ('pgd1000', pgd_1000)]
    plot_timings(timings, range_)


def plot_timings(timings, range_):
    visdom_opts = {'server': 'http://atlas.robots.ox.ac.uk',
                   'port': 9016, 'env': 'timing_timinggpu2'}
    vis = visdom.Visdom(**visdom_opts)
    # vis = visdom_opts

    x_ = []
    for i in range_:
        x_.append(i)

    trace = []
    idx = 0
    # colour = ['blue', 'green', 'red', 'yellow', 'purple', 'black', 'pink']
    colour = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'black', 'grey']
    cl_len = len(colour)
    for name_, values in timings:
        trace_i = dict(x=x_, y=values, mode="lines", type='custom',
                       marker={'color': colour[idx % cl_len], 'symbol': 104, 'size': "10"},
                       text=["one", "two", "three"], name=name_)
        trace.append(trace_i)
        idx += 1

    layout = dict(title='Timings ', xaxis={'title': 'batch size'}, yaxis={'title': 'time in seconds'})
    vis._send({'data': trace, 'layout': layout, 'win': 'prox'})


def main():
    time_methods(gpu_=True)


if __name__ == "__main__":
    main()
