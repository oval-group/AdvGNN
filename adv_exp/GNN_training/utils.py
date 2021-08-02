import os
import glob
import random
import pickle
import torch as th
import mlogger
import numpy as np
import copy
from adv_exp.run_attack import _run_lp_as_init
from tools.bab_tools.model_utils import load_cifar_1to1_exp
from plnn.proxlp_solver.solver import SaddleLP
from adv_exp.analysis.load_mnist import load_mnist_wide_net


########################################################################
#   Includes the following utility functions needed to train the GNN:
#   - load_dataset(args)
#   - set_optimizer(model, args)
#   - process_files(args, prop, cur_files, _cuda)
#   - load_model_only(model, model_file, cpu) TODO check
#   - load_model_and_optimizer(model, optimizer, path, model_name, cpu)
#   - save_state(model, optimizer, filename)
#   - initialize_logger(args)
#   - set_seed(args, print_out=True)
#   - load_data(gt_results, idx, args)
#   - run_lp(cuda_verif_layers, domain, args)
#   - TODO
#   - process files doesn't actually need prop
########################################################################


def load_dataset(args, val=False):
    # _path contains a number of subdirectories. Each subdirectory corresponds to a single property
    #   and contains a number of files, each corresponding to a subdomain visited in the BaB tree

    if val and args.val_sub_dir:
        _path = args.train_dir + args.val_sub_dir
    else:
        _path = args.train_dir + args.train_sub_dir

    # list of all subdirectories
    list_of_properties = []
    # list of list of all subdomains
    list_of_subdomains = []

    for file in os.listdir(_path):
        d = os.path.join(_path, file)
        if os.path.isdir(d) and len(os.listdir(d)) > 1:
            list_of_properties.append(d)

    list_of_properties.sort()

    # reduce the size of the list of subdirectories
    if args.max_num_prop is not None:
        list_of_properties = list_of_properties[0:args.max_num_prop]
    else:
        args.max_num_prop = len(list_of_properties)

    # number of files per
    if args.max_train_size is not None:
        max_files_per_prop = int(args.max_train_size / args.max_num_prop)

    # dict with the keys being the names of the property
    # and the values being a list of subdomains
    dict_files = dict.fromkeys(list_of_properties)

    for prop_i in list_of_properties:
        # train_dir = f'{_path}/{prop_i}'
        train_dir = prop_i
        assert(os.path.exists(train_dir))
        assert(os.path.exists(train_dir+'/_model')), (train_dir)
        files_train = []
        files_train.extend(glob.glob(train_dir+'/*')[0:300000])
        if len(files_train) == 0:
            continue

        files_train.remove(prop_i+'/_model')
        if args.pick_data_rdm:
            sample_num = min(max_files_per_prop, len(files_train))
            file_picked = random.sample(files_train, sample_num)
        else:
            file_picked = files_train[:max_files_per_prop]

#
        if args.duplicate and len(file_picked) < max_files_per_prop:
            # duplicate files to ensure that that propert is not underpresented in the dataset
            file_picked = (file_picked * (int(max_files_per_prop/len(file_picked))+1))[:max_files_per_prop]
#
        list_of_subdomains.append(file_picked)
        dict_files[prop_i] = file_picked

    print(f"total num properties {len(list_of_properties)}")
    print("num of elem", [len(i) for i in list_of_subdomains])
    print("total elem", sum([len(i) for i in list_of_subdomains]))
    num_props = len(list_of_properties)
    num_subdoms = sum([len(i) for i in list_of_subdomains])

    # to load the data we either need (list_of_properties and list_of_subdomains)
    # or dict_files
    return list_of_properties, list_of_subdomains, dict_files, num_props, num_subdoms


def set_optimizer(model, args):
    decay = args.weight_decay

    if args.opt == 'sgd':
        # optimizer = th.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        optimizer = th.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=decay)
    elif args.opt == 'rmsprop':
        optimizer = th.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=decay)
    elif args.opt == 'adagrad':
        optimizer = th.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=decay)
    elif args.opt == 'adam':
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=decay)
    else:
        raise ValueError(args.opt)

    print("\n\n\n----------------initializing optimizer - lr=", args.lr, "--------------------\n\n\n")
    return optimizer


def _dicts_to_tensor(_dict, _key, _cuda):
    # takes a list of dictionaries and a key and returns a stacked tensor

    _list = []
    for elem_i in _dict:
        _list.append(elem_i[_key])

    if _cuda and th.cuda.is_available():
        return th.stack(_list).cuda()
    else:
        return th.stack(_list)


def _dicts_to_list_of_tensors(_dict, _key, _cuda):
    # takes a list of dictionaries and a key and returns a list of stacked tensors

    lst = []
    for elem_i in _dict:
        lst.append(elem_i[_key])

    new_list = []
    for idx_ in range(len(lst[0])):
        if _cuda and th.cuda.is_available():
            new_list.append(th.stack([item[idx_].squeeze(0) for item in lst]).cuda())
        else:
            new_list.append(th.stack([item[idx_].squeeze(0) for item in lst]))

    if _cuda:
        return new_list
    else:
        return new_list


def process_files(args, prop, cur_files, _cuda):
    # collecting all necessary training data for the GNN
    # need to put all data into a dict of big tensors

    # _path = args.train_dir + args.train_sub_dir + prop

    # graph_info_all is a list of dicts, each containing the data for one subdomain
    graph_info_all = []
    for f in cur_files:
        _file = f
        with open(_file, 'rb') as f:
            net = th.load(f)

        graph_info_all.append(net)

    # now we want to create a dict ("new_dict") whose values are tensors each containing all data for one
    # of the following categories:

    new_dict = {}

    # new_dict['y'] = graph_info_all[0]['y']
    new_dict['data'] = graph_info_all[0]['data']
    new_dict['target'] = graph_info_all[0]['target']
    new_dict['init_tensor'] = _dicts_to_tensor(graph_info_all, 'init_tensor', _cuda)

    for _key in ['lbs', 'ubs', 'dual_vars']:
        new_dict[_key] = _dicts_to_list_of_tensors(graph_info_all, _key, _cuda)

    # add the model we want to verify to the dict
    with open(prop + '/_model', 'rb') as f:
        net = th.load(f)

    if _cuda and th.cuda.is_available():
        new_dict['model'] = net['model'].cuda()
        new_dict['layers'] = [l_i.cuda() for l_i in net['layers']]
    else:
        new_dict['model'] = net['model']
        new_dict['layers'] = net['layers']

    return new_dict


def load_model_only(model, model_file, cpu):
    # loads the GNN but not the optimizer
    # TODO double check whether up to date
    # DELETE and move to utils one level up
    if os.path.exists(model_file):
        if cpu:
            device = th.device('cpu')
            model_state = th.load(model_file, map_location=device)['model']
            model.load_state_dict(model_state)
            print('Loaded best model from {}'.format(model_file))
        else:
            device = th.device('cuda')
            model_state = th.load(model_file)['model']
            model.load_state_dict(model_state)
            model.to(device)
            print('Loaded best model from {}'.format(model_file))
    else:
        print("I tried to find a best model at {} but I did not find any.\n ----> Using current model"
              .format(model_file))
        input("wait")
    return model


def load_model_and_optimizer(model, optimizer, path, model_name, cpu):
    # tries to load both the model and the optimizer
    model_file = path + model_name

    if os.path.exists(model_file):
        if cpu:
            device = th.device('cpu')
            model_state = th.load(model_file, map_location=device)['model']
            model.load_state_dict(model_state)
            print('Loaded best model and optimizer from {}'.format(model_file))
            optimizer_state = th.load(model_file, map_location=device)['optimizer']
            optimizer.load_state_dict(optimizer_state)
        else:
            device = th.device('cuda')
            model_state = th.load(model_file)['model']
            model.load_state_dict(model_state)
            model.to(device)
            print('Loaded best model and optimizer from {}'.format(model_file))
            optimizer_state = th.load(model_file)['optimizer']
            optimizer.load_state_dict(optimizer_state)
    else:
        print("I tried to find a best model at {} but I did not find any.\n ----> Using current model"
              .format(model_file))
    return model, optimizer


def save_state(model, optimizer, filename):
    # save the model and the optimizer
    th.save({'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}, filename)


def initialize_logger(args, baseline_names, num_props, num_subdoms):
    # move visdom data to args (maybe a logger subgroup)
    if args.visdom:
        pl = mlogger.VisdomPlotter({'server': args.visdom_server,  # 'http://atlas.robots.ox.ac.uk',
                                    'port': args.visdom_port,  # 9016,
                                    'env': '{}'.format(args.exp_name)})
    else:
        pl = None

    xp = mlogger.Container()

    dict_args = vars(args)
    dict_args['num_subdoms'] = num_subdoms
    dict_args['num_props'] = num_props
    dict_args['PID'] = os.getpid()

    xp.config = mlogger.Config(visdom_plotter=pl, **vars(args))
    # xp.config = mlogger.Config(visdom_plotter=pl, **vars(args))

    xp.epoch = mlogger.metric.Simple()

    # ######### outer problem q
    xp.loss = mlogger.Container()
    xp.loss.loss = mlogger.metric.Sum(visdom_plotter=pl, plot_title="Targetedloss", plot_legend="GNN accumulated")
    xp.loss.gnn_final = mlogger.metric.Sum(visdom_plotter=pl, plot_title="Targetedloss", plot_legend="GNN final")

    xp.timer = mlogger.Container()
    xp.timer.train = mlogger.metric.Timer(visdom_plotter=pl, plot_title="Time", plot_legend="training")
    xp.timer.val = mlogger.metric.Timer(visdom_plotter=pl, plot_title="Time", plot_legend="validation")

    xp.lossiteration = mlogger.Container()
    for i in range(args.horizon):
        string_ = 'step' + str(i)
        setattr(xp.lossiteration, string_, mlogger.metric.Sum(visdom_plotter=pl, plot_title="Lossiteration",
                                                              plot_legend=string_))

    xp.isadv = mlogger.Container()
    xp.isadv.per = mlogger.metric.Sum(visdom_plotter=pl, plot_title="Is Adversarial",
                                      plot_legend="subdoms - GNN")
    xp.isadv.props = mlogger.metric.Average(visdom_plotter=pl, plot_title="Is Adversarial",
                                            plot_legend="props - GNN")

    xp.opt = mlogger.Container()
    string_ = 'optimizer'
    setattr(xp.opt, string_, mlogger.metric.Average(visdom_plotter=pl, plot_title="Optimizer",
                                                    plot_legend="log_learningrate"))

    for string_ in baseline_names:
        setattr(xp.loss, string_, mlogger.metric.Sum(visdom_plotter=pl, plot_title="Targetedloss",
                                                     plot_legend=string_))
        setattr(xp.isadv, "subdoms - "+string_, mlogger.metric.Sum(visdom_plotter=pl, plot_title="Is Adversarial",
                                                                   plot_legend="subdoms - "+string_))
        setattr(xp.isadv, "props - "+string_, mlogger.metric.Average(visdom_plotter=pl, plot_title="Is Adversarial",
                                                                     plot_legend="props - "+string_))

    return xp


def set_seed(args, print_out=True):
    if args.seed is None:
        np.random.seed(None)
        args.seed = np.random.randint(1e5)
    if print_out:
        print('Seed:\t {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(args.seed)


def load_data(gt_results, nn_name, idx, args):
    imag_idx = gt_results.loc[idx]["Idx"]
    prop_idx = gt_results.loc[idx]['prop']
    eps_temp = gt_results.loc[idx]["Eps"]

    if args.dataset == 'cifar10':
        x, verif_layers, test, y, model = load_cifar_1to1_exp(nn_name, int(imag_idx),
                                                              int(prop_idx), return_true_class=True)
    elif args.dataset == 'mnist':
        x, verif_layers, test, y, model = load_mnist_wide_net(int(imag_idx), network=nn_name, test=int(prop_idx), printing=False)
    else:
        input("Dataset not implemented in load_data")

    domain = th.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
    if not args.cpu and th.cuda.is_available():
        x = x.cuda()
        model.cuda()

    model.eval()

    if not args.cpu and th.cuda.is_available():
        cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
        domain = domain.cuda()
    else:
        cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

    data = (x.squeeze(), y, x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp)

    return data, model, prop_idx, domain, cuda_verif_layers


def run_lp(cuda_verif_layers, domain, batch_size, args):
    init_tensor, ub, lbs_all, ubs_all, dual_vars = _run_lp_as_init(cuda_verif_layers, domain, args)
    init_tensor.squeeze_(0)

    def _repeat_list(list_, args):
        # ctd = args.count_particles
        ctd = batch_size
        return [l.repeat([ctd] + (len(l.size())-1)*[1]) for l in list_]

    lbs_all = _repeat_list(lbs_all, args)
    ubs_all = _repeat_list(ubs_all, args)
    dual_vars = _repeat_list(dual_vars, args)
    init_tensor_rpt = init_tensor.unsqueeze(0).repeat([batch_size] + (len(init_tensor.size()))*[1])

    return init_tensor, lbs_all, ubs_all, dual_vars, init_tensor_rpt
