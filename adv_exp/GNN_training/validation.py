from adv_exp.GNN_training.utils import process_files, run_lp, load_data
import mlogger
import pandas as pd
import torch as th

########################################################################
#   main file to run validation of the SAT GNN
#   TODO
#   - function that runs the GNN on a dataset
#   - a plot function that plots over GNN iterations rather than epochs
########################################################################


def validation_from_dataset(adv_model, args, dict_files, xp, previous_val_results, epoch, baselines):

    val_iters = args.horizon

    # adv_model.params['iters'] = val_iters

    size_dataset = len([item for sublist in [dict_files[key] for key in dict_files.keys()] for item in sublist])

    # TODO increase number of iters for GNN

    loss_list = [0] * val_iters

    for prop in dict_files.keys():
        # list of current subdomains
        subdoms = dict_files[prop]

        # batch_size = args.batch_size
        batch_size = 10
        mini_batch_num = int((len(subdoms)-2)/batch_size)+1

        # for loop over batches
        for batch_idx in range(mini_batch_num):

            file_idx = batch_idx*batch_size
            if batch_size == 1:
                cur_files = [subdoms[batch_idx]]
            elif batch_idx == (mini_batch_num-1):
                cur_files = subdoms[file_idx:]
            else:
                cur_files = subdoms[file_idx:file_idx+batch_size]

            # size of the current batch relative to the total training dataset
            factor = len(cur_files) / float(size_dataset)

            #     load_data
            data_dict = process_files(args, prop, cur_files, not args.cpu)

            #     run GNN
            adv_model.set_layers(data_dict['layers'])
            adv_examples, is_adv = adv_model.create_adv_examples(data_dict['data'], data_dict['model'],
                                                                 return_criterion='not_early',
                                                                 target=data_dict['target'],
                                                                 init_tensor=data_dict['init_tensor'],
                                                                 lbs_all=data_dict['lbs'], ubs_all=data_dict['ubs'],
                                                                 dual_vars=data_dict['dual_vars'],
                                                                 gpu=(not args.cpu))

            loss_progress = adv_model.loss_progress

            loss_progress = [l * factor for l in loss_progress]
            loss_list = [float(sum(x)) for x in zip(loss_list, loss_progress)]
            # loss_list += loss_progress * factor

    previous_val_results.append((epoch, loss_list))

    plot_val(previous_val_results, val_iters, args, baselines)

    # adv_model.params['iters'] = val_iters

    return previous_val_results


def validation_from_table(adv_model, args, xp, previous_val_results, epoch, baselines, val_type='val'):

    val_iters = args.horizon
    val_size = args.val_size

    if val_type == 'val':
        table_name = 'val_SAT_jade.pkl'
        nn_name = 'cifar_base_kw'
        batch_size = 10
    if val_type == 'wide':
        table_name = 'wide_SAT_jade.pkl'
        nn_name = 'cifar_wide_kw'
        batch_size = 10
    if val_type == 'deep':
        table_name = 'deep_SAT_jade.pkl'
        nn_name = 'cifar_deep_kw'
        batch_size = 10
    if val_type == 'madry':
        table_name = '/madry_easy_SAT_jade.pkl'
        nn_name = 'cifar_madry'
        batch_size = 10

    adv_model.params['num_adv_ex'] = batch_size

    # load all properties
    path = './cifar_results/adv_results/'
    path = './batch_verification_results/jade/'
    gt_results = pd.read_pickle(path + table_name).dropna(how='all')[:val_size]
    factor = 1. / len(gt_results.index)

    # size_dataset = len([item for sublist in [dict_files[key] for key in dict_files.keys()] for item in sublist])
    size_dataset = min(val_size, len(gt_results.index))

    loss_list = [0] * val_iters

    # for loop over images
    for new_idx, idx in enumerate(gt_results.index):

        # load image
        data, model, target, domain, cuda_verif_layers = load_data(gt_results, nn_name, idx, args)
        adv_model.set_layers(cuda_verif_layers)

        # run the lp
        init_tensor, lbs_all, ubs_all, dual_vars, lp_primal = run_lp(cuda_verif_layers, domain, batch_size, args)

        if not args.lp_init:
            init_tensor = None

        #     run GNN
        adv_examples, is_adv = adv_model.create_adv_examples(data, model,
                                                             return_criterion='not_early',
                                                             target=target,
                                                             init_tensor=init_tensor,
                                                             lbs_all=lbs_all, ubs_all=ubs_all,
                                                             dual_vars=dual_vars, lp_primal=lp_primal,
                                                             gpu=(not args.cpu))

        loss_progress = adv_model.loss_progress

        loss_progress = [l * factor for l in loss_progress]
        loss_list = [float(sum(x)) for x in zip(loss_list, loss_progress)]
        # loss_list += loss_progress * factor

    previous_val_results.append((epoch, loss_list))

    adv_model.params['num_adv_ex'] = args.batch_size

    plot_val(previous_val_results, val_iters, args, baselines, win=val_type, title=f'Validation - {val_type}')

    # adv_model.params['iters'] = val_iters

    return previous_val_results


def plot_val(loss_models, num_iter, args, baselines_dict, win='prox', title='Validation'):
    x_ = []
    for i in range(num_iter):
        x_.append(i)

    trace = []
    idx = 0
    # colour = ['blue', 'green', 'red', 'yellow', 'purple', 'black', 'pink']
    colour = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'black', 'grey']
    cl_len = len(colour)
    for iter_, q_ in loss_models:
        trace_i = dict(x=x_, y=q_, mode="lines", type='custom',
                       marker={'color': colour[idx % cl_len], 'symbol': 104, 'size': "10"},
                       text=["one", "two", "three"], name='epoch iter ' + str(iter_))
        trace.append(trace_i)
        idx += 1
    for string_ in baselines_dict.keys():
        q_ = [baselines_dict[string_]['loss']] * num_iter
        trace_i = dict(x=x_, y=q_, mode="lines", type='custom',
                       marker={'color': colour[idx % cl_len], 'symbol': 104, 'size': "10"},
                       text=["one", "two", "three"], name=string_)
        trace.append(trace_i)
        idx += 1

    layout = dict(title=title, xaxis={'title': 'iterations'}, yaxis={'title': 'loss'})

    if args.visdom:
        import visdom
        visdom_opts = {'server': 'http://atlas.robots.ox.ac.uk',
                       'port': 9016, 'env': '{}'.format(args.exp_name)}
        visdom_opts = {'server': args.visdom_server,  # 'http://atlas.robots.ox.ac.uk',
                       'port': args.visdom_port,  # 9016,
                       'env': '{}'.format(args.exp_name)}
        vis = visdom.Visdom(**visdom_opts)
        vis._send({'data': trace, 'layout': layout, 'win': win})
    else:
        dict_dump = {'data': trace, 'layout': layout, 'win': win}
        th.save(dict_dump, f'{args.save_path}/{win}')

