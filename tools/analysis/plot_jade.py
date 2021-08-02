import argparse
import mlogger
import visdom
import torch as th


def plot_jade(args):
    exp_name = args.GNN_name
    visdom_opts = {'server': 'http://themis.robots.ox.ac.uk',
                                         'port': 9018,
                                         'env': '{}'.format(exp_name)}
    new_plotter = mlogger.VisdomPlotter(visdom_opts)

    save_path = './adv_exp/GNNs/jade'
    try:
        new_xp = mlogger.load_container(f'{save_path}/{exp_name}/state.json')
        new_xp.plot_on(new_plotter)
        new_plotter.update_plots()
    except Exception:
        pass

    vis = visdom.Visdom(**visdom_opts)
    for exp in ['val', 'deep', 'wide']:
        try:
            dict_ = th.load(f'{save_path}/{exp_name}/{exp}')
            vis._send(dict_)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GNN_name', type=str, help='GNN name to copy', required=True)
    args = parser.parse_args()

    plot_jade(args)


if __name__ == '__main__':
    main()
