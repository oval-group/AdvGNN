import argparse

########################################################################
#   command line arguments for the training file
#   TODO:
#   - todo remove commented out code
########################################################################


def get_args():
    parser = argparse.ArgumentParser()

    _add_dataset_parser(parser)
    _add_model_parser(parser)
    _add_optimization_parser(parser)
    _add_loss_parser(parser)
    _add_misc_parser(parser)

    args = parser.parse_args()
    return args


def _add_dataset_parser(parser):
    parser.add_argument('--dataset', nargs='?', default='cifar10', type=str,
                        help='what dataset are we using')
    parser.add_argument('--train_dir', nargs='?', default=('/data0/florian/SAT_GNN_training_dataset/'), type=str,
                        help='directory of sets of training datasets')
    parser.add_argument('--train_sub_dir', nargs='?', default=('2021_01_06'), type=str,
                        help='subdirectory of args.train_dir where our explicit dataset is stored')
    parser.add_argument('--val_sub_dir', nargs='?', default=None, type=str,
                        help='subdirectory of args.train_dir where our explicit dataset is stored')

    parser.add_argument('--max_train_size', type=int, default=None,
                        help="maximum size of the training data set (number of subdomains)")
    parser.add_argument('--max_num_prop', type=int, default=None,
                        help="maximum size of the training data set"
                             "(number of properties for which we have a many subdomains)")
    parser.add_argument('--val_size', type=int, default=None,
                        help="number of images in validation set")
    parser.add_argument('--pick_data_rdm', action='store_true',
                        help='flag whether we randomly pick the training dataset or we pick the first k ones'
                             'adv random: better results. disadv: not reproducable (unless we fix seed)')
    parser.add_argument('--duplicate', action='store_true',
                        help='duplicate subdomains so that every property has the same number of subdomains')
    parser.add_argument('--batch_size', type=int, default=1,
                        help="maximum size of the training data set")
    parser.add_argument('--nn_name', type=str, help='network architecture name', default='cifar_base_kw')
    parser.add_argument('--pdprops', type=str, default='train_SAT_bugfixed.pkl',
                        help='pandas table with all props we are interested in')


def _add_model_parser(parser):
    parser.add_argument('--T', nargs='?', default=1, type=int,
                        help='number of embedding layer updates')
    parser.add_argument('--p', nargs='?', default=32, type=int,
                        help='dimension of embedding vectors')
    parser.add_argument('--GNN_lr_init', type=float, default=0.01,
                        help='intial GNN stepsize')
    parser.add_argument('--GNN_lr_fin', type=float, default=0.001,
                        help='final GNN stepsize')
    parser.add_argument('--GNN_optimizer', type=str, default='GNN',
                        choices=["adam", "sgd", "default", "SGLD", "GNN"],
                        help='TODO')
    parser.add_argument('--feature_grad', action='store_true', help='include gradient in features')
    parser.add_argument('--feature_lp_primal', action='store_true', help='include lp primal in features')
    parser.add_argument('--lp_init', action='store_true', help='initialize GNN with lp primal')
    parser.add_argument('--GNN_momentum', type=float, help='GNN momentum', default=None)
    parser.add_argument('--GNN_adam', action='store_true', help='using adam with GNN')
    parser.add_argument('--GNN_rel_decay', action='store_true',
                        help='decay the stepsize proportionally rather than linearly')


def _add_optimization_parser(parser):
    parser.add_argument('--epoch', nargs='?', default=100, type=int,
                        help='number of epoches')
    parser.add_argument('--opt', type=str, required=True, choices=("rmsprop", "adagrad", "sgd", "adam"),
                        help='which optimizer to use')
    parser.add_argument('--lr', nargs='?', default=0.001, type=float,
                        help='learning rate for sgd')
    parser.add_argument('--lr_decay', type=int, default=[-1], nargs='+',
                        help="epochs when we decay lr")
    parser.add_argument('--lr_decay_factor', nargs='?', default=0.1, type=float,
                        help='decay factor')
    parser.add_argument('--reset_optimizer', type=bool, default=False,
                        help='also load optimizer when loading model?')
    parser.add_argument('--weight_decay', nargs='?', default=0, type=float,
                        help='weight decay')


def _add_loss_parser(parser):
    parser.add_argument('--horizon', type=int, default=1,
                        help='how far too look into the future to')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='reinforcement learning decay factor')


def _add_misc_parser(parser):
    parser.add_argument('--seed', type=int, default=None,
                        help="seed for pseudo-randomness")
    parser.add_argument('--cpu',  action='store_true',
                        help='use CPU (default is GPU if GPU is available)')
    parser.add_argument('--logger', action='store_true',
                        help='turn logger on (default is off)')
    parser.add_argument('--visdom', action='store_true',
                        help='turn visdom on (default is off)')
    parser.add_argument('--visdom_server', type=str, default='http://themis.robots.ox.ac.uk',
                        help='turn visdom on (default is off)')
    parser.add_argument('--visdom_port', type=int, default=9018,
                        help='turn visdom on (default is off)')
    parser.add_argument('--exp_name', nargs='?', default='default_name', type=str,
                        help='name of the experiment for the logger')
    parser.add_argument('--save_path', nargs='?', default='./adv_exp/GNNs/',
                        type=str, help='the path to save the GNN')
    parser.add_argument('--save_model', type=int, default=None,
                        help="frequency the model should be saved at")
    parser.add_argument('--load_model', default=None, help='data file with model')
    parser.add_argument('--max_solver_batch', type=float, default=10000,
                        help='max batch size for bounding computations')
    parser.add_argument('--lp_type', type=str, default='adam', choices=["adam", "KW", "naive_KW", "naive"])
