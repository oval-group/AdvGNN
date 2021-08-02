import argparse

########################################################################
#   command line arguments for run_attack.py
#   TODO:
#       separate the arguments into different categories (e.g. dataset, model etc.)
#       make help definitions more informative
########################################################################


def get_args():
    parser = argparse.ArgumentParser()

    # _add_dataset_parser(parser)
    # _add_model_parser(parser)
    # _add_optimization_parser(parser)
    # _add_loss_parser(parser)
    _add_misc_parser(parser)

    args = parser.parse_args()
    return args


def _add_misc_parser(parser):

    parser.add_argument('--record_name', type=str, help='file to save results')
    parser.add_argument('--pdprops', type=str, default='jodie-base_hard.pkl',
                        help='pandas table with all props we are interested in')
    parser.add_argument('--cpu', action='store_true', help='run experiments on cpus rather than CUDA')
    parser.add_argument('--nn_name', type=str, help='network architecture name', default='cifar_base_kw')
    parser.add_argument('--bin_search_eps', type=float, default=1e-3, help='epsilon for binary search')
    parser.add_argument('--pgd_steps', type=float, default=1e4, help='number of steps for pgd attack')
    parser.add_argument('--pgd_abs_lr', type=float, help='abs lr for pgd attack')
    parser.add_argument('--random_restarts', type=int, default=0,
                        help='number of times we restart PGD if unsuccesful')
    parser.add_argument('--table_name', type=str, help='optional name of the result table')
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--num_props', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=None,
                        help="seed for pseudo-randomness")
    parser.add_argument('--change_eps_const', type=float,
                        help="change eps by a constant to make the dataset easier or harder")
    parser.add_argument('--adv_method', type=str, default='pgd_attack',
                        choices=["pgd_attack", "GNN", "mi_fgsm_attack", "a_pgd_attack"])

    parser.add_argument('--pgd_iters', type=int, help='pgd_iters', default=10000)
    parser.add_argument('--pgd_optimizer', type=str, default='default',
                        choices=["adam", "sgd", "default", "SGLD", "GNN"])
    parser.add_argument('--pgd_optimizer_lr', type=float, help='learning rate pgd attack', default=1e-3)
    parser.add_argument('--count_particles', type=int, help='count runs', default=100)
    parser.add_argument('--check_adv', type=int, default=100,
                        help='number of iters after which we should check for adv')
    parser.add_argument('--pgd_momentum', type=float, help='decay factor mu for mi-gsm', default=1e-1)

    parser.add_argument('--SAT_GNN_name', type=str, default=None, help='load a pretrained GNN if not None')
    parser.add_argument('--GNN_grad_feat', action='store_true', help='the gradient is one of the GNN features')
    parser.add_argument('--GNN_rel_decay', action='store_true',
                        help='decay the stepsize proportionally rather than linearly')
    parser.add_argument('--GNN_lr_init', type=float, help='inital step size for GNN', default=1e-2)
    parser.add_argument('--GNN_lr_fin', type=float, help='final step size for GNN', default=1e-3)
    parser.add_argument('--GNN_iters', type=int, help='pgd_iters', default=20)
    parser.add_argument('--GNN_momentum', type=float, help='GNN momentum', default=None)
    parser.add_argument('--GNN_adam', action='store_true', help='using adam with GNN')
    parser.add_argument('--GNN_optimized', action='store_true', help='using adam with GNN')
    parser.add_argument('--old_GNN', action='store_true',
                        help='old fashioned GNN without momentum and with linearly decaying stepsize')

    parser.add_argument('--mi_fgsm_set_alpha', action='store_true',
                        help='set to True if we want to manualy set alpha (the step size) rather than using eps/T')
    parser.add_argument('--mi_fgsm_decay', action='store_true', help='set to true if we want to decay the stepsize')

    parser.add_argument('--dump_domain', type=str, default=None, help='directory to dump subdomains in')

    parser.add_argument('--max_solver_batch', type=float, default=10000,
                        help='max batch size for bounding computations')

    parser.add_argument('--run_lp', action='store_true', help='running an lp before the attack')
    parser.add_argument('--lp_init', action='store_true', help='initialize the attack with lp primal')

    parser.add_argument('--printing', action='store_true', help='print results for each prop')

    parser.add_argument('--lp_type', type=str, default='adam', choices=["adam", "KW", "naive_KW", "naive"])
