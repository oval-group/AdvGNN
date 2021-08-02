import os
import argparse


def copy_GNNS(args):
    jade_path = 'jade_robots:/jmain01/home/JAD035/pkm01/fxj15-pkm01/workspace/plnn-bab/adv_exp/GNNs/'
    laptop_path = 'adv_exp/GNNs/jade/'
    if args.machine == 'themis':
        oval_path = 'themis_robots:/home/florian/Verification/plnn-copy/adv_exp/GNNs/jade/'
    else:
        input("not implemented")

    os.system('scp -r "%s%s" "%s%s"' % (jade_path, args.GNN_name, laptop_path, args.GNN_name))
    print('jade to laptop done')
    os.system('scp -r "%s%s" "%s%s"' % (laptop_path, args.GNN_name, oval_path, args.GNN_name))
    print('laptop to oval done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default='themis', help='machine name')
    parser.add_argument('--GNN_name', type=str, help='GNN name to copy', required=True)
    args = parser.parse_args()

    copy_GNNS(args)


if __name__ == '__main__':
    main()
