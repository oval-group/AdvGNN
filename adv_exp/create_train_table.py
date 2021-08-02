import torch as th
import pandas as pd
from tools.bab_tools.model_utils import load_cifar_1to1_exp
import torchvision.datasets as cifar_datasets
import torchvision.transforms as transforms


####################################################################
#     Creates datasets for training and fintuning the GNN
####################################################################


def create_data_table():
    datasets = ['jodie-base_easy.pkl',
                'jodie-base_med.pkl',
                'jodie-base_hard.pkl',
                'jodie_base_val.pkl',
                'jodie-deep.pkl',
                'jodie-wide.pkl',
               ]
    im_idces = []
    for d_i in datasets:
        path = './batch_verification_results/'

        table_ = pd.read_pickle(path + d_i)
        im_idces += list(table_['Idx'])
    im_idces.sort()
    print(im_idces)

    nn_name = 'cifar_base_kw'
    new_imag_list = []
    prop_list = []

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    cifar_test = cifar_datasets.CIFAR10('./cifardata/', train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), normalize]))

    for imag_idx in range(10000):
        if imag_idx not in im_idces:
            try:
                _, _, prop_idx, true_class, _ = load_cifar_1to1_exp(nn_name, int(imag_idx), return_true_class=True,
                                                                    cifar_test=cifar_test)
                # print(f"image_idx {imag_idx}, true: {true_class}, prop {prop_idx}")
                new_imag_list.append(imag_idx)
                # prop_list.append(int(prop_idx))
                prop_list.append(prop_idx)
                # input("Wait")
            except Exception:
                # print(imag_idx, 'failed')
                continue
    print('num new images', len(new_imag_list))
    assert(len(new_imag_list) == len(prop_list))
    bnb_ids = range(len(prop_list))
    _columns = ['Idx', 'Eps', 'prop', 'BSAT']
    graph_df = pd.DataFrame(index=bnb_ids, columns=_columns)
    graph_df['Idx'] = new_imag_list
    graph_df['Eps'] = [0.25]*len(new_imag_list)
    graph_df['BSAT'] = ['unknown']*len(new_imag_list)
    graph_df['prop'] = prop_list
    print(graph_df)
    record_name = './cifar_results/adv_results/train_eps025.pkl'
    graph_df.to_pickle(record_name)


def create_data_table_deep():
    # get average epsilon from training dataset
    table_train = pd.read_pickle('./batch_verification_results/jade/train_SAT_jade.pkl')
    # table_train = pd.read_pickle('./batch_verification_results/jade/train_large_easy_SAT_jade.pkl')
    eps_list = list(table_train['Eps'])
    eps_avg = float(th.FloatTensor(eps_list).mean())
    print(eps_avg)

    datasets = [
              'jodie-deep.pkl',
             ]
    im_idces = []
    for d_i in datasets:
        path = './batch_verification_results/'

        table_ = pd.read_pickle(path + d_i)

        im_idces += list(table_['Idx'])
    im_idces.sort()
    print(im_idces)

    nn_name = 'cifar_deep_kw'
    new_imag_list = []
    prop_list = []

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    cifar_test = cifar_datasets.CIFAR10('./cifardata/', train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), normalize]))

    table_len = 500
    table_ctd = 0
    for imag_idx in range(10000):
        if table_ctd >= table_len:
            break
        if imag_idx not in im_idces:
            try:
                _, _, prop_idx, true_class, _ = load_cifar_1to1_exp(nn_name, int(imag_idx), return_true_class=True,
                                                                    cifar_test=cifar_test)
                new_imag_list.append(imag_idx)
                prop_list.append(prop_idx)
                table_ctd += 1
            except Exception:
                continue
    print('num new images', len(new_imag_list))
    assert(len(new_imag_list) == len(prop_list))
    bnb_ids = range(len(prop_list))
    _columns = ['Idx', 'Eps', 'prop', 'BSAT']
    graph_df = pd.DataFrame(index=bnb_ids, columns=_columns)
    graph_df['Idx'] = new_imag_list
    graph_df['Eps'] = [eps_avg]*len(new_imag_list)
    graph_df['BSAT'] = ['unknown']*len(new_imag_list)
    graph_df['prop'] = prop_list
    print(graph_df)
    record_name = f'./cifar_results/adv_results/deep_finetuning_{eps_avg}.pkl'
    graph_df.to_pickle(record_name)


def create_data_table_wide(eps=0.25):
    datasets = [
              'jodie-wide.pkl',
             ]
    im_idces = []
    for d_i in datasets:
        path = './batch_verification_results/'

        table_ = pd.read_pickle(path + d_i)
        im_idces += list(table_['Idx'])
    im_idces.sort()
    print(im_idces)

    nn_name = 'cifar_wide_kw'
    new_imag_list = []
    prop_list = []

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    cifar_test = cifar_datasets.CIFAR10('./cifardata/', train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), normalize]))

    table_len = 500
    table_ctd = 0
    for imag_idx in range(10000):
        if table_ctd >= table_len:
            break
        if imag_idx not in im_idces:
            try:
                _, _, prop_idx, true_class, _ = load_cifar_1to1_exp(nn_name, int(imag_idx), return_true_class=True,
                                                                    cifar_test=cifar_test)
                new_imag_list.append(imag_idx)
                prop_list.append(prop_idx)
                table_ctd += 1
            except Exception:
                continue
    print('num new images', len(new_imag_list))
    assert(len(new_imag_list) == len(prop_list))
    bnb_ids = range(len(prop_list))
    _columns = ['Idx', 'Eps', 'prop', 'BSAT']
    graph_df = pd.DataFrame(index=bnb_ids, columns=_columns)
    graph_df['Idx'] = new_imag_list
    graph_df['Eps'] = [eps]*len(new_imag_list)
    graph_df['BSAT'] = ['unknown']*len(new_imag_list)
    graph_df['prop'] = prop_list
    print(graph_df)
    record_name = f'./cifar_results/adv_results/wide_finetuning_{eps}.pkl'
    graph_df.to_pickle(record_name)


def create_data_table_madry(eps=0.25):
    datasets = [
              'jodie-wide.pkl',
             ]
    im_idces = []
    for d_i in datasets:
        path = './batch_verification_results/'

        table_ = pd.read_pickle(path + d_i)
        im_idces += list(table_['Idx'])
    im_idces.sort()
    print(im_idces)

    nn_name = 'cifar_madry'
    new_imag_list = []
    prop_list = []

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    cifar_test = cifar_datasets.CIFAR10('./cifardata/', train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), normalize]))

    table_len = 500
    table_ctd = 0
    for imag_idx in range(10000):
        if table_ctd >= table_len:
            break
        if imag_idx not in im_idces:
            try:
                _, _, prop_idx, true_class, _ = load_cifar_1to1_exp(nn_name, int(imag_idx), return_true_class=True,
                                                                    cifar_test=cifar_test)
                if prop_idx:
                    new_imag_list.append(imag_idx)
                    prop_list.append(prop_idx)
                    table_ctd += 1
            except Exception:
                continue
    print('num new images', len(new_imag_list))
    assert(len(new_imag_list) == len(prop_list))
    bnb_ids = range(len(prop_list))
    _columns = ['Idx', 'Eps', 'prop', 'BSAT']
    graph_df = pd.DataFrame(index=bnb_ids, columns=_columns)
    graph_df['Idx'] = new_imag_list
    graph_df['Eps'] = [eps]*len(new_imag_list)
    graph_df['BSAT'] = ['unknown']*len(new_imag_list)
    graph_df['prop'] = prop_list
    print(graph_df)
    record_name = f'./cifar_results/adv_results/madry_finetuning_{eps}.pkl'
    print(record_name)
    input("wati")
    graph_df.to_pickle(record_name)


def create_data_table_mnist(eps=0.25):
    from adv_exp.analysis.load_mnist import load_mnist_wide_net
    im_idces = []

    nn_name = 'wide'
    new_imag_list = []
    prop_list = []

    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    mnist_test = datasets.MNIST("./mnistdata/", train=False, download=True, transform=transforms.ToTensor())

    table_len = 500
    table_ctd = 0
    for imag_idx in range(10000):
        if table_ctd >= table_len:
            break
        if imag_idx not in im_idces:
            _, _, prop_idx, true_class, _ = load_mnist_wide_net(int(imag_idx), network='wide', mnist_test=mnist_test)
            print("success", prop_idx, true_class)
            if prop_idx:
                new_imag_list.append(imag_idx)
                prop_list.append(prop_idx)
                table_ctd += 1
    print('num new images', len(new_imag_list))
    assert(len(new_imag_list) == len(prop_list))
    bnb_ids = range(len(prop_list))
    _columns = ['Idx', 'Eps', 'prop', 'BSAT']
    graph_df = pd.DataFrame(index=bnb_ids, columns=_columns)
    graph_df['Idx'] = new_imag_list
    graph_df['Eps'] = [eps]*len(new_imag_list)
    graph_df['BSAT'] = ['unknown']*len(new_imag_list)
    graph_df['prop'] = prop_list
    print(graph_df)
    record_name = f'./cifar_results/adv_results/mnist_wide_finetuning_{eps}.pkl'
    print(record_name)
    input("wati")
    graph_df.to_pickle(record_name)


def make_table_easier(name, eps):
    if name == 'val':
        table = 'val_SAT_jade.pkl'
    elif name == 'base':
        table = 'base_easy_SAT_jade.pkl'
    elif name == 'wide':
        table = 'wide_SAT_jade.pkl'
    elif name == 'deep':
        table = 'deep_SAT_jade.pkl'

    path = './batch_verification_results/jade/'
    graph_df = pd.read_pickle(path + table)

    _columns = ['Idx', 'Eps', 'prop', 'BSAT']
    graph_new = pd.DataFrame(index=graph_df.index, columns=_columns)
    graph_new['Idx'] = graph_df['Idx']
    graph_new['prop'] = graph_df['prop']
    graph_new['BSAT'] = True
    graph_new['Eps'] = graph_df['Eps'] + eps

    new_name = 'easy_' + table
    print(graph_df)
    print(graph_new)
    print(path + new_name)
    graph_new.to_pickle(path + new_name)


def main():
    create_data_table_mnist(eps=0.20)
    # create_data_table_madry(eps=0.20)
    # create_data_table()
    # create_data_table_deep()
    # create_data_table_wide(eps=0.20)

    # for name in ['val', 'base', 'wide', 'deep']:
    #     make_table_easier(name, 0.01)


if __name__ == "__main__":
    main()

