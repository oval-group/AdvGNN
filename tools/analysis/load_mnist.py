import argparse
import os
import torch
import time
import copy
import random
from torch import nn
from tools.bab_tools.model_utils import mnist_model, mnist_model_deep, add_single_prop


def load_mnist_wide_net(idx, network="wide", mnist_test = None, test=None, printing=False):
    if network == "wide":
        model_name = './models/mnist_wide_kw.pth'
        model = mnist_model()
    else:
        # "deep"
        model_name = './models/mnist_deep_kw.pth'
        model = mnist_model_deep()
    model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    if mnist_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        mnist_test = datasets.MNIST("./mnistdata/", train=False, download=True, transform =transforms.ToTensor())

    x,y = mnist_test[idx]
    x = x.unsqueeze(0)
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()

    if printing:
        print('predicted label ', y_pred, ' correct label ', y)
    if  y_pred != y:
        print('model prediction is incorrect for the given model')
        return None, None, None, None, None
    else:
        if test ==None:
            choices = list(range(10))
            choices.remove(y_pred)
            test = random.choice(choices)

        if printing:
            print('tested against ',test)
        for p in model.parameters():
            p.requires_grad =False

        layers = list(model.children())
        added_prop_layers = add_single_prop(layers, y_pred, test)
        return x, added_prop_layers, test, y_pred, model


def make_elided_models(model, return_error=False):
    """
    Default is to return GT - other
    Set `return_error` to True to get instead something that returns a loss
    (other - GT)
    mono_output=False is an argument I removed
    """
    elided_models = []
    layers = [lay for lay in model]
    assert isinstance(layers[-1], nn.Linear)

    net = layers[:-1]
    last_layer = layers[-1]
    nb_classes = last_layer.out_features

    for gt in range(nb_classes):
        new_layer = nn.Linear(last_layer.in_features,
                              last_layer.out_features-1)

        wrong_weights = last_layer.weight[[f for f in range(last_layer.out_features) if f != gt], :]
        wrong_biases = last_layer.bias[[f for f in range(last_layer.out_features) if f != gt]]

        if return_error:
            new_layer.weight.data.copy_(wrong_weights - last_layer.weight[gt])
            new_layer.bias.data.copy_(wrong_biases - last_layer.bias[gt])
        else:
            new_layer.weight.data.copy_(last_layer.weight[gt] - wrong_weights)
            new_layer.bias.data.copy_(last_layer.bias[gt] - wrong_biases)

        layers = copy.deepcopy(net) + [new_layer]
        # if mono_output and new_layer.out_features != 1:
        #     layers.append(View((1, new_layer.out_features)))
        #     layers.append(nn.MaxPool1d(new_layer.out_features,
        #                                stride=1))
        #     layers.append(View((1,)))
        new_elided_model = nn.Sequential(*layers)
        elided_models.append(new_elided_model)
    return elided_models

