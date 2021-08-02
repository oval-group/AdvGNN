# OVAL Verification and Adversarial Datasets

Here the OVAL datasets can be found. The adversarial dataset introduced in the [Generating Adversarial Examples with Graph Neural Networks](https://arxiv.org/pdf/2105.14644.pdf) paper can be found in the `CIFAR10/SAT/` directory.

The Verification dataset introduced in the [Neural Network Branching for Neural Network Verification](https://openreview.net/pdf?id=B1evfa4tPB) paper can be found in the `CIFAR10/UNSAT` directory.


## Adversarial Datasets
* `base_SAT.pkl`, `wide_SAT.pkl`, and `deep_SAT.pkl` include the datasets described in Figures 1 (a), (b), and (c) respectively.

* `base_easy_SAT.pkl`, `wide_easy_SAT.pkl`, and `deep_easy_SAT.pkl` include the datasets described in Figures 2 (a), (b), and (c) respectively.

* `madry_SAT.pkl` is the dataset used in experiments described in Appendix F3


## How to use the datasets:
The corresponding trained neural networks can be found in `../models/`.
The base, deep, and wide models are normalised with: <img src="https://render.githubusercontent.com/render/math?math=\bar{x}= [0.485,0.456,0.406], \sigma= [0.225,0.225,0.225]">.
The properties are of the following type:  given a correctly classified image <img src="https://render.githubusercontent.com/render/math?math=x"> with label <img src="https://render.githubusercontent.com/render/math?math=y">, find a slighly perturbed image that the neural network misclassifies as <img src="https://render.githubusercontent.com/render/math?math=y' \neq y">. The label <img src="https://render.githubusercontent.com/render/math?math=y'"> is randomly selected, and the allowed perturbation is determined by an epsilon value under <img src="https://render.githubusercontent.com/render/math?math=l_\infty">, which is specific to each network and image (found via binary search) and applied to thenormalised input space.