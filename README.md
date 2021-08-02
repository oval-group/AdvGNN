# Generating Adversarial Examples with Graph Neural Networks

This repository contains the implementation of the paper [Generating Adversarial Examples with Graph Neural Networks](https://arxiv.org/pdf/2105.14644.pdf) in PyTorch. If you use this work for your research, please cite the paper:

```
@article{jaeckle2021generating,
  title={Generating Adversarial Examples with Graph Neural Networks},
  author={Jaeckle, Florian and Kumar, M Pawan},
  journal={Conference on Uncertainty in Artificial Intelligence},
  year={2021}
}
```

## Repository structure

* `./scripts/` is a set of python scripts. The README in the directory provides further details of how to launch the experiments described in the paper
*  `./adv_exp/` contains implementations of various adversarial attack methods including our own AdvGNN method. It further contains files to train new GNNs and has a folder with trained GNNs used in the UAI paper.
* `./tools/'` contains auxiliary functions to run attacks
* `./datasets/` contains the Verification and Adversarial Attack Datasets referenced in the paper
* `./models/` contains the trained Neural Networks that we are attacking in our experiments
* `./UAI_experiments/` is the directory for result files generated while running attacks on CIFAR-10
* `./lp_solver/` contains an implementation of 

  
## Running the code
### Dependencies
The code was implemented assuming to be run under `python3.6`.
We have a dependency on:
* [Pytorch](http://pytorch.org/) to represent the Neural networks and to use as
  a Tensor library. 

  
### Installation
(TODO: double check whether this is still working)
 
We assume the user's Python environment is based on Anaconda.

```bash
git clone --recursive https://github.com/oval-group/plnn-bab.git

cd plnn-bab

#Create a conda environment
conda create -n plnn-bab python=3.6
conda activate plnn-bab

# Install pytorch to this virtualenv
# (or check updated install instructions at http://pytorch.org)
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch 

# Install the code of this repository
python setup.py install
```

### Execution
Finally, all experiments can be replicated by the different python scripts as explained further in the directory [scripts](scripts)
