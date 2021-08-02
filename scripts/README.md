# Running Experiments

The experiments reported in the [paper](https://arxiv.org/pdf/2105.14644.pdf) can be reproduced as follows:


## Main UAI Experiments
The experimental results shown in Figure 1 and Tables 1-3 can be run as follows:

AdvGNN:
`python scripts/UAI_experiments.py --exp_name experiments_GNN_base_wide_deep_seed_2222_seed_3333_seed_4444`

MI-FGSM+:
`python scripts/UAI_experiments.py --exp_name experiments_mi_fgsm_set_alpha_base_wide_deep_seed_2222_seed_3333_seed_4444`

PGD:
`python scripts/UAI_experiments.py --exp_name experiments_pgd_base_wide_deep_seed_2222_seed_3333_seed_4444`

## Easy Experiments:
The experimental results shown in Figure 2 and Tables 10 can be run as follows:

AdvGNN:
`python scripts/UAI_experiments.py --exp_name experiments_GNN_base_wide_deep_easy_seed_2222_seed_3333_seed_4444`

MI-FGSM+:
`python scripts/UAI_experiments.py --exp_name experiments_mi_fgsm_set_alpha_base_wide_deep_easy_seed_2222_seed_3333_seed_4444`

PGD:
`python scripts/UAI_experiments.py --exp_name experiments_pgd_base_wide_deep_easy_seed_2222_seed_3333_seed_4444`

## Further Experiments:
The experiments in Appendices F3 and F4 that are summarized in Tables 11 and 12 can be run as follows:

AdvGNN:
`python scripts/UAI_experiments.py --exp_name rebuttal_madry_GNN_med_seed_2222_3333_4444`

MI-FGSM+:
`python scripts/UAI_experiments.py --exp_name rebuttal_madry_mi_fgsm_set_alpha_med_seed_2222_3333_4444`

PGD:
`python scripts/UAI_experiments.py --exp_name rebuttal_madry_pgd_med_seed_2222_3333_4444`


## GNN Training:
TODO

## Validation Experiments:

The hyper-parameter search for MI-FGSM, MI-FGSM+, and PGD as reported in Appendix E can be run using the following commands: 

MI-FGSM
`python scripts/UAI_experiments.py --exp_name hparam_mi_fgsm_original_first_round`

MI-FGSM+
`python scripts/UAI_experiments.py --exp_name hparam_mi_fgsm_set_alpha_first_round`

PGD:
`python scripts/UAI_experiments.py --exp_name hparam_pgd_first_round`