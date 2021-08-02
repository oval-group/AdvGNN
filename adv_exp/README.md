## Subdirectory structure

* `run_attack.py` runs the main adversarial example experiments
*  `pgd_attack.py, mi_fgsm_attack.py, and GNN_attack.py` contain implementations for [PGD Attack](https://openreview.net/pdf?id=rJzIBfZAb), [MI-FGSM+](https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf), and [AdvGNN](https://arxiv.org/pdf/2105.14644.pdf). All implement the `Attack_Class` superclass found in `attack_class.py`
* `GNN_SAT_optimized.py` implements the GNN architecture that is the main component of AdvGNN.
* `create_dataset.py` is used to generate the new challenging SAT dataset that can be used to compare different adversarial example generation methods.

* `./GNN_training/'` contains code to train or fine-tune new GNNs

* `./GNNs/` contains trained GNNs

