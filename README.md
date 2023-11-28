# Pseudo-Label Correction for Instance-Dependent Noise using Teacher-Student Framework

This is the official repository for the paper [Pseudo-Label Correction for Instance-Dependent Noise using Teacher-Student Framework](https://arxiv.org/abs/2311.14237).
The following sections provide information on replicating the experiments. We test our method on three different datasets MNIST, FashionMNIST, and SVHN. Please download all files in the repository.

## 1. Instance-Dependent Noise (IDN) Generation

We adopted the IDN generation proposed by Chen et al.'s [Beyond Class-Conditional Assumption: A Primary Attempt to Combat Instance-Dependent Label Noise](https://github.com/chenpf1025/IDN/tree/master). Please refer this paper to generate noisy labels with desired noise level.


We use 22%, 32%, 42%, and 52% noise in our method, but 20%, 30%, 40%, and 50% noise for existing methods. This is done to accommodate for methods that do not need a small set of clean data and ensure fair comparisons.

## 2. Pseudo-Label Correction (P-LC) Experiments
All datasets must be downloaded through Pytorch. We test each dataset on all noise levels {0.22, 0.32, 0.42, 0.52} on three different seeds {0, 1, 2}.
### MNIST
The following script must be run to reproduce results for MNIST.



    python3 train_mnist.py --seed 0 --noise_rate 0.22 --epochs_retrain 50


### Fashion-MNIST
The following script must be run to reproduce results for Fashion-MNIST.



    python3 train_fmnist.py --seed 0 --noise_rate 0.22 --epochs_retrain 50
    


### SVHN
The following script must be run to reproduce results for SVHN.



    python3 train_svhn.py --seed 0 --noise_rate 0.22 --epochs_retrain 50
