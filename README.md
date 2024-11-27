# Trusted Aggregation (TAG)
Backdoor Defense in Federated Learning  
Authors: Joseph Lavond, Minhao Cheng, and Yao Li

## Overview
This repository contains the code for our paper ["Trusted Aggregation (TAG): Backdoor Defense in Federated Learning"](https://openreview.net/pdf?id=r9eNUDe2im). 

We propose a backdoor defense based on the assumption that we can trust a subset of the clients in a federated learning setting. 
Our defense, TAG, leverages the trustworthiness of these clients to identify and remove unusual local updates before global aggregation.
TAG is able to keep the backdoor attack from poisoning the global model, while maintaining the performance of the benign clients.

## Getting Started
Here we provide a step-by-step guide to set up the environment for our experiments. 
Our code is written using **Python 3.10**, and we provide requirements files for both conda (recommended) and pip environments.

1. Clone the repository
```bash
git clone https://github.com/JoeLavond/TAG.git
```

2. Install the required packages. 
```bash
# using conda
conda create --name tag python=3.10
conda activate tag
conda install --name tag --file conda_requirements.txt

# using pip (and virtualenv)
pip install -r requirements.txt
```

## Usage
To run our main experiment, TAG on CIFAR10 with a simple backdoor attack, use the following command:
```bash
# navigate to the TAG directory for the CIFAR10 dataset
cd ./cifar10/classic/tag

# run the tag training script
# the full experiment requires 250 epochs
# d_scale=2 is the default value for the scaling coefficient of TAG
python train.py --n_epochs=2 --d_scale=2
```

Common federated setup command line arguments include:
- `--n_users`: number of clients in the federated learning setting
- `--n_user_data`: number of data points per client
- `--p_report`: proportion of clients that participate in each round of federated training
- `--alpha`: controls how non-iid (smaller values) or iid (larger values) local data is

Common backdoor attack command line arguments include:
- `--p_malicious`: proportion of clients that are malicious
- `--dba`: whether the backdoor attack is distributed (True) or not (False) among the malicious clients
- `---p_pois`: proportion of the local data that is poisoned by the backdoor attack
- `---target`: target label for the backdoor attack
- `--row_size`: size of the backdoor trigger in rows. Similarly for columns with `--col_size`.

For running other experiments, navigate to the appropriate dataset and attack type directories and run the corresponding training script.
See the code structure section for more details.

## Code Structure
There are two main directories in the repository: `./cifar10` and `./utils`.

Datasets are further organized into whether or not the attack uses the Neurotoxin projection.
Then, each attack type directory contains the baseline defense methods and the TAG training script.
Other datasets (like `./cifar_100` and `./stl_10`) are organized the same way as `./cifar10`. 

The `./utils` directory contains source code for modifying various aspects of the federated learning setup, backdoor attack, and defense methods.
The main implementation(s) needed for TAG are:
- `./utils/training/eval.py`: when we evaluate the global or local models, we need to save the model outputs
- `./utils/training/dist.py`: distance computation between outputs of the global model and the local models

The full code structure is as follows:

```{bash}
├── <dataset-name>/             # the actual dataset name (e.g., cifar10) 
│   ├── <attack-type>/          # whether (e.g., neuro) or not (e.g., classic) the attack uses the Neurotoxin projection
│   │   ├── base/               # baseline defense methods
│   │   │   ├── robust.py       # robust aggregation (e.g. median, trimmed mean)
│   │   │   ├── trust.py        # FLTrust defense
│   │   ├── tag/                
│   │   │   ├── train.py        # main training script for TAG
│   
├── utils/                       
│   ├── modeling/               # various model architectures
│   │   ├── pre.py              # model layer for standardization
│   │   ├── resnet.py            
│   │   ├── vgg.py               
│   ├── training/
│   │   ├── agg.py              # non-standard aggregation methods
│   │   ├── atk.py              # visible backdoor attack
│   │   ├── dist.py             # distance computation
│   │   ├── eval.py             
│   │   ├── neuro.py            # neurotoxin projection 
│   │   ├── train.py             
│   ├── data.py                 # custom data sets for federated data
│   ├── setup.py                # logging and reproducibility
│   
```

## Citation
If you find our work useful, please consider citing our paper:
```bibtex
@article{
lavond2024trusted,
title={Trusted Aggregation ({TAG}): Backdoor Defense in Federated Learning},
author={Joseph Lavond and Minhao Cheng and Yao Li},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=r9eNUDe2im},
note={}
}
```
