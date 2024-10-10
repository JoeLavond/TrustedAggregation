# Trusted Aggregation (TAG)
Backdoor Defense in Federated Learning 

## Getting Started
Here we provide a step-by-step guide to set up the environment for our experiments. 
Our code is written using Python 3.10, and we provide requirements files for both conda (recommended) and pip environments.

1. Clone the repository
```bash
git clone https://github.com/JoeLavond/TAG.git
```

2. Install the required packages. 
```bash
# using conda
conda install --name tag --file conda_requirements.txt

# using pip
pip install -r requirements.txt
```

## Code Structure
```{bash}
├── <dataset-name>/             # the actual dataset name (e.g., cifar10) 
│   
│   ├── <attack-type>/          # whether (e.g., neuro) or not (e.g., classic) the attack uses the Neurotoxin projection
│   │   ├── base/               # baseline defense methods
│   │   │   ├── robust.py       # robust aggregation (e.g. median, trimmed mean)
│   │   │   ├── trust.py        # FLTrust defense
│   │   ├── tag/                
│   │   │   ├── train.py        # main training script for TAG
│   
├── utils/                      # 
│   ├── modeling/
│   │   ├── pre.py              # 
│   │   ├── resnet.py           # 
│   │   ├── vgg.py              # 
│   ├── training/
│   │   ├── data.py             #
│   │   ├── setup.py            #
│   ├── data.py
│   ├── setup.py
│   
```
