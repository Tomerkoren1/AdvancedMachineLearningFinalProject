# Advanced Machine Learning Final Project
This repository provides PyTorch code for the paper "tBERT: Topic Models and BERT Joining Forces for Semantic Similarity Detection" (https://www.aclweb.org/anthology/2020.acl-main.630/).

The implementation is part of the final project in Advanced Machine Learning Course (RUNI)

## Installation
- Clone this repository
- Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.txt:
```bash
pip install -r .\requirements.txt
```

## Training
- Run the following:
```bash
python train.py
``
- The following parameters are supported:
1. --batch_size - batch size (default=10)
2. --learning_rate - chose learning rate value (default=3e-05)
3. --epoch_num - epochs number (default=9)
4. --use_aug - use data augmentation
5. --use_lda - use LDA topic model 
6. --use_wandb - use wandb
7. --wandb-entity - enter your wandb entity

### Hyper Parameters
We use the wandb sweep feature to run hyper parameters optimization. To reprduce our work please do the following:
- Create an account for wandb and login on your computer. For more details, see https://wandb.ai/site
- Create a new project
- Create a new sweep, copy one of our sweep configuration (one per model type) and paste it in your sweep configuration. 
Our sweep configuration files located at the 'Sweeps' folder. 
- Open the 'main.py' file and mofify the wand.init project and entity parameter:
```
run = wandb.init(project="AML-FP", entity=userArgs.wandb_entity, config=vars(userArgs), reinit=True)
```
- Launch the sweep agent from the command line
