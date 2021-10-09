#!/usr/bin/python3

import argparse
import itertools
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from trainer import Cyc_Trainer,Nice_Trainer,P2p_Trainer
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/P2p.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'CycleGan':
        trainer = Cyc_Trainer(config)
    elif config['name'] == 'Munit':
        trainer = Munit_Trainer(config)
    elif config['name'] == 'Unit':
        trainer = Unit_Trainer(config)
    elif config['name'] == 'Nice':
        trainer = Nice_Trainer(config)
    elif config['name'] == 'U-gat':
        trainer = Ugat_Trainer(config)
    elif config['name'] == 'P2p':
        trainer = P2p_Trainer(config)

    trainer.train()
    
    



###################################
if __name__ == '__main__':
    main()