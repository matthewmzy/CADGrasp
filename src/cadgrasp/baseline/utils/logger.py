import os
from os.path import dirname, join, abspath
import glob
from datetime import datetime
import yaml
import torch
import wandb
from cadgrasp.baseline.utils.config import to_dict

class Logger:
    def __init__(self, config: dict):
        """
            automatically create experiment directory and save config
            config needs to have at least 'exp_name'
        """
        self.config = config
        if config.exp_name != 'temp':
            wandb.init(project='DexGraspNet2', 
                    name=config.exp_name, 
                    config=config)
            wandb.run.log_code(root='./src')

        # create exp directory
        exp_path = join('experiments', config.exp_name)
        os.makedirs(exp_path, exist_ok=True)
        log_path = join(exp_path, 'log')
        os.makedirs(log_path, exist_ok=True)
        self.ckpt_path = join(exp_path, 'ckpt')
        os.makedirs(self.ckpt_path, exist_ok=True)

        # save config
        with open(join(exp_path, 'config.yaml'), 'w') as f:
            yaml.dump(to_dict(config), f)

    def log(self, dic: dict, mode: str, step: int):
        """
            log a dictionary, requires all values to be scalar
            mode is used to distinguish train, val, ...
            step is the iteration number
        """
        if self.config.exp_name != 'temp':
            wandb.log({f'{mode}/{k}': v for k, v in dic.items()}, step=step)
    
    def save(self, dic: dict, step: int):
        """
            save a dictionary to a file
        """
        torch.save(dic, join(self.ckpt_path, f'ckpt_{step}.pth'))