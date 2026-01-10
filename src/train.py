import os
import sys

import argparse
from pprint import pprint
from tqdm import trange
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from math import sqrt

from cadgrasp.baseline.network.model import get_model
from cadgrasp.baseline.utils.dataset import GraspNetDataset, Loader, minkowski_collate_fn
from cadgrasp.baseline.utils.logger import Logger
from cadgrasp.baseline.utils.config import load_config, add_argparse
from cadgrasp.baseline.utils.util import set_seed

arg_mapping = [
    # (key in console, (key in config, type, default value))
    ('exp_name', ('exp_name', str, None)),
    ('type', ('model/type', str, None)), # key in config can be hierarchical
    ('backbone', ('model/backbone', str, None)),
    ('split', ('train_split', str, None)), 
    ('grasp_data', ('data/grasp_data', str, None)), 
    ('iter', ('max_iter', int, None)), 
    ('camera', ('camera', str, None)), 
    ('dist_joint', ('model/dist_joint', int, None)), 
    ('frac', ('data/fraction', int, None)), 
    ('scene_frac', ('data/scene_fraction', int, None)), 
    ('diff_pred', ('model/diffusion/scheduler/prediction_type', str, None)), 
    ('yaml', ('yaml', str, os.path.join('configs', 'network', 'train.yaml'))), 
]

def main():
    # process config, seed, logger, device
    parser = argparse.ArgumentParser()
    add_argparse(parser, arg_mapping)
    args = parser.parse_args()
    config = load_config(args.yaml, arg_mapping, args)
    pprint(config)
    set_seed(config.seed)
    logger = Logger(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # loading data, val loader can have multiple splits
    train_dataset = GraspNetDataset(config, config.train_split, is_train=True)
    val_datasets = [GraspNetDataset(config, split, is_train=False) for split in config.val_split]
    train_loader = Loader(DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True, num_workers=config.num_workers, shuffle=True, collate_fn=minkowski_collate_fn))
    val_loader = [Loader(DataLoader(dataset, batch_size=config.batch_size, drop_last=True, num_workers=config.num_workers, shuffle=True, collate_fn=minkowski_collate_fn)) for dataset in val_datasets]

    # model and optimizer
    config.model['voxel_size'] = config.data.voxel_size
    model = get_model(config.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, config.max_iter, eta_min=config.lr_min)

    # load ckpt if exists
    if config.ckpt is not None:
        ckpt = torch.load(config.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        cur_iter = ckpt['iter']
        for _ in range(cur_iter):
            scheduler.step()
        print(f'loaded ckpt from {config.ckpt}')
    else:
        cur_iter = 0
    
    # training
    model.to(device)
    model.train()
    
    for it in trange(cur_iter, config.max_iter):
        optimizer.zero_grad()
        data = train_loader.get()
        data = {k: v.to(device) for k, v in data.items()}
        loss, result_dict = model(data)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    p.grad.zero_()
                # if p.grad.norm().item() > config.grad_clip * sqrt(p.grad.numel()):
                    # p.grad *= config.grad_clip * sqrt(p.grad.numel()) / p.grad.norm().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        # print({k: v.mean().item() for k, v in result_dict.items()})
        if it % config.log_every == 0:
            logger.log({k: v.mean().item() for k, v in result_dict.items()}, 'train', it)
        
        if (it + 1) % config.save_every == 0:
            logger.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), iter=it+1), it+1)

        if it % config.val_every == 0:
            with torch.no_grad():
                model.eval()
                for split, loader in zip(config.val_split, val_loader):
                    result_dicts = []
                    for _ in range(config.val_num):
                        data = loader.get()
                        data = {k: v.to(device) for k, v in data.items()}
                        loss, result_dict = model(data)
                        result_dicts.append(result_dict)
                    logger.log({k: torch.cat([(dic[k] if len(dic[k].shape) else dic[k][None]) for dic in result_dicts]).mean() for k in result_dicts[0].keys()}, split, it)
                model.train()
    
if __name__ == '__main__':
    main()