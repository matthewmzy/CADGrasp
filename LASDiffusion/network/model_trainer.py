import copy
import os, sys
sys.path.append('.')
from LASDiffusion.utils.utils import set_requires_grad
from torch.utils.data import DataLoader
from LASDiffusion.network.model_utils import EMA
from LASDiffusion.network.data_loader import IBS_Dataset
from pathlib import Path
from torch.optim import AdamW,Adam
from LASDiffusion.utils.utils import update_moving_average
from pytorch_lightning import LightningModule
from LASDiffusion.network.model import OccupancyDiffusion
import torch.nn as nn

import random


class DiffusionModel(LightningModule):
    def __init__(
        self,
        ibs_path: str = "",
        scene_pc_path:str = "",
        ibs_load_per_scene:int = 64,
        results_folder: str = './results',
        model_mode: str = "diffusion",
        image_size: int = 32,
        base_channels: int = 32,
        lr: float = 2e-4,
        batch_size: int = 8,
        attention_resolutions: str = "16,8",
        optimizier: str = "adam",
        with_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        ema_rate: float = 0.999,
        verbose: bool = False,
        save_every_epoch: int = 1,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        use_sketch_condition: bool = False,
        use_text_condition: bool = True,
        noise_schedule: str = "linear",
        debug: bool = False,
        image_feature_drop_out: float = 0.1,
        view_information_ratio: float = 0.5,
        data_augmentation: bool = False,
        kernel_size: float = 2.0,
        vit_global: bool = False,
        vit_local: bool = True,
        split_dataset: bool = False,
        elevation_zero: bool = False,
        detail_view: bool = False,
        zero_num_workers=False,
        scaling=1,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.results_folder = Path(results_folder)
        self.model = OccupancyDiffusion(model_mode=model_mode,
                                        image_size=image_size, base_channels=base_channels,
                                        attention_resolutions=attention_resolutions,
                                        with_attention=with_attention,
                                        kernel_size=kernel_size,
                                        dropout=dropout,
                                        use_sketch_condition=use_sketch_condition,
                                        use_text_condition=use_text_condition,
                                        num_heads=num_heads,
                                        noise_schedule=noise_schedule,
                                        vit_global=vit_global,
                                        vit_local=vit_local,
                                        verbose=verbose,
                                        )

        self.ibs_path = ibs_path
        self.scene_pc_path = scene_pc_path
        self.ibs_load_per_scene = ibs_load_per_scene
        self.view_information_ratio = view_information_ratio
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.data_augmentation = data_augmentation
        self.with_attention = with_attention
        self.save_every_epoch = save_every_epoch
        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.use_sketch_condition = use_sketch_condition
        self.use_text_condition = use_text_condition
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)
        self.image_feature_drop_out = image_feature_drop_out
        self.scaling = scaling
        self.vit_global = vit_global
        self.vit_local = vit_local
        self.split_dataset = split_dataset
        self.elevation_zero = elevation_zero
        self.detail_view = detail_view
        self.optimizier = optimizier
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)
        if zero_num_workers:
            self.num_workers = 0
        elif debug:
            self.num_workers = 1
        else:
            self.num_workers = os.cpu_count()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def configure_optimizers(self):
        if self.optimizier == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizier == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        return [optimizer]

    def train_dataloader(self):
        _dataset = IBS_Dataset(path=self.ibs_path,
                               scene_path=self.scene_pc_path,
                               ibs_load_per_scene=self.ibs_load_per_scene,
                               split="train",
                               scaling=self.scaling)
        dataloader = DataLoader(_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        return dataloader
    
    def test_dataloader(self):
        _dataset = IBS_Dataset(path=self.ibs_path,
                               scene_path=self.scene_pc_path,
                               ibs_load_per_scene=self.ibs_load_per_scene,
                               split="test",scaling=self.scaling)
        dataloader = DataLoader(_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        return dataloader

    def training_step(self, batch, batch_idx):
        ibs_occupancy = batch["ibs"].reshape(-1,2,40,40,40).to('cuda')             # (B, 2, 40, 40, 40)
        scene_occupancy = batch["scene"].reshape(-1,1,40,40,40).to('cuda')         # (B, 1, 40, 40, 40)
        loss = self.model.training_loss(ibs_occupancy, scene_occupancy).mean()
        self.log("loss", loss.clone().detach().item(), prog_bar=True)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_val)
        opt.step()

        self.update_EMA()

    def on_train_epoch_end(self):
        self.log("current_epoch", self.current_epoch)
        return super().on_train_epoch_end()
