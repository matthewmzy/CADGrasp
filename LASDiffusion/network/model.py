import torch
import torch.nn.functional as F
from tqdm import tqdm
from LASDiffusion.network.model_utils import *
from LASDiffusion.network.unet import UNetModel
from einops import rearrange, repeat
from random import random
from functools import partial
from torch import nn
from torch.special import expm1
from termcolor import cprint

TRUNCATED_TIME1 = 0.7
TRUNCATED_TIME2 = 0.5


class OccupancyDiffusion(nn.Module):
    def __init__(
            self,
            model_mode: str = "diffusion",
            image_size: int = 64,
            base_channels: int = 128,
            attention_resolutions: str = "16,8",
            with_attention: bool = False,
            num_heads: int = 4,
            dropout: float = 0.0,
            verbose: bool = False,
            use_sketch_condition: bool = True,
            use_text_condition: bool = True,
            eps: float = 1e-6,
            noise_schedule: str = "linear",
            kernel_size: float = 1.0,
            vit_global: bool = False,
            vit_local: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.model_mode = model_mode
        if image_size == 8:
            channel_mult = (1, 4, 8)
        elif image_size == 32:
            channel_mult = (1, 2, 4, 8)
        elif image_size == 64:
            channel_mult = (1, 2, 4, 8, 8)
        elif image_size == 40:
            channel_mult = (1, 2, 4, 8)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
        self.eps = eps
        self.verbose = verbose
        self.use_sketch_condition = use_sketch_condition
        self.use_text_condition = use_text_condition
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        self.denoise_fn = UNetModel(
            model_mode=self.model_mode,
            image_size=image_size,
            base_channels=base_channels,
            dim_mults=channel_mult, dropout=dropout,
            world_dims=3,
            num_heads=num_heads,
            attention_resolutions=tuple(attention_ds), with_attention=with_attention,
            verbose=verbose)
        self.vit_global = vit_global
        self.vit_local = vit_local

    @property
    def device(self):
        return next(self.denoise_fn.parameters()).device

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def training_loss(
            # self, img, img_features, text_feature, projection_matrix, kernel_size=None, *args, **kwargs
            self, ibs, scene
            ):
        """
        ibs: (B,2,40,40,40)
        scene: (B,1,40,40,40)
        """
        batch = ibs.shape[0]
        assert ibs.shape[0] == scene.shape[0], "Batch size mismatch in model.py"
        if self.model_mode == "diffusion":
            times = torch.zeros(
                (batch,), device=self.device).float().uniform_(0, 1)
            noise = torch.randn_like(ibs)

            noise_level = self.log_snr(times)
            padded_noise_level = right_pad_dims_to(ibs, noise_level)
            alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
            noised_ibs = alpha * ibs + sigma * noise
            self_cond = None
            if random() < 0.5:
                with torch.no_grad():
                    self_cond = self.denoise_fn(noised_ibs, noise_level, scene).detach_()
            pred = self.denoise_fn(noised_ibs, noise_level, scene, x_self_cond=self_cond)
        elif self.model_mode == "classification":
            pred = self.denoise_fn(ibs, None, scene)
        cont_mask = ibs[:,1]>0
        cont_loss = F.mse_loss(pred[:,1][cont_mask], ibs[:,1][cont_mask])
        thco_mask = ibs[:,1]>1.5
        thco_loss = F.mse_loss(pred[:,1][thco_mask], ibs[:,1][thco_mask])
        return F.mse_loss(pred, ibs) * 0.4 + cont_loss * 0.3 + thco_loss * 0.3

    @torch.no_grad()
    def sample_classifier(self, batch_size=16, scene_voxel=None,
                             steps=50, truncated_index: float = 0.0, verbose: bool = True):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        
        batch, device = shape[0], self.device

        img = self.denoise_fn(img, None, scene_voxel, None)        # CHANGE

        return img
    
    @torch.no_grad()
    def sample_based_on_scene(self, batch_size=16, scene_voxel=None,
                              steps=50, truncated_index: float = 0.0, verbose: bool = True):
        image_size = self.image_size
        shape = (batch_size, 2, image_size, image_size, image_size)
        
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)

        img = torch.randn(shape, device=device)
        x_start = None

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        for time, time_next in _iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)
            x_start = self.denoise_fn(img, noise_cond, scene_voxel.to(device=device), None)        # CHANGE

            if time[0] < 0.7:
                x_start[:,0] = torch.where(x_start[:,0]<0, -1.0, 1.0)
            # elif time[0] < 0.7:
            #     x_start[:,0] = torch.where(x_start[:,0]<-1., -1.0, torch.where(x_start[:,0]>1.0, 1.0, x_start[:,0]))
            if time[0] < 0.5:
                x_start[:,1] = torch.where(x_start[:,1]<0.5, -1.0, torch.where(x_start[:,1]>1.5, 2.0, 1.0))
            elif time[0] < 0.7:
                x_start[:,1] = torch.where(x_start[:,1]<-1., -1.0, torch.where(x_start[:,1]>2.0, 2.0, x_start[:,1]))

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )
            img = mean + torch.sqrt(variance) * noise

        return img

