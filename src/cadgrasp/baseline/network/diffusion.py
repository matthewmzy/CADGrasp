import os
import numpy as np
import torch
from torch import nn
import diffusers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from cadgrasp.baseline.network.mlp import MLP
import math
from einops import rearrange, repeat

def jacobian_matrix(f, z):
    """Calculates the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    jacobian = torch.zeros(
        (*f.shape, z.shape[-1]), device=f.device)
    for i in range(f.shape[-1]):
        jacobian[..., i, :] = torch.autograd.grad(
            f[..., i].sum(), z, retain_graph=(i != f.shape[-1]-1), allow_unused=True)[0]
    return jacobian.contiguous()

def approx_jacobian_trace(f, z):
    e = torch.normal(mean=0, std=1, size=f.shape,
                     device=f.device, dtype=f.dtype)
    grad = torch.autograd.grad(f, z, grad_outputs=e)[0]
    return torch.einsum('nka,nka->nk', e, grad)

def jacobian_trace(log_prob_type, dx, dy):
    if log_prob_type == 'accurate_cont':
        # time consuming
        jacobian_mat = jacobian_matrix(dy, dx)
        return jacobian_mat.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    elif log_prob_type == 'estimate':
        # quick
        return approx_jacobian_trace(dy, dx)
    else:
        return 0

class SinusoidalPosEmb(nn.Module):
    # compute sinusoidal positional embeddings
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor):
        """
            x: torch.Tensor, shape (B,)
            return emb: torch.Tensor, shape (B, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * self.theta
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLPWrapper(MLP):
    def __init__(self, channels, feature_dim, *args, **kwargs):
        self.channels = channels
        input_dim = channels + feature_dim
        super().__init__(input_dim = input_dim,*args, **kwargs)
        self.embedding = SinusoidalPosEmb(feature_dim)
    
    def forward(self, x, t, cond):
        t = self.embedding(t)
        return super().forward(torch.cat([x, cond + t], dim = -1))


class GaussianDiffusion1D(nn.Module):
    def __init__(self, model, config, cond_fn=lambda x, t, cond: cond):
        super().__init__()
        self.config = config
        self.model = model
        self.cond_fn = cond_fn
        if config.scheduler_type == 'DDPMScheduler':
            self.scheduler = DDPMScheduler(**config.scheduler)
        else:
            raise NotImplementedError()
        self.timesteps = config.scheduler.num_train_timesteps
        self.inference_timesteps = config.num_inference_timesteps
        self.prediction_type = config.scheduler.prediction_type
        
    def forward(self, *args, **kwargs):
        return self.calculate_loss(*args, **kwargs)
    
    def calculate_loss(self, x, cond):
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)
        noised_x = self.scheduler.add_noise(x, noise, t)
        cond = self.cond_fn(noised_x, t / self.timesteps, cond)
        pred = self.model(noised_x, t / self.timesteps, cond=cond)
        if self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(x, noise, t)
        else:
            raise NotImplementedError()
        
        loss = (pred - target).square().mean()
        
        return loss
    
    def sample(self, cond):
        x = torch.randn(cond.shape[0], self.model.channels, device=cond.device)
        log_prob = (-x.square()/2 - np.log(2 * np.pi)/2).sum(1)
        self.scheduler.set_timesteps(self.inference_timesteps, device=cond.device)

        need_log_prob = self.config.log_prob_type is not None
        last_t = self.timesteps
        with torch.set_grad_enabled(need_log_prob):
            for t in self.scheduler.timesteps:
                dx = torch.zeros_like(x)
                dx.requires_grad_(need_log_prob)
                x += dx
                dt = torch.full((x.shape[0], 1), (last_t - t.item())/self.timesteps, device=x.device, dtype=torch.float)
                last_t = t.item()
                t_pad = torch.full((x.shape[0],), t.item(), device=x.device, dtype=torch.long)
                cond_now = self.cond_fn(x, t_pad / self.timesteps, cond)
                model_output = self.model(x, t_pad / self.timesteps, cond=cond_now)
                alpha_prod = self.scheduler.alphas_cumprod.to(x.device)[t_pad][:, None]
                betas = self.scheduler.betas.to(x.device)[t_pad][:, None]
                if self.prediction_type == 'epsilon':
                    noise = model_output
                elif self.prediction_type == 'v_prediction':
                    noise = model_output * alpha_prod.sqrt() + x * (1-alpha_prod).sqrt()
                score = - 1 / (1-alpha_prod).sqrt() * noise
                beta = betas * self.timesteps
                if self.config.ode:
                    dy = (-0.5 * beta * x - score * beta / 2) * dt 
                else:
                    dy = (-0.5 * beta * x - score * beta) * dt + beta.sqrt() * torch.randn_like(x) * dt.sqrt()
                log_prob -= jacobian_trace(self.config.log_prob_type, dx, -dy / dt) * dt[:, 0]
                x = x - dy
                x = x.detach()
                log_prob = log_prob.detach()
                # x = self.scheduler.step(model_output, t, x).prev_sample
        if not need_log_prob:
            log_prob *= 0
        return x, log_prob
    
    def log_prob(self, x, cond):
        raise NotImplementedError()
