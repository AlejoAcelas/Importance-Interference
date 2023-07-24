### Imports

import os; # os.environ['ACCELERATE_DISABLE_RICH'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from typing import Optional, Union, List, Tuple, Callable, Any
from jaxtyping import Float, Int, Bool


from dataclasses import dataclass, replace
import numpy as np
import einops

from tqdm.notebook import trange

import time
import pandas as pd
from functools import reduce

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib
import plotly_utils

importlib.reload(plotly_utils)
from plotly_utils import imshow, line, hist, scatter

import matplotlib.pyplot as plt

if torch.cuda.is_available():
  DEVICE = 'cuda'
else:
  DEVICE = 'cpu'

### Basic initialization

@dataclass
class Config:
  n_features: int
  n_hidden: int
  n_instances: Tuple[int]
  device: str = DEVICE
 
class Model(nn.Module):
  def __init__(self, 
               config, 
               feature_probability: Optional[torch.Tensor] = None,
               importance: Optional[torch.Tensor] = None,               
               device=DEVICE):
    super().__init__()
    self.config = config
    self.device = config.device

    if feature_probability is None:
      feature_probability = torch.ones(self.config.n_instances, 1)
    self.feature_probability = nn.Parameter(feature_probability.to(device), requires_grad=False)

    if importance is None:
      importance = torch.ones(self.config.n_features)
    self.importance = nn.Parameter(importance.to(device), requires_grad=False)

  ### Batch generators
  # I assume that batch is of shape [n_batch, n_instances, n_features] or [n_batch, extra_dim, n_instances, n_features]

  def generate_batch_rand(self, n_batch):
    feat = torch.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device)
    batch = torch.where(
        torch.rand((n_batch, self.config.n_instances, self.config.n_features), device=self.W.device) <= self.feature_probability,
        feat,
        torch.zeros((), device=self.W.device),
    )
    return batch, batch.clone().detach()
  
  def generate_batch_one_hot(self, n_batch, feature_probability=None):
    if feature_probability is None:
      feature_probability = self.feature_probability
    # I reserve space for an extra custom defined feature
    all_features = torch.eye(self.config.n_features-1, device=self.W.device)
    feat = einops.repeat(all_features, 'f1 f2 -> batch instance f1 f2', batch=n_batch, instance=self.config.n_instances)
    sparsitiy_filter = (
      torch.rand((n_batch, self.config.n_instances, self.config.n_features-1), device=self.W.device) <= self.feature_probability
    ).float()
    
    batch = (feat * sparsitiy_filter[:, :, None, :]).sum(-1) # shape [n_batch, n_instances, n_features]
    batch = torch.cat([
            batch,
            (batch.sum(-1, keepdim=True) == 0).float() # Feature that corresponds to all other features being zero 
          ], dim=-1)

    target = torch.multinomial(
      einops.rearrange(batch, 'b ... i f -> (b ... i) f'),
      num_samples=1
    ).squeeze(-1)

    if batch.ndim == 3:
      target = einops.rearrange(target, '(b i) -> b i', b=n_batch, i=self.config.n_instances)
    else:
      target = einops.rearrange(target, '(b a i) -> b a i', b=n_batch, i=self.config.n_instances)
    return batch, target

  def generate_batch_one_hot_noiseless(self, n_batch):
    if n_batch % self.config.n_features != 0:
      print('Warning: n_batch should be a multiple of n_features')
    n_chunks = n_batch // self.config.n_features
    feat = torch.eye(self.config.n_features, device=self.W.device)
    target = torch.arange(self.config.n_features, device=self.W.device)
    batch = einops.repeat(feat, 'f1 f2 -> (chunk f1) instance f2', chunk=n_chunks, instance=self.config.n_instances)
    target = einops.repeat(target, 'f -> (chunk f) instance', chunk=n_chunks, instance=self.config.n_instances)
    return batch, target

  ## Loss functions

  def mse_loss(self, out, target, per_feature=False):
    error = self.importance*((target.abs() - out))**2
    if per_feature:
      return error
    else:
      return einops.reduce(error, 'b ... i f -> ... i', 'mean')
    
  def mse_loss_unweighted(self, out, target, per_feature=False):
    error = (target.abs() - out)**2
    if per_feature:
      return error
    else:
      return einops.reduce(error, 'b ... i f -> ... i', 'mean')

  def cross_entropy_loss(self, out, target, per_feature=False):
    loss = F.cross_entropy(
    einops.rearrange(out, 'b ... i f -> b ... f i'), 
    target,
    weight=self.importance.squeeze(),
    reduction='none')
    if per_feature:
      loss_per_feat = torch.zeros_like(out)
      loss_per_feat.scatter_add_(-1, target.unsqueeze(-1), loss.unsqueeze(-1)).squeeze(-1)
      return loss_per_feat
    else:
      return einops.reduce(loss, 'b ... i -> ... i', 'mean') # Mean over batch dim (there's no feature dim)
          
  def cross_entropy_loss_unweighted(self, out, target, per_feature=False):
    loss = F.cross_entropy(
      einops.rearrange(out, 'b ... i f -> b ... f i'), 
      target,
      reduction='none')
    if per_feature:
      loss_per_feat = torch.zeros_like(out)
      loss_per_feat.scatter_add_(-1, target.unsqueeze(-1), loss.unsqueeze(-1)).squeeze(-1)
      return loss_per_feat
    else:
      return einops.reduce(loss, 'b ... i -> ... i', 'mean')

  def accuracy(self, out, target, per_feature=False):
      if target.ndim == out.ndim:
        target = target.argmax(-1)
      assert target.ndim == out.ndim - 1
      
      acc = (out.argmax(-1) == target).float() # shape [n_batch, ..., n_instances]
      if per_feature:
        acc_by_feature = torch.zeros_like(out) # shape [n_batch, ..., n_instances, n_features]
        acc_by_feature.scatter_add_(-1, target[..., None], acc[..., None])
        return acc_by_feature # shape [n_batch, ..., n_instances, n_features]
      else:
        return acc.mean()
    
def batch_together(batch_fn, reps, *args, **kwargs):
  batch_out = [batch_fn(*args, **kwargs) for _ in range(reps)]
  batch, target = zip(*batch_out)
  return torch.stack(batch, dim=1), torch.stack(target, dim=1)

### 1-layer MLP model

class BasicMLP(Model):
  def __init__(self,
               config,
               feature_probability: Optional[torch.Tensor] = None,
               importance: Optional[torch.Tensor] = None,
               device='cuda'):
    super().__init__(config, feature_probability, importance, device)

    self.W = nn.Parameter(torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device))
    nn.init.xavier_normal_(self.W)
    self.b_final = nn.Parameter(torch.zeros((config.n_instances, config.n_features), device=device))

  def forward(self, features):
    # features: [..., instance, n_features]
    # W: [instance, n_features, n_hidden]
    hidden = torch.einsum("b...if,ifh->b...ih", features, self.W)
    out = torch.einsum("b...ih,ifh->b...if", hidden, self.W)
    out = out + self.b_final
    out = F.relu(out)
    return out

  def run_with_noise(self,
                     features: Float[Tensor, 'batch noise inst feat'],
                     noise_std: Float[Tensor, 'noise']):
    hidden = torch.einsum("b...if,ifh->b...ih", features, self.W)
    noise = torch.randn_like(hidden) * noise_std[:, None, None]
    hidden = hidden + noise
    out = torch.einsum("b...ih,ifh->b...if", hidden, self.W)
    out = out + self.b_final
    out = F.relu(out)
    return out

### Training

def linear_lr(step, steps):
  return (1 - (step / steps))

def constant_lr(*_):
  return 1.0

def cosine_decay_lr(step, steps):
  return np.cos(0.5 * np.pi * step / (steps - 1))

def optimize(model, 
             render=False, 
             n_batch=1024,
             steps=10_000,
             batch_fn=Model.generate_batch_rand,
             loss_fn=Model.mse_loss,
             print_freq=100,
             lr=1e-3,
             lr_scale=constant_lr,
             ):
  cfg = model.config

  opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

  start = time.time()
  with trange(steps) as t:
    for step in t:
      step_lr = lr * lr_scale(step, steps)
      for group in opt.param_groups:
        group['lr'] = step_lr
      opt.zero_grad(set_to_none=True)
      batch, target = batch_fn(model, n_batch)
      out = model(batch)
      loss = loss_fn(model, out, target).sum()
      loss.backward()
      opt.step()
    
      if step % print_freq == 0 or (step + 1 == steps):
        t.set_postfix(
            loss=loss.item() / model.config.n_instances,
            lr=step_lr,
        )
