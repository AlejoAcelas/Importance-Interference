### Imports

import os; # os.environ['ACCELERATE_DISABLE_RICH'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional, Union, List, Tuple, Callable, Any

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
 
class Model(nn.Module):
  def __init__(self, 
               config, 
               feature_probability: Optional[torch.Tensor] = None,
               importance: Optional[torch.Tensor] = None,               
               device='cuda'):
    super().__init__()
    self.config = config

    if feature_probability is None:
      feature_probability = torch.ones(())
    self.feature_probability = feature_probability.to(device)
    if importance is None:
      importance = torch.ones(())
    self.importance = importance.to(device)

  ### Batch generators
  
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

    labels = torch.multinomial(
      einops.rearrange(batch, 'b i f -> (b i) f'),
      num_samples=1
    ).squeeze(-1)
    labels = einops.rearrange(labels, '(b i) -> b i', b=n_batch)
    return batch, labels

  def generate_batch_one_hot_noiseless(self, *_, **kwargs):
    n_chunks = kwargs.get('n_chunks', 1)
    feat = torch.eye(self.config.n_features, device=self.W.device)
    labels = torch.arange(self.config.n_features, device=self.W.device)
    batch = einops.repeat(feat, 'f1 f2 -> (chunk f1) instance f2', chunk=n_chunks, instance=self.config.n_instances)
    labels = einops.repeat(labels, 'f -> (chunk f) instance', chunk=n_chunks, instance=self.config.n_instances)
    return batch, labels

  ## Loss functions

  def mse_loss(self, out, labels, per_feature=False):
    error = (self.importance*(labels.abs() - out))**2
    if per_feature:
      return einops.reduce(error, 'b ... f -> ... f', 'mean')
    else:
      return einops.reduce(error, 'b ... f -> ...', 'mean')
    
  def mse_loss_unweighted(self, out, labels, per_feature=False):
    error = (labels.abs() - out)**2
    if per_feature:
      return einops.reduce(error, 'b ... f -> ... f', 'mean')
    else:
      return einops.reduce(error, 'b ... f -> ...', 'mean')

  def cross_entropy_loss(self, out, labels):
    loss = F.cross_entropy(out.transpose(-1, -2), labels, weight=self.importance, reduction='none')
    return loss.mean(0) # shape [n_instances]
        
  def cross_entropy_loss_unweighted(self, out, labels, per_feature=False):
    loss = F.cross_entropy(
      einops.rearrange(out, 'b ... f -> b f ...'), 
      labels,
      reduction='none')
    if per_feature:
      loss_per_feat = torch.zeros_like(out)
      loss_per_feat.scatter_add_(-1, labels.unsqueeze(-1), loss.unsqueeze(-1)).squeeze(-1)
      return loss_per_feat
    else:
      return einops.reduce(loss, 'b ... -> ...', 'mean')
    
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
    hidden = torch.einsum("b...f,...fh->b...h", features, self.W)
    out = torch.einsum("b...h,...fh->b...f", hidden, self.W)
    out = out + self.b_final
    out = F.relu(out)
    return out

  def run_with_noise(self, features, noise_std=0.1):
    hidden = torch.einsum("bif,ifh->bih", features, self.W)
    
    if isinstance(noise_std, torch.Tensor):
      noise = torch.randn((len(noise_std), *hidden.shape), device=features.device) * noise_std[:, None, None, None]
      hidden_with_noise = hidden + noise
    elif isinstance(noise_std, float):
      hidden_with_noise = hidden + torch.randn_like(hidden) * noise_std
    else:
      raise ValueError(f'noise_std must be float or torch.Tensor, but is {type(noise_std)}')

    out = torch.einsum("...h,...fh->...f", hidden_with_noise, self.W)
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
             hooks=[]):
  cfg = model.config

  opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

  start = time.time()
  with trange(steps) as t:
    for step in t:
      step_lr = lr * lr_scale(step, steps)
      for group in opt.param_groups:
        group['lr'] = step_lr
      opt.zero_grad(set_to_none=True)
      batch, labels = batch_fn(model, n_batch)
      out = model(batch)
      loss = loss_fn(model, out, labels).sum()
      loss.backward()
      opt.step()
    
      if step % print_freq == 0 or (step + 1 == steps):
        t.set_postfix(
            loss=loss.item() / model.config.n_instances,
            lr=step_lr,
        )