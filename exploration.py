# %%

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

# %% Model 

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

  def generate_batch_rand(self, n_batch):
    feat = torch.rand((n_batch, *self.config.n_instances, self.config.n_features), device=self.W.device)
    batch = torch.where(
        torch.rand((n_batch, *self.config.n_instances, self.config.n_features), device=self.W.device) <= self.feature_probability,
        feat,
        torch.zeros((), device=self.W.device),
    )
    return batch, batch.clone().detach()
  
  def generate_batch_one_hot(self, n_batch):
    assert len(self.config.n_instances) == 1, "Only one instance dim supported for one-hot encoding"
    batch, labels = [], []
    ammended_feature_prob = einops.repeat(self.feature_probability, 'instance 1 -> instance 1 feature', feature=self.config.n_features).clone().detach()
    ammended_feature_prob[..., 0] = 1.0 # always include the first feature to avoid batches with no label
    
    # ToDo: vectorize
    for _ in range(n_batch):
      # n_instances x n_features x n_features. 
      feat = torch.eye(self.config.n_features, device=self.W.device).repeat(*self.config.n_instances, 1, 1)
      batch_expanded = torch.where(
          torch.rand_like(feat) <= ammended_feature_prob,
          feat,
          torch.zeros((), device=self.W.device),
      )
      batch_i = batch_expanded.sum(-1)
      labels_i = torch.multinomial(batch_i, 1).squeeze(-1) # n_instances
      batch.append(batch_i) # n_instances x n_features
      labels.append(labels_i) # n_instances
  
    batch, labels = torch.stack(batch), torch.stack(labels)
    return batch, labels


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
    return loss.mean(0) # shape [*n_instances]
        
  def cross_entropy_loss_unweighted(self, out, labels, per_feature=False):
    loss = F.cross_entropy(out.transpose(-1, -2), labels, reduction='none')
    if per_feature:
      loss_per_feat = torch.zeros_like(out)
      loss_per_feat.scatter_add_(-1, labels[:, :, None], loss[:, :, None]).squeeze(-1)
      return loss_per_feat.mean(0)
    else:
      return einops.reduce(loss, 'b ... -> ...', 'mean')

# %% Basic MLP

class BasicMLP(Model):
  def __init__(self,
               config,
               feature_probability: Optional[torch.Tensor] = None,
               importance: Optional[torch.Tensor] = None,
               device='cuda'):
    super().__init__(config, feature_probability, importance, device)

    self.W = nn.Parameter(torch.empty((*config.n_instances, config.n_features, config.n_hidden), device=device))
    nn.init.xavier_normal_(self.W)
    self.b_final = nn.Parameter(torch.zeros((*config.n_instances, config.n_features), device=device))

  def forward(self, features):
    # features: [..., instance, n_features]
    # W: [instance, n_features, n_hidden]
    hidden = torch.einsum("b...f,...fh->b...h", features, self.W)
    out = torch.einsum("b...h,...fh->b...f", hidden, self.W)
    out = out + self.b_final
    out = F.relu(out)
    return out

  def run_with_noise(self, features, noise_std=0.1):
    hidden = torch.einsum("b...f,...fh->b...h", features, self.W)
    hidden = hidden + torch.randn_like(hidden) * noise_std
    out = torch.einsum("b...h,...fh->b...f", hidden, self.W)
    out = out + self.b_final
    out = F.relu(out)
    return out

# %%
class TestModel(BasicMLP):
  config = Config(n_features=2, n_hidden=5, n_instances=(8,))
  model = Model(config, device=DEVICE)

  def __init__(self, feature_probability=None, importance=None):
    super().__init__(self.config,
                     device=DEVICE,
                     feature_probability=feature_probability,
                     importance=importance)

test_model = TestModel(feature_probability=torch.tensor(8*[0.5])[:, None])
batch, labels = test_model.generate_batch_one_hot(100)
batch.shape
# %% Optimize

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
            loss=loss.item() / reduce(lambda x, y: x*y, cfg.n_instances),
            lr=step_lr,
        )

# %%

config = Config(
    n_features = 20,
    n_hidden = 2,
    n_instances = (10,),
)

model = BasicMLP(
    config=config,
    device=DEVICE,
    # Exponential feature importance curve from 1 to 1/100
    importance = (0.80**torch.arange(config.n_features))[None, :],
    # importance = torch.ones(config.n_features)[None, :],
    # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
    feature_probability = (20 ** -torch.linspace(0, 1, config.n_instances[-1]))[:, None]
)

optimize(model,
         steps=10_000,
         batch_fn=Model.generate_batch_one_hot,
         loss_fn=Model.cross_entropy_loss,
         lr=1e-3,
         )

# %%

importance = (0.95**torch.arange(config.n_features))[None, :]
W, b = model.W.detach(), model.b_final.detach()
out = F.softmax(W @ W.transpose(1, 2) + model.b_final[:, None], dim=-1)
# imshow(out, animation_frame=0)
x, y = W[0].unbind(-1)
scatter(x, y, color=importance[0].tolist())
# scatter(x, y, color=[f'feature {i}' for i in range(config.n_features)])
# line(model.b_final[0])

# %%

cfg = model.config
W = model.W.detach()
W_norm = W / (1e-5 + torch.linalg.norm(W, 2, dim=-1, keepdim=True))

interference = torch.einsum('ifh,igh->ifg', W_norm, W)
interference[:, torch.arange(cfg.n_features), torch.arange(cfg.n_features)] = 0

polysemanticity = torch.linalg.norm(interference, dim=-1).cpu()
# net_interference = (interference**2 * model.feature_probability[:, None, :]).sum(-1).cpu()
# norms = torch.linalg.norm(W, 2, dim=-1).cpu()
# WtW = torch.einsum('sih,soh->sio', W, W).cpu()


# %% 
n_batch = config.n_features
batch, labels = model.generate_batch_one_hot(n_batch)
noise_std = [0.1, 0.5, 1.0, 2.0, 5.0]
out = torch.stack([model.run_with_noise(batch, noise_std=std) for std in noise_std])
imshow(out, facet_col=0, facet_labels=[f'noise std = {std}' for std in noise_std])


# %%
n_batch = 1024

# batch, labels = model.generate_batch_rand(n_batch)
# loss_per_feat = model.mse_loss_unweighted(model.run_with_noise(batch), labels, per_feature=True)
# loss_per_feat_base = model.mse_loss_unweighted(model(batch), labels, per_feature=True)
# imshow(loss_per_feat - loss_per_feat_base)
# imshow(loss_per_feat_base)
# imshow(loss_per_feat)

batch, labels = model.generate_batch_one_hot(n_batch)
loss_per_feat = model.cross_entropy_loss_unweighted(model.run_with_noise(batch, noise_std=1), labels, per_feature=True)
loss_per_feat_base = model.cross_entropy_loss_unweighted(model(batch), labels, per_feature=True)
imshow(loss_per_feat - loss_per_feat_base)


# %%

line((loss_per_feat - loss_per_feat_base).mean(0), labels={'value': 'loss difference', 'index': 'feature'})
# l = [x for x in (loss_per_feat - loss_per_feat_base)]
# line(l, labels={'value': 'loss difference', 'index': 'feature'},
#      names=[f'Sparsity: {s.item(): .3f}' for s in model.feature_probability[:,0]])

# %% Plot functions

def plot_intro_diagram(model):
  from matplotlib import colors  as mcolors
  from matplotlib import collections  as mc
  cfg = model.config
  WA = model.W.detach()
  N = len(WA[:,0])
  sel = range(*config.n_instances) # can be used to highlight specific sparsity levels
  plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(model.importance[0].cpu().numpy()))
  plt.rcParams['figure.dpi'] = 200
  fig, axs = plt.subplots(1,len(sel), figsize=(2*len(sel),2))
  for i, ax in zip(sel, axs):
      W = WA[i].cpu().detach().numpy()
      colors = [mcolors.to_rgba(c)
            for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
      ax.scatter(W[:,0], W[:,1], c=colors[0:len(W[:,0])])
      ax.set_aspect('equal')
      ax.add_collection(mc.LineCollection(np.stack((np.zeros_like(W),W), axis=1), colors=colors))
      
      z = 1.5
      ax.set_facecolor('#FCFBF8')
      ax.set_xlim((-z,z))
      ax.set_ylim((-z,z))
      ax.tick_params(left = True, right = False , labelleft = False ,
                  labelbottom = False, bottom = True)
      for spine in ['top', 'right']:
          ax.spines[spine].set_visible(False)
      for spine in ['bottom','left']:
          ax.spines[spine].set_position('center')
  plt.show()

def render_features(model, which=np.s_[:]):
  cfg = model.config
  W = model.W.detach()
  W_norm = W / (1e-5 + torch.linalg.norm(W, 2, dim=-1, keepdim=True))

  interference = torch.einsum('ifh,igh->ifg', W_norm, W)
  interference[:, torch.arange(cfg.n_features), torch.arange(cfg.n_features)] = 0

  polysemanticity = torch.linalg.norm(interference, dim=-1).cpu()
  net_interference = (interference**2 * model.feature_probability[:, None, :]).sum(-1).cpu()
  norms = torch.linalg.norm(W, 2, dim=-1).cpu()

  WtW = torch.einsum('sih,soh->sio', W, W).cpu()

  # width = weights[0].cpu()
  # x = torch.cumsum(width+0.1, 0) - width[0]
  x = torch.arange(cfg.n_features)
  width = 0.9

  which_instances = np.arange(*cfg.n_instances)[which]
  fig = make_subplots(rows=len(which_instances),
                      cols=2,
                      shared_xaxes=True,
                      vertical_spacing=0.02,
                      horizontal_spacing=0.1)
  for (row, inst) in enumerate(which_instances):
    fig.add_trace(
        go.Bar(x=x, 
              y=norms[inst],
              marker=dict(
                  color=polysemanticity[inst],
                  cmin=0,
                  cmax=1
              ),
              width=width,
        ),
        row=1+row, col=1
    )
    data = WtW[inst].numpy()
    fig.add_trace(
        go.Image(
            z=plt.cm.coolwarm((1 + data)/2, bytes=True),
            colormodel='rgba256',
            customdata=data,
            hovertemplate='''\
In: %{x}<br>
Out: %{y}<br>
Weight: %{customdata:0.2f}
'''            
        ),
        row=1+row, col=2
    )

  fig.add_vline(
    x=(x[cfg.n_hidden-1]+x[cfg.n_hidden])/2, 
    line=dict(width=0.5),
    col=1,
  )
    
  # fig.update_traces(marker_size=1)
  fig.update_layout(showlegend=False, 
                    width=600,
                    height=100*len(which_instances),
                    margin=dict(t=0, b=0))
  fig.update_xaxes(visible=False)
  fig.update_yaxes(visible=False)
  return fig

# %%

# plot_intro_diagram(model)
render_features(model, np.s_[:1])
# fig = render_features(model, np.s_[::2])
# fig.update_layout()
# %%
