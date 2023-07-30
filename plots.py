# %%

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from typing import Optional, Union, List, Tuple, Callable, Any, Literal
from jaxtyping import Float, Int, Bool

import numpy as np
import einops
from math import ceil

from tqdm.notebook import trange

import time
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib
import my_plotly_utils

importlib.reload(my_plotly_utils)
from my_plotly_utils import *
import matplotlib.pyplot as plt

from model_settings import *

# %%
### 
# For now I'll assume I'm only working with MSE and continuous features

# Import a model for testing

# config = Config(
#     n_features = 20,
#     n_hidden = 2,
#     n_instances = 10,
# )

# # It doesn't save feature probability or importance! Fix this
# model = BasicMLP(config,
#                  importance = (0.75**torch.arange(config.n_features))[None, :],
#                  feature_probability = (20 ** -torch.linspace(0, 1, config.n_instances))[:, None],)
# model.load_state_dict(torch.load('models/MLP_f20_h2_d75.pth'))

# %%

def plot_noise_curves(model: Model, 
                      noise_std: Float[Tensor, 'noise'] = None,
                      model_idx: Union[Literal['avg'], int] = 'avg', 
                      batch_size: int = 500):
    device = model.config.device
    if noise_std is None:
        noise_std = torch.linspace(0, 2, 10)
    noise_std = noise_std.to(device)

    # batch [batch_size, noise, n_instances, n_features]
    batch, target = batch_together(model.generate_batch_rand, reps=len(noise_std), n_batch=batch_size)
    print(batch.shape, target.shape)
    out = model.run_with_noise(batch, noise_std=noise_std)
    loss = model.mse_loss_unweighted(out, target, per_feature=True) # [noise, n_instances, n_features]
    if model_idx == 'avg':
        loss = loss.mean(1)
        fig = line([l for l in loss],
             title=f'Loss per feature with Gaussian noise on hidden space (avg over {model.config.n_instances} models)',
             color_discrete_sequence=px.colors.sequential.Viridis,
             yaxis_title='Loss', xaxis_title='Feature', return_fig=True)
    else:
        loss = loss[:, model_idx]
        fig = line([l for l in loss],
            title=f'Loss per feature with Gaussian noise on hidden space (model {model_idx})',
            color_discrete_sequence=px.colors.sequential.Viridis,
            yaxis_title='Loss', xaxis_title='Feature', return_fig=True)
    
    return fig

# plot_noise_curves(model).show()

# %%

def plot_hidden_2d(model: Model, rows=2, cols=5):
    W = model.W.detach()
    figs = []
    axis_lim = W.abs().max().item()
    feature_names = [f'Feature {i}' for i in range(model.config.n_features)]
    model_names = [f'Model {r+c*rows}' for r in range(rows) for c in range(cols)]
    # Change plot limits 
    for w_i in W:
        x, y = w_i[:, 0], w_i[:, 1]
        fig = scatter(x, y, color=feature_names,
                      color_discrete_map= dict(zip(feature_names, px.colors.sequential.Inferno)),
                      return_fig=True)
        # return fig
        figs.append(fig)
    big_fig = figs_to_subplots(figs, rows=rows, cols=cols, return_fig=True, showlegend=False,
                               subplot_titles=model_names, horizontal_spacing=None, vertical_spacing=0.05,
                               title='Hidden space', shared_xaxes=True, shared_yaxes=True)
    return big_fig


def plot_hidden_2D(model: BasicMLP, rows: Optional[int] = 2, cols: Optional[int] = None):
    W = model.W.detach().cpu().numpy()
    b = model.b_final.detach().cpu().numpy()
    n_instances, n_features, _ = W.shape

    rows = rows
    cols = cols if cols is not None else ceil(n_instances / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows), sharex=True, sharey=True)

    # Define a set of colors for different features
    colors = ['b', 'r', 'g', 'm', 'darkorange', 'y', 'c']
    print(list(zip([f'Feat {f}' for f in range(b.shape[-1])], colors)))

    # Make sure axes is always a 2D array, even when n_instances=1
    if n_instances == 1:
        axes = np.array([[axes]])

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n_instances:
                # Create origin points for each feature vector
                X = np.zeros(n_features)
                Y = np.zeros(n_features)
                U = W[idx, :, 0]
                V = W[idx, :, 1]
                
                for f in range(n_features):
                    axes[i, j].quiver(X[f], Y[f], U[f], V[f], angles='xy', scale_units='xy', scale=1, color=colors[f % len(colors)], alpha=0.9)
                    axes[i, j].scatter(b[idx, f]*U[f], b[idx, f]*V[f], color=colors[f % len(colors)], label=f'Feature {f+1}')
                
                axes[i, j].set_xlim(-1, 1)
                axes[i, j].set_ylim(-1, 1)
                axes[i, j].set_title(f"Instance {idx}")
            else:
                axes[i, j].axis('off')  # hide unused subplots
    # axes[0, 0].legend()

    plt.tight_layout()
    plt.show()


# plot_hidden_2d(model).show()

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

  which_instances = np.arange(cfg.n_instances)[which]
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


def plot_intro_diagram(model):
  from matplotlib import colors  as mcolors
  from matplotlib import collections  as mc
  cfg = model.config
  WA = model.W.detach()
  N = len(WA[:,0])
  sel = range(config.n_instances) # can be used to highlight specific sparsity levels
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

# %%

# import numpy as np

# plot_list = [
#     line(np.arange(10), return_fig=True),
#     line([np.random.rand(20) for _ in range(5)], return_fig=True),
# ]

# figs_to_subplots(plot_list, return_fig=True, 
#                  subplot_titles=['Plot 1', 'Plot 2'],
#                  xaxis='My x-axis', yaxis='My y-axis', shared_yaxes=True)
# %%
