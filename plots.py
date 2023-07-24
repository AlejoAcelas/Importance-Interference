# %%

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from typing import Optional, Union, List, Tuple, Callable, Any, Literal
from jaxtyping import Float, Int, Bool

import numpy as np
import einops

from tqdm.notebook import trange

import time
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib
import plotly_utils

importlib.reload(plotly_utils)
from plotly_utils import *
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
