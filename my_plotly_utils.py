import torch as t
from torch import Tensor
from typing import List, Union, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy
from typing import Dict
import pandas as pd
from jaxtyping import Float
import einops

# TODO: Include your own version of circuitvis attention patterns plot

# GENERIC PLOTTING FUNCTIONS

update_layout_set = {"xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor", "showlegend", "xaxis_tickmode", "yaxis_tickmode", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "coloraxis_showscale"}

def imshow(tensor, renderer=None, return_fig=False, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "color_continuous_midpoint" not in kwargs_pre:
        kwargs_pre["color_continuous_midpoint"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        # Weird thing where facet col wrap means labels are in wrong order
        if "facet_col_wrap" in kwargs_pre:
            facet_labels = reorder_list_in_plotly_way(facet_labels, kwargs_pre["facet_col_wrap"])
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    if return_fig:
        return fig
    fig.show(renderer=renderer)


def reorder_list_in_plotly_way(L: list, col_wrap: int):
    '''
    Helper function, because Plotly orders figures in an annoying way when there's column wrap.
    '''
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]
    return L_new


def line(y: Union[t.Tensor, List[t.Tensor]], renderer=None, return_fig=False, **kwargs):
    '''
    Edit to this helper function, allowing it to take args in update_layout (e.g. yaxis_range).
    '''
    # TODO: Receive add line kwarg
    # TODO: Swap the word 'value' for 'y' in the labels when there are multiple lines
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    names = kwargs_pre.pop("names", None)
    if "add_line" in kwargs:
        add_line = kwargs.pop("add_line")
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if "xaxis_tickvals" in kwargs_pre:
        tickvals = kwargs_pre.pop("xaxis_tickvals")
        kwargs_post["xaxis"] = dict(
            tickmode = "array",
            tickvals = kwargs_pre.get("x", np.arange(len(tickvals))),
            ticktext = tickvals
        )
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "x unified"    
    if "use_secondary_yaxis" in kwargs_pre and kwargs_pre["use_secondary_yaxis"]:
        del kwargs_pre["use_secondary_yaxis"]
        if "labels" in kwargs_pre:
            labels: dict = kwargs_pre.pop("labels")
            kwargs_post["yaxis_title_text"] = labels.get("y1", None)
            kwargs_post["yaxis2_title_text"] = labels.get("y2", None)
            kwargs_post["xaxis_title_text"] = labels.get("x", None)
        for k in ["title", "template", "width", "height"]:
            if k in kwargs_pre:
                kwargs_post[k] = kwargs_pre.pop(k)
        fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(**kwargs_post)
        y0 = to_numpy(y[0])
        y1 = to_numpy(y[1])
        x0, x1 = kwargs_pre.pop("x", [np.arange(len(y0)), np.arange(len(y1))])
        name0, name1 = kwargs_pre.pop("names", ["yaxis1", "yaxis2"])
        fig.add_trace(go.Scatter(y=y0, x=x0, name=name0), secondary_y=False)
        fig.add_trace(go.Scatter(y=y1, x=x1, name=name1), secondary_y=True)
    else:
        y = list(map(to_numpy, y)) if isinstance(y, list) and not (isinstance(y[0], int) or isinstance(y[0], float)) else to_numpy(y)
        fig = px.line(y=y, **kwargs_pre).update_layout(**kwargs_post)
        if names is not None:
            fig.for_each_trace(lambda trace: trace.update(name=names.pop(0)))
    
    if return_fig:
        return fig
    fig.show(renderer)
        

def scatter(x, y, renderer=None, return_fig=False, **kwargs):
    x = to_numpy(x)
    y = to_numpy(y)
    add_line = None
    if "add_line" in kwargs:
        add_line = kwargs.pop("add_line")
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "facet_labels" in kwargs_pre:
        facet_labels = kwargs_pre.pop("facet_labels")
    else:
        facet_labels = None
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.scatter(y=y, x=x, **kwargs_pre).update_layout(**kwargs_post)
    if add_line is not None:
        xrange = fig.layout.xaxis.range or [x.min(), x.max()]
        yrange = fig.layout.yaxis.range or [y.min(), y.max()]
        add_line = add_line.replace(" ", "")
        if add_line in ["x=y", "y=x"]:
            fig.add_trace(go.Scatter(mode='lines', x=xrange, y=xrange, showlegend=False))
        elif re.match("(x|y)=", add_line):
            try: c = float(add_line.split("=")[1])
            except: raise ValueError(f"Unrecognized add_line: {add_line}. Please use either 'x=y' or 'x=c' or 'y=c' for some float c.")
            x, y = ([c, c], yrange) if add_line[0] == "x" else (xrange, [c, c])
            fig.add_trace(go.Scatter(mode='lines', x=x, y=y, showlegend=False))
        else:
            raise ValueError(f"Unrecognized add_line: {add_line}. Please use either 'x=y' or 'x=c' or 'y=c' for some float c.")
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    if return_fig:
        return fig
    fig.show(renderer)

def bar(tensor: Union[t.Tensor, List[t.Tensor]], renderer=None, return_fig=False, **kwargs):
    '''
    '''
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "x unified"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if isinstance(tensor, list) or tensor.ndim == 2:
        names = kwargs_pre.pop("names", [f'trace-{i}' for i in range(len(tensor))])
        # TODO: Support tensors of different lenghts
        tensor = t.stack(tensor) if isinstance(tensor, list) else tensor
        df = pd.DataFrame(to_numpy(tensor.T), columns=names)
        fig = px.bar(df, y=names, **kwargs_pre).update_layout(**kwargs_post)
    else:
        fig = px.bar(y=to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post)
    if return_fig:
        return fig
    fig.show(renderer)

def hist(tensor: Union[t.Tensor, List[t.Tensor]], renderer=None, return_fig=True, **kwargs):
    '''
    '''
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.1
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if isinstance(tensor, list) or tensor.ndim == 2:
        names = kwargs_pre.pop("names", [f'trace-{i}' for i in range(len(tensor))])
        # TODO: Support tensors of different lenghts
        tensor = t.stack(tensor) if isinstance(tensor, list) else tensor
        x = pd.DataFrame(to_numpy(tensor.T), columns=names)
    else:
        x = to_numpy(tensor)
    fig = px.histogram(x, **kwargs_pre).update_layout(**kwargs_post)
    if return_fig:
        return fig
    fig.show(renderer)

def figs_to_subplots(figs, rows=1, cols=None, subplot_titles=[], shared_yaxes=False, shared_xaxes=False, 
                     xaxis="", yaxis="", title="", reverse_y=False, reverse_x=False, showlegend=False, shared_coloraxis=True,
                     return_fig=False, horizontal_spacing=None, vertical_spacing=None, **layout_kwargs):
    """ 
    Janky function that takes a list of figures and makes a plot with each as a subplot. Assumes the list is flattened, and will put it into any subplot shape.
    """
    if cols is None:
        cols = len(figs)
    assert (rows * cols)==len(figs)
    sfig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, 
                         shared_xaxes=shared_xaxes, shared_yaxes=shared_yaxes,
                         vertical_spacing=vertical_spacing, horizontal_spacing=horizontal_spacing)
    
    for i, fig in enumerate(figs):
        c = 1 + (i//(rows))
        r = 1 + (i%rows)
        for trace in fig.data:
            # If you want to have more control on the properties of the graph you can either pass things to
            # update_layout or take advantage of the special classes (e.g go.Heatmap) instead of adding traces (which are go.Figure)
            sfig.add_trace(trace, row=r, col=c)
    
    sfig.update_layout(coloraxis=figs[0].layout.coloraxis, barmode=figs[0].layout.barmode, showlegend=showlegend,
                       hovermode=figs[0].layout.hovermode)
    sfig.update_layout(title_text=title, **layout_kwargs)
    if shared_xaxes:
        for c in range(1, cols+1): 
            sfig.update_xaxes(title_text=xaxis, col=c, row=1)
    else:
        sfig.update_xaxes(title_text=xaxis)
    if shared_yaxes:
        for r in range(1, rows+1): 
            sfig.update_yaxes(title_text=yaxis, col=1, row=r)
    else:
        sfig.update_yaxes(title_text=yaxis)
    if reverse_y:
        sfig.update_yaxes(autorange="reversed")
    if reverse_x:
        sfig.update_xaxes(autorange="reversed")

    if return_fig:
        return sfig
    sfig.show()