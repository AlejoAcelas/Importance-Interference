import streamlit as st
import numpy as np
from my_plotly_utils import *

with st.sidebar:
    p = st.slider('p', 0.0, 1.0, 0.01)
    n = st.slider('n', 1, 1000, 10)
    alpha = st.slider('alpha', 0.0, 1.0, 0.01)

var_alpha = np.linspace(0, 1, 100)
var_p = np.linspace(0, 1, 100)
var_n = np.linspace(1, 1000, 100)

plot_kwargs = dict(names=['Inactive loss', 'Interference loss'],return_fig=True)

alpha_plot = line([(1-p)**(var_alpha*n), var_alpha*n*p], x=var_alpha, **plot_kwargs,
                  labels=dict(x='Number of neighbors', value='Probability of the loss class'))
# p_plot = line([(1-var_p)**(alpha*n), alpha*n*var_p], x=var_p, **plot_kwargs)
# n_plot = line([(1-p)**(alpha*var_n), alpha*var_n*p], x=var_n, **plot_kwargs)


st.plotly_chart(alpha_plot)
# st.plotly_chart(p_plot)
# st.plotly_chart(n_plot)
