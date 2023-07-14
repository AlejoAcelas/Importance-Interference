import streamlit as st
import torch
from utils import *
from plotly_utils import *

st.title('Interpretability of Neural Networks')
st.plotly_chart(imshow(torch.rand(10, 10), return_fig=True))

st.header('1. Introduction')
