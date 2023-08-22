import streamlit as st
import torch
from model_settings import *
from my_plotly_utils import *
import os 
import glob
from custom_plots import *

if torch.cuda.is_available():
  DEVICE = 'cuda'
else:
  DEVICE = 'cpu'

### Set up the sidebar

st.title('Importance and Interference in Toy Models')

with st.sidebar:
    st.header('Model parameters')
    st.write("Select the model parameters")
    n_features = st.select_slider('Number of features', options=[20, 50, 100], value=20)
    n_hidden = st.select_slider('Hidden dimension', options=[2, 3, 5], value=2)
    importance_discount = st.select_slider('Factor for discounting feature importance', 
                                           options=[0.75, 0.95, 1.0], value=0.75)
    st.write('Select the feature probability... ')

    st.divider()
    training_data = st.selectbox('Training data', options=['Uniform', 'One-hot'], index=0)
    loss_fn = st.selectbox('Loss function', options=['MSE', 'Cross-entropy'], index=0)

    st.divider()
    run_model = st.checkbox('Run model', value=False)
    save_model = st.checkbox('Save model', value=False)
    # uniform_feature_prob = cols2[0].checkbox('Uniform feature ', value=False)

if training_data == 'One-hot' and loss_fn == 'MSE':
    st.write('One-hot data and MSE loss not supported yet. Select something else.')
    
model_name = f"MLP_f{n_features}_h{n_hidden}_d{int(100*importance_discount)}.pth"

## Set up the model

if run_model:
    config = Config(
        n_features = n_features,
        n_hidden = n_hidden,
        n_instances = 10,
    )

    importance = importance_discount**torch.arange(config.n_features)
    feature_probability = ((2*config.n_features) ** -torch.linspace(0.2, 1, config.n_instances))

    model = BasicMLP(
        config=config,
        device=DEVICE,
        importance = importance[None, :],
        feature_probability = feature_probability[:, None]
    )

    ## Verify whether a model like this has been trained before

    matching_files = glob.glob(os.path.join(os.getcwd(), 'models', model_name))
    if matching_files:
        st.write(f"Model found in storage. Uploading model from memory")
        model.load_state_dict(torch.load(matching_files[0]))
    else:
        st.write(f"Model not found in storage. Training model on the fly.")
        optimize(
           model, 
           steps=10_000,
           lr=1e-3,
           batch_fn=Model.gen_batch_rand,
           loss_fn=Model.mse_loss,)
        ## Modify the optimize function to yield the loss and training step


        if save_model:
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'models', model_name))

else:
   st.write("Select model parameters and check 'Run model' to plot the model.")

## Plot the model

if run_model:
    st.header('Model description perhaps?')
    plots_tab, annotations_tab = st.tabs(['Plots', 'Annotations'])

    with plots_tab:
        fig = plot_noise_curves(model)
        st.plotly_chart(fig, use_container_width=True)

        fig = plot_hidden_2d(model)
        st.plotly_chart(fig, use_container_width=True)

    with annotations_tab:

        new_anot = st.text_area('Annotations', value='At some point this field might have persistence across sessions', height=200, on_change=lambda *args: print(new_anot))
