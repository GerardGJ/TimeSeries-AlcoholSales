import streamlit as st
import requests

st.write("""
    # Playground

    In this section you are going to be able to make different 
    models which then can be visualized in the model visualization
    page.
""")

hyperparameters = dict()

columns = st.columns(3)
with columns[0]:
    modelName = st.text_input("Name of the model")

columns_row1 = st.columns(3)

with columns_row1[0]:
    hyperparameters['forecast_length'] = st.number_input(
        "Forecast lenght",
        1,10
    )

with columns_row1[1]:
    hyperparameters['backcast_length'] = st.number_input(
        "Backcast lenght",
        1,30
    )

with columns_row1[2]:
    hyperparameters['hidden_layer_units'] = st.selectbox(
        "Number of hidden layers",
        [64, 128, 256]
    )

columns_row2 = st.columns(3)

with columns_row2[0]:
    hyperparameters['learning_rate'] = st.slider(
        "Select a value",
        min_value=0.0001,
        max_value=0.01,
        value=0.005,  
        step=0.0001,  
        format="%.4f"  
    )

with columns_row2[1]:
    hyperparameters['batch_size'] = st.selectbox(
        "Batch size",
        [16, 32, 64]
    )

with columns_row2[2]:
    hyperparameters['epochs'] = st.selectbox(
        "Epochs",
        [50, 100, 200]
    )

columns_row3 = st.columns(3)

with columns_row3[1]:
    hyperparameters['ln'] = st.toggle("Want to ln the data?")

start_execution = st.button("Start training of the model")
if start_execution:
    response = requests.put(
        f"http://localhost:8080/train/{modelName}", 
        json=hyperparameters
    )