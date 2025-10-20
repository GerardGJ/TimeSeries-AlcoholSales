# Alcohol sales

## Objective

The objective of this project is to try different Time Series models in order to start learning about them.

## Models

The different models that will be tested in this project are:

**Time Series Models:**

- Exponential Smoothing
- Prophet

**Machine Learning Model:**

- XGBoost (In-progress)
- LightGBM (In-progress)

**Modern Specialized Models:**

- N-BEATS
- DeepAR

## Deployment

For the deployment section, I am going to make a deployment of the best NBeats-Net model after hyperparameter tuning using optuna.

The whole application will be containarized using Dockers and also will use make a little UI using streamlit.

### Streamlit

Regarding streamlit the intention is to have several modules:

- The best model: This module will show the best model predictions as well as the plot the model predictions.
- Retraing/Playground to train and visualize several hyperparameters.
