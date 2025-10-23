import torch
import pandas as pd
import os
import yaml

from dotenv import load_dotenv
from nbeats_pytorch.model import NBeatsNet

from src.logging_config import logger
from src.repository import Repository
from src.pytorchDataset import TimeSeriesDataset
from src.hyperparametrs import HyperParameters

load_dotenv()

def ModelTrainer():
    def __init__(self):
        with open("config.yaml") as file:
            self.config = yaml.safe_load(file)
    
    def createModel(self,hyperParameters:HyperParameters,device):

        model = NBeatsNet(
                device=device,
                stack_types=(NBeatsNet.TREND_BLOCK, 
                             NBeatsNet.SEASONALITY_BLOCK),
                forecast_length=hyperParameters.forecast_length,
                backcast_length=hyperParameters.backcast_length,
                hidden_layer_units=hyperParameters.hidden_layer_units
            ).to(device)
        
        return model
    
    def load_data(self):
        repo = Repository()
        

    def train_model(self,hyperParameters:HyperParameters):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = createModel(hyperParameters,device)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=hyperParameters.learning_rate)
        
        for epoch in range(hyperParameters.epochs):
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                backcast, forecast = model(x_batch)
                loss = criterion(forecast, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch + 1}/{hyperParameters.epochs}, Loss: {loss.item()}')
    
    def model_saver(self, modelName:str):
        PATH = os.path(self.config['modelFolder'],modelName+".pt")
        if os.path.isfile(PATH):
            return "This file already exists"
        torch.save(self.model.state_dict(), PATH)

        logger.info(f"Model succesfully saved with name {modelName}")