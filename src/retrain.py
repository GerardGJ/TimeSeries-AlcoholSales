import torch
import pandas as pd
import os
import yaml
import math

from dotenv import load_dotenv
from nbeats_pytorch.model import NBeatsNet
from torch.utils.data import DataLoader

from src.logging_config import logger
from src.repository import Repository
from src.pytorchDataset import TimeSeriesDataset
from src.hyperparametrs import HyperParameters

load_dotenv()

class ModelTrainer():
    def __init__(self):
        with open("config.yaml") as file:
            self.config = yaml.safe_load(file)
    
    def createModel(self,hyperParameters:HyperParameters,device):

        self.model = NBeatsNet(
                device=device,
                stack_types=(NBeatsNet.TREND_BLOCK, 
                             NBeatsNet.SEASONALITY_BLOCK),
                forecast_length=hyperParameters.forecast_length,
                backcast_length=hyperParameters.backcast_length,
                hidden_layer_units=hyperParameters.hidden_layer_units
            ).to(device)

    def load_data(self,train_model) -> pd.DataFrame:
        repo = Repository()

        df = repo.getTable(train_model)

        return df

    def train_model(self,hyperParameters:HyperParameters):

        data = self.load_data('data')
        data.columns = ["DATE","SALES"]
        data.set_index('DATE',inplace=True)
        data.index = pd.DatetimeIndex(data.index,freq='MS')

        if hyperParameters.ln:
            data = data.map(lambda x: math.log(x))

        df_nparry = data.transpose().to_numpy()[0]

        dataset = TimeSeriesDataset(df_nparry, 
                                    hyperParameters.backcast_length, 
                                    hyperParameters.forecast_length
                                )
    
        dataloader = DataLoader(dataset, 
                                batch_size=32, 
                                shuffle=True
                            )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.createModel(hyperParameters,device)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=hyperParameters.learning_rate)
        
        for epoch in range(hyperParameters.epochs):
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                backcast, forecast = self.model(x_batch)
                loss = criterion(forecast, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch + 1}/{hyperParameters.epochs}, Loss: {loss.item()}')
        
        logger.info("Model trained successfully")
    
    def model_saver(self, modelName:str):
        print(self.config)
        PATH = os.path.join(self.config['modelFolder'],modelName+".pt")
        if os.path.isfile(PATH):
            return "This file already exists"
        torch.save(self.model.state_dict(), PATH)

        logger.info(f"Model succesfully saved with name {modelName}")