import torch
import pandas as pd
import os
import yaml
import math
import json
import numpy as np

from dotenv import load_dotenv
from nbeats_pytorch.model import NBeatsNet
from torch.utils.data import DataLoader

from src.logging_config import logger
from src.repository import Repository
from src.pytorchDataset import TimeSeriesDataset
from src.hyperparametrs import HyperParameters

load_dotenv()

class Modeler():
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
    
    def model_saver(self, modelName:str, hyperParameters:HyperParameters):
        FOLDERPATH = os.path.join(self.config['modelFolder'],modelName)
        PATHMODEL = os.path.join(FOLDERPATH,modelName+".pt")
        PATHMETADATA = os.path.join(FOLDERPATH,modelName+".json")

        if not os.path.exists(FOLDERPATH):
            os.makedirs(FOLDERPATH)

        if os.path.isfile(PATHMODEL):
            logger.info("This file already exists")

        with open(PATHMETADATA, "r+") as f:
            json.dump(hyperParameters.__dict__,f)

        torch.save(self.model, PATHMODEL)
        logger.info(f"Model succesfully saved with name {modelName}")
    
    def load_model(self,modelName:str):
        FOLDERPATH = os.path.join(self.config['modelFolder'],modelName)
        PATHMODEL = os.path.join(FOLDERPATH,modelName+".pt")
        PATHMETADATA = os.path.join(FOLDERPATH,modelName+".json")

        self.model = torch.load(PATHMODEL, weights_only=False)
        self.model.eval()

        with open(PATHMETADATA) as f:
            hyperParameters = json.load(f)

        logger.info(f"Model loaded correctly")

        return hyperParameters

    def predict(self, modelName:str, x_batch:np.ndarray, y_batch:np.ndarray):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        with torch.no_grad():
            backcast, forecast = self.model(x_batch)

        # Convert to numpy for plotting
        backcast = backcast.cpu().numpy()
        forecast = forecast.cpu().numpy()

        return backcast, forecast