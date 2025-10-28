from src.modeler import Modeler
from src.repository import Repository
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import yaml

class Visualizer():
    def __init__(self):
        with open("config.yaml") as file:
            self.config = yaml.safe_load(file)
    def plotPredictions(self,modelName:str):
        model = Modeler()
        repo = Repository(self.config["database"])

        hyperparameters = model.load_model(modelName)
        dataToGet = hyperparameters['backcast_length'] + hyperparameters['forecast_length']

        data = repo.getTable('data')
        data.columns = ["DATE","SALES"]
        data.set_index('DATE',inplace=True)
        data.index = pd.DatetimeIndex(data.index,freq='MS')

        x_batch = data.iloc[-dataToGet:-hyperparameters['forecast_length'],0] 
        y_batch = data.iloc[-hyperparameters['forecast_length']:,0]

        backcast, forecast = model.predict(modelName,x_batch)

        # Plotting a sample:
        backcast = backcast[0]
        forecast = forecast[0]

        # Get the lengths
        backcast_len = len(backcast)
        forecast_len = len(forecast)

        # Create time indices
        backcast_time = np.arange(backcast_len)
        forecast_time = np.arange(backcast_len, backcast_len + forecast_len)

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot backcast and x_batch (historical data)
        plt.plot(backcast_time, backcast, label='Backcast', linewidth=2, color='green')
        plt.plot(backcast_time, x_batch, label='Backcast Input', linewidth=2, color='blue', alpha=0.7)

        # Plot forecast and y_batch (future data)
        plt.plot(forecast_time, forecast, label='Forecast', linewidth=2, color='red', linestyle='--')
        plt.plot(forecast_time, y_batch, label='Actual Forecast', linewidth=2, color='orange', alpha=0.7)

        # Add a vertical line to separate historical and forecast
        plt.axvline(x=backcast_len, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

        # Labels and legend
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Backcast vs Forecast Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)  # Important: reset pointer to beginning
        plt.close()  # Close the figure to free memory

        return buf