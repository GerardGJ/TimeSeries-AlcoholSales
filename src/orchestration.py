from src.modeler import Modeler
from src.visualizer import Visualizer
from src.logging_config import logger
from io import BytesIO
import os

def train_model(name:str,hyperParameter):
    
    modelTrainer = Modeler()

    modelTrainer.train_model(hyperParameter)

    logger.info("Saving model ...")
    modelTrainer.model_saver(name,hyperParameter)
    logger.info("Model saved")
    
    logger.info("Model Training done and saved")

def visualization(modelname:str):
    viz = Visualizer()
    plot:BytesIO = viz.plotPredictions(modelname)

    return plot

def get_folders_scandir(directory):
    """Get all folder names using os.scandir (more efficient)"""
    folders = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir():
                folders.append(entry.name)
    return folders
    