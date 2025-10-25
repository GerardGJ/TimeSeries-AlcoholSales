from src.modeler import Modeler
from src.visualizer import Visualizer
from src.logging_config import logger

def train_model(name:str,hyperParameter):
    
    modelTrainer = Modeler()

    modelTrainer.train_model(hyperParameter)

    modelTrainer.model_saver(name,hyperParameter)
    logger.info("Model Training done and saved")

def visualization(modelname:str):
    viz = Visualizer()
    viz.plotPredictions(modelname)
    