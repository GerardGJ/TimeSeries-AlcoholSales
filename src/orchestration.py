from src.retrain import ModelTrainer

def train_model(name,hyperParameter):
    
    modelTrainer = ModelTrainer()

    modelTrainer.train_model(hyperParameter)

    modelTrainer.model_saver(name)


    