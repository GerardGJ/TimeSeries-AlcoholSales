from pydantic import BaseModel

class HyperParameters(BaseModel):
    forecast_length:int
    backcast_length:int
    hidden_layer_units:int
    learning_rate:int
    batch_size:int
    epochs:int