from pydantic import BaseModel

class HyperParameters(BaseModel):
    forecast_length:int
    backcast_length:int
    hidden_layer_units:int
    learning_rate:float
    batch_size:int
    epochs:int
    ln:bool

# class HyperParameters():
#     pass