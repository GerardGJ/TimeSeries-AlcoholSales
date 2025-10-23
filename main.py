from fastapi import FastAPI
from src.hyperparametrs import HyperParameters
from src import orchestration

app = FastAPI()


@app.put("/train")
def train_model(request: HyperParameters):
    orchestration.train_model("ts")
