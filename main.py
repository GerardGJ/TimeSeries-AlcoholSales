from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from src.hyperparametrs import HyperParameters
from src import orchestration
import uvicorn

app = FastAPI(
    title="Alcohol sales API",
    description="A simple API to train and use a machine learning model.",
    version="1.0.0",
)


@app.put("/train/{name}")
def train_model(name:str,request: HyperParameters):
    try:
        orchestration.train_model(name,request)
        return {"status": "success", "message": "Model Trained Correctly"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/visualization/{name}")
def visualization(name:str):
    plt = orchestration.visualization(name)
    return StreamingResponse(plt, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)