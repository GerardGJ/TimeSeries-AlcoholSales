from src.orchestration import train_model, visualization
from src.hyperparametrs import HyperParameters


if __name__ == "__main__":
    # a = HyperParameters()
    # a.forecast_length=5
    # a.backcast_length=30
    # a.hidden_layer_units=256
    # a.learning_rate=0.0012403393645262744
    # a.batch_size=32
    # a.epochs=100
    # a.ln=True

    # train_model("ts",a)
    visualization('ts')
