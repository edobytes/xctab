import typer
import sklearn
import pandas as pd
from pickle import dump
from pathlib import Path
from typing_extensions import Annotated
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from config import DATA_DIRECTORY, LOGS_DIRECTORY, MODELS_DIRECTORY
from utils import logger, loader


app = typer.Typer()


@app.command()
def main(wine_type: Annotated[str, typer.Option("--wine-type", "-wt")],
        model_name: Annotated[str, typer.Option("--model-name", "-mn")]    
        ):
    """
    Trains & persists a model.

    Args:
        wine_type (str): red or white
        model_name (str): a small selection from pyod's many options

    Returns:
        None
    """
    assert wine_type in ['red', 'white'], "Invalid wine type"
    assert model_name in ['autoencoder', 'ecod', 'knn', 'iforest'], "Invalid model name"

    data_directory = DATA_DIRECTORY
    logs_directory = LOGS_DIRECTORY
    models_directory = MODELS_DIRECTORY

    log = logger(logs_directory / "train.log")
    log.info(f"Loading parameters from configuration")

    if wine_type == 'red':
        train_set = data_directory / "red_train.csv"
    else:
        train_set = data_directory / "white_train.csv"

    log.info(f"Separating features from response variable")
    X, y, _ = loader(train_set)

    log.info(f"Training a {model_name} model")
    if model_name == 'autoencoder': 
        model = AutoEncoder()
    elif model_name == 'ecod': 
        model = ECOD()
    elif model_name == 'knn':
        model = KNN()
    elif model_name == 'iforest':
        model = IForest()
    model.fit(X)

    log.info(f"Saving the {model_name} trained model")
    models_directory.parent.mkdir(parents=True, exist_ok=True)
    model_path = models_directory.joinpath(f"{model_name}-{wine_type}.pkl")
    with open(model_path, "wb") as f:
        dump(model, f)


if __name__ == "__main__":
    app()
