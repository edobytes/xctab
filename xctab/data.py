import typer
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import logger
from config import DATA_DIRECTORY, DATASET_PATH, LOGS_DIRECTORY


app = typer.Typer()


@app.command()
def main():
    """
    Preparing the dataset for analysis & training
    """
    data_directory = DATA_DIRECTORY
    dataset_path = DATASET_PATH
    logs_directory = LOGS_DIRECTORY

    random_state = 42
    log = logger(logs_directory / "data.log")
    log.info(f"Loading variables from configuration")

    log.info("Extracting data")
    wine = pd.read_csv(dataset_path)
    wine = wine.dropna()

    log.info("Transforming data")
    bins = [0, 4, 10]
    labels =  ['bad', 'good']
    wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = labels)
    label_quality = LabelEncoder()
    wine['quality'] = label_quality.fit_transform(wine['quality'])

    log.info("Splitting data")
    red = wine[wine['type'] == 'red']
    white = wine[wine['type'] == 'white']
    red = red.drop("type", axis=1)
    white = white.drop("type", axis=1)

    red_good = red[red['quality'] == 1]
    red_bad = red[red['quality'] == 0]
    white_good = white[white['quality'] == 1]
    white_bad = white[white['quality'] == 0]

    red_train, red_good_remainder = train_test_split(red_good, \
        test_size=0.1, random_state=random_state, stratify=red_good['quality'])

    white_train, white_good_remainder = train_test_split(white_good, \
        test_size=0.1, random_state=random_state, stratify=white_good['quality'])

    red_test = pd.concat([red_good_remainder, red_bad], axis=0) 
    white_test = pd.concat([white_good_remainder, white_bad], axis=0) 

    for df in [red_train, red_test, white_train, white_test]:
        df = df.sample(frac=1).reset_index(drop=True)

    log.info(f"Exporting train/test dataset to the `{data_directory}` directory")
    red_train.to_csv(data_directory/ "red_train.csv", index=False)
    red_test.to_csv(data_directory / "red_test.csv", index=False)
    white_train.to_csv(data_directory / "white_train.csv", index=False)
    white_test.to_csv(data_directory / "white_test.csv", index=False)
    log.info(f" - red wine train set: {red_train.shape}")
    log.info(f" - red wine test set: {red_test.shape}")
    log.info(f" - white wine train set: {white_train.shape}")
    log.info(f" - white wine test set: {white_test.shape}")
    

if __name__ == "__main__":
    app()