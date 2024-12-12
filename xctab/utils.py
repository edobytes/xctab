import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def logger(log_path):
    """
    Save log to specified path.

    Args:
        log_path (str): eg: "../log/train.log"

    Returns:
        logger (logging.get_Logger): logging object
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def loader(data_path):
    """
    Load data from specified path.

    Args:
        data (str): file path
    
    Returns:
        (tuple): X (features), y (target variable), and column names (cols)
    """
    df = pd.read_csv(data_path)
    X = df.drop("quality", axis=1)
    y = df['quality']
    cols = list(df.columns)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X, y, cols


def sampler(data_path, num_samples):
    """
    Sample n times from a specified set.

    Args:
        data_path (str): data path
        num_samples (int): n

    Returns:
        samples (pd.Dataframe): n samples 
    """
    df = pd.read_csv(data_path)
    samples = df.sample(n=num_samples, random_state=1)
    if 'quality' in samples.columns:
        samples = samples.drop("quality", axis=1)
    return samples


def recoder(predictions, implementation='pyod'):
    """
    Recode output labels from different packages.

    Args:
        predictions (list): predicted labels
        implementation (str): package name

    Returns:
        np.array: recoded labels
    """
    assert implementation in ['sklearn', 'pyod']
    if implementation == 'sklearn':
        l = lambda x: 1 if x == 1 else 0
    else:
        l = lambda x: 0 if x == 1 else 1
    return np.array(list(map(l, predictions)))