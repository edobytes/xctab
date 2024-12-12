import pytest
import numpy as np
import pandas as pd

from xctab.utilities import loader, logger


def test_loader():
    X, y, cols = loader("data/winequality.csv")

    # type check
    assert isinstance(X, np.ndarray)
    assert isinstance(y, pd.core.series.Series)
    assert isinstance(cols, list)

    # shape check
    assert X.shape[0] == y.shape[0], "# row of features = # rows of target"
    assert X.shape[1] >= 1, "# features >= 1"
    assert len(cols) == 13, "dataset should have 13 cols"


def test_logger():
    log = logger("logs/logger.log")
    log.info(f"Testing the logger")
