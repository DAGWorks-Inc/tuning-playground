import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from hamilton.function_modifiers import config

@config.when_in(mode=["training"])
def accuracy(predictions: np.ndarray, y_test: pd.Series) -> float:
    return accuracy_score(y_test, predictions)