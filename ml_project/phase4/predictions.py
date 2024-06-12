import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from hamilton.function_modifiers import config

###########
# Bridges #
###########
@config.when_in(mode=["training"])
def X__training(X_test: pd.DataFrame) -> pd.DataFrame:
    return X_test


@config.when_in(mode=["inference"])
def X__inference(processed_inference_dataset: pd.DataFrame) -> pd.DataFrame:
    return processed_inference_dataset.drop("target", axis=1)
############

@config.when_in(mode=["training", "inference"])
def predictions(trained_model: ClassifierMixin, X: pd.DataFrame) -> np.ndarray:
    return trained_model.predict(X)