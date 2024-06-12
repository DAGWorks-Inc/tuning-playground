import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from hamilton.function_modifiers import load_from, source, extract_fields, config


@config.when_in(mode=["training", "inference"])
@load_from.parquet(path=source("data_path"))
def dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Load and optionally preprocess raw data from a Parquet file."""
    processed_df = raw_df.dropna().reset_index(drop=True)
    return processed_df


@config.when_in(mode=["inference"])
def processed_inference_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset


@config.when_in(mode=["training"])
@extract_fields(dict(features=pd.DataFrame, target=pd.Series))
def features_and_target(dataset: pd.DataFrame) -> dict:
    features = dataset.drop("target", axis=1)
    target = dataset["target"]
    return {"features": features, "target": target}


@config.when_in(mode=["training"])
@extract_fields(
    dict(X_train=pd.DataFrame, X_test=pd.DataFrame, y_train=pd.Series, y_test=pd.Series)
)
def train_test_split_data(features: pd.DataFrame, target: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


@config.when_in(mode=["training"])
def X__training(X_test: pd.DataFrame) -> pd.DataFrame:
    return X_test


@config.when_in(mode=["inference"])
def X__inference(processed_inference_dataset: pd.DataFrame) -> pd.DataFrame:
    return processed_inference_dataset.drop("target", axis=1)


@config.when_in(mode=["training", "inference"])
def predictions(trained_model: ClassifierMixin, X: pd.DataFrame) -> np.ndarray:
    return trained_model.predict(X)


@config.when_in(mode=["training"])
def accuracy(predictions: np.ndarray, y_test: pd.Series) -> float:
    return accuracy_score(y_test, predictions)
