import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from hamilton.function_modifiers import load_from, save_to, source, extract_fields


# Load datasets from a Parquet file specified by the data_path
@load_from.parquet(path=source("data_path"))
def dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Load and optionally preprocess raw data from a Parquet file."""
    processed_df = raw_df.dropna().reset_index(drop=True)
    return processed_df


# Extract features and target for modeling
@extract_fields(dict(features=pd.DataFrame, target=pd.Series))
def features_and_target(dataset: pd.DataFrame) -> dict:
    features = dataset.drop("target", axis=1)
    target = dataset["target"]
    return {"features": features, "target": target}


# Splitting the data into training and testing sets using extract_fields to manage outputs
@extract_fields(
    dict(X_train=pd.DataFrame, X_test=pd.DataFrame, y_train=pd.Series, y_test=pd.Series)
)
def train_test_split_data(features: pd.DataFrame, target: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


# Training an RF model
def trained_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


# Function to predict using the trained model
def predictions(
    trained_model: RandomForestClassifier, X_test: pd.DataFrame
) -> np.ndarray:
    return trained_model.predict(X_test)


# Evaluate model based on a given scoring function
def accuracy(predictions: np.ndarray, y_test: pd.Series) -> float:
    return accuracy_score(y_test, predictions)
