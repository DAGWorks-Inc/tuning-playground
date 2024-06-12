import pandas as pd
from sklearn.model_selection import train_test_split
from hamilton.function_modifiers import extract_fields, config
from typing import Dict

@config.when_in(mode=["training"])
@extract_fields(
    dict(X_train=pd.DataFrame, X_test=pd.DataFrame, y_train=pd.Series, y_test=pd.Series)
)
def train_test_split_data(features: pd.DataFrame, target: pd.Series) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
