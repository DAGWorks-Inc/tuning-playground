import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from hamilton.function_modifiers import config

@config.when_in(mode=["training"])
def trained_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model