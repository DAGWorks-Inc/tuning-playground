from pandas import pd

def fit_model_xgb(transformed_dataset: pd.DataFrame) -> object:
    return {"model": "xgb"}

def fit_model_lgb(transformed_dataset: pd.DataFrame) -> object:
    return {"model": "lgb"}
