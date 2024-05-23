import pandas as pd

def transformed_dataset(raw_dataset: pd.DataFrame, impute_method: str) -> pd.DataFrame:
    return raw_dataset + 1