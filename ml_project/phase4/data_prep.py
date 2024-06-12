import pandas as pd
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