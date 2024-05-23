from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def stack_models(xgb : XGBClassifier, lgb : LGBMClassifier) -> object:
    return {"model": "stacked_model"}
