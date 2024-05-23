from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def ensemble_models(xgb : XGBClassifier, lgb : LGBMClassifier) -> object:
    return {"model": "ensemble_model"}
