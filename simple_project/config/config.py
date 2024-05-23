#from hamilton.config import CategoricalSpace, DiscreteSpace

class CategoricalSpace:
    def __init__(self, options):
        self.options = options

class DiscreteSpace:
    def __init__(self, low, high, step):
        self.low = low
        self.high = high
        self.step = step

ingestion_dct = {
    "toy": {"n_rows": 10},
    "medium": {"n_rows": 1000},
    "large": {"n_rows": 10000},
}

transform_dct = {"impute_method" : CategoricalSpace(["mean", "median", "mode"])}

model_dct = {"toy" : {"n_iterations": DiscreteSpace(1, 10, step=5)},
             "medium" : {"n_iterations": DiscreteSpace(10, 100, step=10)},
             "large" : {"n_iterations": DiscreteSpace(100, 1000, step=100)}}
