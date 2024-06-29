
"""
Goal of this module is to extract what configuration is
being requested from Hamilton nodes.
"""

from hamilton import graph_utils
from hamilton.function_modifiers import config
def find_configurations(module):
    """
    Find all functions that are decorated with the `config` decorator.

    .. code-block:: python

        @config.when_in(mode=["training"])
        def trained_model(
            X_train: pd.DataFrame,
            y_train: pd.Series,
            n_estimators: int = 100,
            random_state: int = 42,
        ) -> GradientBoostingClassifier:
            model = GradientBoostingClassifier(
                n_estimators=n_estimators, random_state=random_state
            )
            model.fit(X_train, y_train)
            return model


    """
    configurations = []
    config_values = {}  # config key -> values
    # get members of the module
    functions = graph_utils.find_functions(module)
    for function_name, function in functions:
        # check if the function is decorated with the config decorator
        if hasattr(function, "resolve") and function.resolve:
            for decorator in function.resolve:
                if isinstance(decorator, config):
                    configurations += decorator._config_used
                    # get bound values out
                    resolver = decorator.does_resolve.resolves
                    bound_values = [c.cell_contents for c in resolver.__closure__]
                    for bound_value in bound_values:
                        # should be dict
                        for key, value in bound_value.items():
                            if key not in config_values:
                                config_values[key] = []
                            config_values[key].append(value)
    return configurations, config_values

if __name__ == '__main__':
    import full_dag
    configurations, config_values = find_configurations(full_dag)
    print(set(configurations))
    print(config_values)
    # TODO: add what type of config resolver -- but I don't think that matters for now.

