import pandas as pd


class Collector:
    def __init__(self):
        self.data = []
        self._variables = None

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Storing the input values
            values = list(args) + list(kwargs.values())

            # Running the decorated function and getting the output
            result = func(*args, **kwargs)

            # Saving the variables' names
            if not self._variables:
                arg_names = list(func.__code__.co_varnames)[:len(args)]
                kwarg_names = list(kwargs.keys())
                self._variables = arg_names + kwarg_names + ['output']

            # Storing the input and output values
            row = values + [result]
            self.data.append(row)

            return result

        return wrapper

    @property
    def dataset(self):
        return pd.DataFrame(self.data, columns=self._variables)

    @property
    def variables(self):
        return self._variables[:-1]  # Excluding 'output'
