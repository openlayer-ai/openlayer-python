"""
Module to handle data collection for functions using the Collector decorator.
"""

import logging

import pandas as pd


class Collector:
    """
    A decorator class used to collect input and output data from the decorated functions.
    """

    def __init__(self):
        self.data = []
        self._variables = None

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            result = None
            try:
                # Storing the input values
                values = list(args) + list(kwargs.values())

                # Running the decorated function and getting the output
                result = func(*args, **kwargs)

                # Saving the variables' names
                if not self._variables:
                    arg_names = list(func.__code__.co_varnames)[: len(args)]
                    kwarg_names = list(kwargs.keys())
                    self._variables = arg_names + kwarg_names + ["output"]

                # Storing the input and output values
                row = values + [result]
                self.data.append(row)

            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Collector failed with error: %s", e)
                result = func(
                    *args, **kwargs
                )  # Ensure the decorated function still runs

            return result

        return wrapper

    @property
    def dataset(self):
        return pd.DataFrame(self.data, columns=self._variables)

    @property
    def variables(self):
        return self._variables[:-1]  # Excluding 'output'
