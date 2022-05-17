class TaskType:
    """A selection of machine learning tasks supported by Unbox. """

    @property
    def TextClassification(self) -> str:
        """
        For sequence classification tasks.
        """
        return "text-classification"

    @property
    def TabularClassification(self) -> str:
        """
        For tabular classification tasks.
        """
        return "tabular-classification"

    @property
    def TabularRegression(self) -> str:
        """
        Coming soon!
        """
        return "tabular-regression"
