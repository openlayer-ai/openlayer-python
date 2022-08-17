import os
import sys
import warnings


class HidePrints:
    """Class that suppresses the prints and warnings to stdout and Jupyter's stdout. Used
    to hide the print / warning statements that can be inside the user's function while
    we test it.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        sys._jupyter_stdout = sys.stdout
        warnings.filterwarnings("ignore")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys._jupyter_stdout = sys.stdout
        warnings.filterwarnings("default")
