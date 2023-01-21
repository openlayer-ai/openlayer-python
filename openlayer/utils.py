import os
import sys
import warnings


# -------------------------- Helper context managers ------------------------- #
class HidePrints:
    """Helper class that suppresses the prints and warnings to stdout and Jupyter's stdout.

    Used as a context manager to hide the print / warning statements that can be inside the user's
    function while we test it.
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


# ----------------------------- Helper functions ----------------------------- #
def write_python_version(dir: str):
    """Writes the python version to the file `python_version` in the specified
    directory (`dir`).

    This is used to register the Python version of the user's environment in the
    when they are uploading a model package.

    Args:
        dir (str): the directory to write the file to.
    """
    with open(f"{dir}/python_version", "w") as f:
        f.write(
            str(sys.version_info.major)
            + "."
            + str(sys.version_info.minor)
            + "."
            + str(sys.version_info.micro)
        )


def remove_python_version(dir: str):
    """Removes the file `python_version` from the specified directory (`dir`).

    Args:
        dir (str): the directory to remove the file from.
    """
    os.remove(f"{dir}/python_version")
