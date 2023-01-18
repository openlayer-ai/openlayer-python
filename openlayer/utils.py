import distutils
import os
import shutil
import sys
import tempfile
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


class TempDirectory(object):
    """Helper class that creates and cleans up a temporary directory.

    >>> with TempDirectory() as tempdir:
    >>>     print(os.path.isdir(tempdir))
    """

    def __init__(
        self,
        cleanup=True,
        prefix="temp",
    ):

        self._cleanup = cleanup
        self._prefix = prefix
        self.path = None

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.path}>"

    def __enter__(self):
        self.create()
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cleanup:
            self.cleanup()

    def create(self):
        if self.path is not None:
            return self.path

        tempdir = tempfile.mkdtemp(prefix=f"openlayer-{self._prefix}-")
        self.path = os.path.realpath(tempdir)

    def cleanup(self, ignore_errors=False):
        """
        Remove the temporary directory created
        """
        if self.path is not None and os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=ignore_errors)
        self.path = None


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


def copy_to_tmp_dir(dir: str) -> str:
    """Copies the contents of the specified directory (`dir`) to a temporary directory.

    Args:
        dir (str): the directory to copy the contents from.

    Returns:
        str: the path to the temporary directory.
    """
    tmp_dir = tempfile.mkdtemp()
    distutils.dir_util.copy_tree(dir, tmp_dir)

    return tmp_dir
