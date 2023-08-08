# pylint: disable=invalid-name,broad-exception-raised, consider-using-with
"""
Module that contains the classes for environment management, such as conda.
"""
import logging
import os
import shutil
import subprocess
from typing import List, Optional, Set

from .. import utils


class CondaEnvironment:
    """Conda environment manager.

    Parameters
    ----------
    env_name : str
        Name of the conda environment.
    requirements_file_path : str
        Path to the requirements file.
    python_version_file_path : str
        Path to the python version file.
    logs_file_path : str, optional
        Where to log the output of the conda commands.
        If None, the output is shown in stdout.
    """

    def __init__(
        self,
        env_name: str,
        requirements_file_path: str,
        python_version_file_path: str,
        logger: Optional[logging.Logger] = None,
    ):
        self._conda_exe = self._get_executable()
        self._conda_prefix = self._get_conda_prefix()
        self._bash = self._get_bash()
        self.env_name = env_name
        self.requirements_file_path = requirements_file_path
        self.python_version_file_path = python_version_file_path
        self.logger = logger or logging.getLogger("validators")

    def __enter__(self):
        existing_envs = self.get_existing_envs()
        if self.env_name in existing_envs:
            self.logger.info("Found existing conda environment '%s'.", self.env_name)
        else:
            self.create()
            self.install_requirements()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.deactivate()

    def _get_executable(self) -> str:
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe is None:
            raise Exception("Conda is not available on this machine.")
        return conda_exe

    def _get_bash(self) -> str:
        """Gets the bash executable."""
        shell_path = shutil.which("bash")
        if shell_path is None:
            raise Exception("Bash is not available on this machine.")
        return shell_path

    def _get_conda_prefix(self) -> str:
        """Gets the conda base environment prefix.

        E.g., '~/miniconda3' or '~/anaconda3'
        """
        prefix = subprocess.check_output([self._conda_exe, "info", "--base"])
        return prefix.decode("UTF-8").strip()

    def create(self):
        """Creates a conda environment with the specified name and python version."""
        self.logger.info("Creating a new conda environment '%s'... \n", self.env_name)

        with open(
            self.python_version_file_path, "r", encoding="UTF-8"
        ) as python_version_file:
            python_version = python_version_file.read().split(".")[:2]
            python_version = ".".join(python_version)

        process = subprocess.Popen(
            [
                self._conda_exe,
                "create",
                "-n",
                f"{self.env_name}",
                f"python={python_version}",
                "--yes",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        with process.stdout:
            utils.log_subprocess_output(self.logger, process.stdout)
        exitcode = process.wait()

        if exitcode != 0:
            raise Exception(
                f"Failed to create conda environment '{self.env_name}' with python "
                f"version {python_version}."
            )

    def delete(self):
        """Deletes the conda environment with the specified name."""
        self.logger.info("Deleting conda environment '%s'...", self.env_name)

        process = subprocess.Popen(
            [
                self._conda_exe,
                "env",
                "remove",
                "-n",
                f"{self.env_name}",
                "--yes",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        with process.stdout:
            utils.log_subprocess_output(self.logger, process.stdout)
        exitcode = process.wait()

        if exitcode != 0:
            raise Exception(f"Failed to delete conda environment '{self.env_name}'.")

    def get_existing_envs(self) -> Set[str]:
        """Gets the names of all existing conda environments."""
        self.logger.info("Checking existing conda environments...")

        awk_command = "awk '{print $1}"
        list_envs_command = f"""
        {self._conda_exe} env list | {awk_command}'
        """

        try:
            envs = subprocess.check_output(
                list_envs_command,
                shell=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to list conda environments."
                f"- Error code returned {err.returncode}: {err.output}"
            ) from None
        envs = set(envs.decode("UTF-8").split("\n"))
        return envs

    def activate(self):
        """Activates the conda environment with the specified name."""
        self.logger.info("Activating conda environment '%s'...", self.env_name)

        activation_command = f"""
        source {self._conda_prefix}/etc/profile.d/conda.sh
        eval $(conda shell.bash hook)
        conda activate {self.env_name}
        """

        try:
            subprocess.check_call(
                activation_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                shell=True,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to activate conda environment '{self.env_name}'."
                f"- Error code returned {err.returncode}: {err.output}"
            ) from None

    def deactivate(self):
        """Deactivates the conda environment with the specified name."""
        self.logger.info("Deactivating conda environment '%s'...", self.env_name)

        deactivation_command = f"""
        source {self._conda_prefix}/etc/profile.d/conda.sh
        eval $(conda shell.bash hook)
        conda deactivate
        """

        try:
            subprocess.check_call(
                deactivation_command,
                shell=True,
                executable=self._bash,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to deactivate conda environment '{self.env_name}'."
                " Please check the model logs for details. \n"
                f"- Error code returned {err.returncode}: {err.output}"
            ) from None

    def install_requirements(self):
        """Installs the requirements from the specified requirements file."""
        self.logger.info(
            "Installing requirements in conda environment '%s'...", self.env_name
        )

        exitcode = self.run_commands(
            ["pip", "install", "-r", self.requirements_file_path],
        )
        if exitcode != 0:
            raise Exception(
                "Failed to install the depencies specified in the requirements.txt file."
            )

    def run_commands(self, commands: List[str]):
        """Runs the specified commands inside the conda environment.

        Parameters
        ----------
        commands : List[str]
            List of commands to run.
        """
        full_command = f"""
        source {self._conda_prefix}/etc/profile.d/conda.sh
        eval $(conda shell.bash hook)
        conda activate {self.env_name}
        {" ".join(commands)}
        """
        process = subprocess.Popen(
            full_command,
            shell=True,
            executable=self._bash,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        with process.stdout:
            utils.log_subprocess_output(self.logger, process.stdout)
        exitcode = process.wait()
        return exitcode
