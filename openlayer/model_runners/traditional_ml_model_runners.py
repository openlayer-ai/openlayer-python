# pylint: disable=invalid-name,broad-exception-raised, consider-using-with
"""
Module with the concrete traditional ML model runners.

"""
import ast
import datetime
import os
import shutil
import tempfile
from abc import ABC, abstractmethod

import pandas as pd

from . import base_model_runner


class TraditionalMLModelRunner(base_model_runner.ModelRunnerInterface, ABC):
    """Extends the base model runner for traditional ML models."""

    @abstractmethod
    def validate_minimum_viable_config(self) -> None:
        pass

    def _run_in_memory(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs the input data through the model in memory."""
        raise NotImplementedError(
            "Running traditional ML in memory is not implemented yet. "
            "Please use the runner in a conda environment."
        )

    def _run_in_conda(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs the input data through the model in the conda
        environment.
        """
        self.logger.info("Running traditional ML model in conda environment...")

        # Copy the prediction job script to the model package
        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        self._copy_prediction_job_script(current_file_dir)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the input data to a csv file
            input_data.to_csv(f"{temp_dir}/input_data.csv", index=False)

            # Run the model in the conda environment
            with self._conda_environment as env:
                self.logger.info(
                    "Running %s rows through the model...", len(input_data)
                )
                exitcode = env.run_commands(
                    [
                        "python",
                        f"{self.model_package}/prediction_job.py",
                        "--input",
                        f"{temp_dir}/input_data.csv",
                        "--output",
                        f"{temp_dir}/output_data.csv",
                    ]
                )
                if exitcode != 0:
                    self.logger.error(
                        "Failed to run the model. Check the stack trace above for details."
                    )
                    raise Exception(
                        "Failed to run the model in the conda environment."
                    ) from None

            self.logger.info("Successfully ran data through the model!")
            # Read the output data from the csv file
            output_data = pd.read_csv(f"{temp_dir}/output_data.csv")

            output_data = self._post_process_output(output_data)
            output_data["output_time_utc"] = datetime.datetime.utcnow().isoformat()

        return output_data

    @abstractmethod
    def _copy_prediction_job_script(self, current_file_dir: str):
        """Copies the correct prediction job script to the model package.

        Needed if the model is intended to be run in a conda environment."""
        pass

    @abstractmethod
    def _post_process_output(self, output_data: pd.DataFrame) -> pd.DataFrame:
        """Performs any post-processing on the output data.

        Needed if the model is intended to be run in a conda environment."""
        pass


# -------------------------- Concrete model runners -------------------------- #
class ClassificationModelRunner(TraditionalMLModelRunner):
    """Wraps classification models."""

    def validate_minimum_viable_config(self) -> None:
        pass

    def _copy_prediction_job_script(self, current_file_dir: str):
        """Copies the classification prediction job script to the model package."""
        shutil.copy(
            f"{current_file_dir}/prediction_jobs/classification_prediction_job.py",
            f"{self.model_package}/prediction_job.py",
        )

    def _post_process_output(self, output_data: pd.DataFrame) -> pd.DataFrame:
        """Post-processes the output data."""
        processed_output_data = output_data.copy()

        # Make the items list of floats (and not strings)
        processed_output_data["output"] = processed_output_data["output"].apply(
            ast.literal_eval
        )

        return processed_output_data


class RegressionModelRunner(TraditionalMLModelRunner):
    """Wraps regression models."""

    def validate_minimum_viable_config(self) -> None:
        pass

    def _copy_prediction_job_script(self, current_file_dir: str):
        """Copies the regression prediction job script to the model package."""
        shutil.copy(
            f"{current_file_dir}/prediction_jobs/regression_prediction_job.py",
            f"{self.model_package}/prediction_job.py",
        )

    def _post_process_output(self, output_data: pd.DataFrame) -> pd.DataFrame:
        """Post-processes the output data."""
        return output_data
