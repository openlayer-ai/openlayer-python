"""Base class for an Openlayer model."""

import abc
import argparse
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import pandas as pd

from ..tracing import tracer


@dataclass
class RunReturn:
    output: Any
    other_fields: Dict[str, Any] = field(default_factory=dict)


class OpenlayerModel(abc.ABC):
    """Base class for an Openlayer model."""

    def run_from_cli(self):
        # Create the parser
        parser = argparse.ArgumentParser(description="Run data through a model.")

        # Add the --dataset-path argument
        parser.add_argument(
            "--dataset-path", type=str, required=True, help="Path to the dataset"
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            required=False,
            help="Directory to dump the results in",
        )

        # Parse the arguments
        args = parser.parse_args()

        return self.batch(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
        )

    def batch(self, dataset_path: str, output_dir: str):
        # Load the dataset into a pandas DataFrame
        if dataset_path.endswith(".csv"):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith(".json"):
            df = pd.read_json(dataset_path, orient="records")

        # Call the model's run_batch method, passing in the DataFrame
        output_df, config = self.run_batch_from_df(df)
        self.write_output_to_directory(output_df, config, output_dir)

    def run_batch_from_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Function that runs the model and returns the result."""
        # Ensure the 'output' column exists
        if "output" not in df.columns:
            df["output"] = None

        # Get the signature of the 'run' method
        run_signature = inspect.signature(self.run)

        for index, row in df.iterrows():
            # Filter row_dict to only include keys that are valid parameters
            # for the 'run' method
            row_dict = row.to_dict()
            filtered_kwargs = {
                k: v for k, v in row_dict.items() if k in run_signature.parameters
            }

            # Call the run method with filtered kwargs
            output = self.run(**filtered_kwargs)

            df.at[index, "output"] = output.output

            for k, v in output.other_fields.items():
                if k not in df.columns:
                    df[k] = None
                df.at[index, k] = v

            trace = tracer.get_current_trace()
            if trace:
                steps = trace.to_dict()
                df.at[index, "steps"] = steps
                # also need cost, latency, tokens, timestamp

        config = {}
        config["outputColumnName"] = "output"
        config["inputVariableNames"] = list(run_signature.parameters.keys())
        config["metadata"] = {
            "output_timestamp": time.time(),
        }

        # pull the config info from trace if it exists, otherwise manually construct it
        # with the bare minimum
        # costColumnName, latencyColumnName, numOfTokenColumnName, timestampColumnName

        return df, config

    def write_output_to_directory(self, output_df, config, output_dir, fmt="json"):
        """
        Writes the output DataFrame to a file in the specified directory based on the
        given format.

        :param output_df: DataFrame to write.
        :param output_dir: Directory where the output file will be saved.
        :param fmt: Format of the output file ('csv' or 'json').
        """
        os.makedirs(
            output_dir, exist_ok=True
        )  # Create the directory if it doesn't exist

        # Determine the filename based on the dataset name and format
        filename = f"dataset.{fmt}"
        output_path = os.path.join(output_dir, filename)

        # Write the config to a json file
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        # Write the DataFrame to the file based on the specified format
        if fmt == "csv":
            output_df.to_csv(output_path, index=False)
        elif fmt == "json":
            output_df.to_json(output_path, orient="records", indent=4)
        else:
            raise ValueError("Unsupported format. Please choose 'csv' or 'json'.")

        print(f"Output written to {output_path}")

    @abc.abstractmethod
    def run(self, **kwargs) -> RunReturn:
        """Function that runs the model and returns the result."""
        pass
