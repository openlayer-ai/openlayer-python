"""Base class for an Openlayer model."""

import os
import abc
import json
import time
import inspect
import argparse
from typing import Any, Dict, Tuple
from dataclasses import field, dataclass

import pandas as pd

from ..tracing import tracer


@dataclass
class RunReturn:
    """The return type of the `run` method in the Openlayer model."""

    output: Any
    """The output of the model."""

    other_fields: Dict[str, Any] = field(default_factory=dict)
    """Any other fields that you want to log."""


class OpenlayerModel(abc.ABC):
    """Interface for the Openlayer model.

    Your model's class should inherit from this class and implement either:
    -  the `run` method (which takes a single row of data as input and returns
    a `RunReturn` object)
    - `run_batch_from_df` method (which takes a pandas DataFrame as input and returns
    a tuple of a DataFrame and a config dict).

    It is more conventional to implement the `run` method.

    Refer to Openlayer's templates for examples of how to implement this class.
    """

    def run_from_cli(self) -> None:
        """Run the model from the command line."""
        parser = argparse.ArgumentParser(description="Run data through a model.")
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

    def batch(self, dataset_path: str, output_dir: str) -> None:
        """Reads the dataset from a file and runs the model on it."""
        # Load the dataset into a pandas DataFrame
        fmt = "csv"
        if dataset_path.endswith(".csv"):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith(".json"):
            df = pd.read_json(dataset_path, orient="records")
            fmt = "json"

        # Call the model's run_batch method, passing in the DataFrame
        output_df, config = self.run_batch_from_df(df)
        self.write_output_to_directory(output_df, config, output_dir, fmt)

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
                processed_trace, _ = tracer.post_process_trace(trace_obj=trace)
                df.at[index, "steps"] = trace.to_dict()
                if "latency" in processed_trace:
                    df.at[index, "latency"] = processed_trace["latency"]
                if "cost" in processed_trace:
                    df.at[index, "cost"] = processed_trace["cost"]
                if "tokens" in processed_trace:
                    df.at[index, "tokens"] = processed_trace["tokens"]
                if "context" in processed_trace:
                    # Convert the context list to a string to avoid pandas issues
                    df.at[index, "context"] = json.dumps(processed_trace["context"])

        config = {
            "outputColumnName": "output",
            "inputVariableNames": list(run_signature.parameters.keys()),
            "metadata": {
                "output_timestamp": time.time(),
            },
        }

        if "latency" in df.columns:
            config["latencyColumnName"] = "latency"
        if "cost" in df.columns:
            config["costColumnName"] = "cost"
        if "tokens" in df.columns:
            config["numOfTokenColumnName"] = "tokens"
        if "context" in df.columns:
            config["contextColumnName"] = "context"

        return df, config

    def write_output_to_directory(
        self,
        output_df: pd.DataFrame,
        config: Dict[str, Any],
        output_dir: str,
        fmt: str = "json",
    ):
        """Writes the output DataFrame to a file in the specified directory based on the
        given format.
        """
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

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

        print(f"Output written to {output_path}")  # noqa: T201

    @abc.abstractmethod
    def run(self, **kwargs) -> RunReturn:
        """Function that runs the model and returns the result."""
        pass
