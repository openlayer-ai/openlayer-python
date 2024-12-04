"""Module containing the BaseMetric definition for Openlayer."""

from __future__ import annotations

import os
import abc
import json
import argparse
import traceback
from typing import Any, Set, Dict, List, Union, Optional
from dataclasses import field, asdict, dataclass

import pandas as pd


@dataclass
class MetricReturn:
    """The return type of the `run` method in the BaseMetric."""

    value: Optional[Union[float, int, bool]]
    """The value of the metric."""

    unit: Optional[str] = None
    """The unit of the metric."""

    meta: Dict[str, Any] = field(default_factory=dict)
    """Any useful metadata in a JSON serializable dict."""

    error: Optional[str] = None
    """An error message if the metric computation failed."""

    added_cols: Set[str] = field(default_factory=set)
    """Columns added to the dataset."""


@dataclass
class Dataset:
    """A dataset object containing the configuration, data and dataset outputs path."""

    name: str
    """The name of the dataset."""

    config: dict
    """The configuration of the dataset."""

    df: pd.DataFrame
    """The dataset as a pandas DataFrame."""

    output_path: str
    """The path to the dataset outputs."""

    data_format: str
    """The format of the written dataset. E.g. 'csv' or 'json'."""

    added_cols: Set[str] = field(default_factory=set)
    """Columns added to the dataset."""


class MetricRunner:
    """A class to run a list of metrics."""

    def __init__(self):
        self.config_path: str = ""
        self.config: Dict[str, Any] = {}
        self.datasets: List[Dataset] = []
        self.likely_dir: str = ""

    def run_metrics(self, metrics: List[BaseMetric]) -> None:
        """Run a list of metrics."""

        # Parse arguments from the command line
        self._parse_args()

        # Load the openlayer.json file
        self._load_openlayer_json()

        # Load the datasets from the openlayer.json file
        self._load_datasets()

        # Compute the metric values
        self._compute_metrics(metrics)

        # Write the updated datasets to the output location
        self._write_updated_datasets_to_output()

    def _parse_args(self) -> None:
        parser = argparse.ArgumentParser(description="Compute custom metrics.")
        parser.add_argument(
            "--config-path",
            type=str,
            required=False,
            default="",
            help=(
                "The path to your openlayer.json. Uses parent parent dir if not "
                "provided (assuming location is metrics/metric_name/run.py)."
            ),
        )
        parser.add_argument(
            "--dataset",
            type=str,
            required=False,
            default="",
            help="The name of the dataset to compute the metric on. Runs on all " "datasets if not provided.",
        )

        # Parse the arguments
        args = parser.parse_args()
        self.config_path = args.config_path
        self.dataset_name = args.dataset
        self.likely_dir = os.path.dirname(os.path.dirname(os.getcwd()))

    def _load_openlayer_json(self) -> None:
        """Load the openlayer.json file."""

        if not self.config_path:
            openlayer_json_path = os.path.join(self.likely_dir, "openlayer.json")
        else:
            openlayer_json_path = self.config_path

        with open(openlayer_json_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def _load_datasets(self) -> None:
        """Compute the metric from the command line."""

        datasets: List[Dataset] = []

        # Check first for a model. If it exists, use the output of the model
        if "model" in self.config:
            model = self.config["model"]
            datasets_list = self.config["datasets"]
            dataset_names = [dataset["name"] for dataset in datasets_list]
            if self.dataset_name:
                if self.dataset_name not in dataset_names:
                    raise ValueError(f"Dataset {self.dataset_name} not found in the openlayer.json.")
                dataset_names = [self.dataset_name]
            output_directory = model["outputDirectory"]
            # Read the outputs directory for dataset folders. For each, load
            # the config.json and the dataset.json files into a dict and a dataframe

            full_output_dir = os.path.join(self.likely_dir, output_directory)

            for dataset_folder in os.listdir(full_output_dir):
                if dataset_folder not in dataset_names:
                    continue
                dataset_path = os.path.join(full_output_dir, dataset_folder)
                config_path = os.path.join(dataset_path, "config.json")
                with open(config_path, "r", encoding="utf-8") as f:
                    dataset_config = json.load(f)
                    # Merge with the dataset fields from the openlayer.json
                    dataset_dict = next(
                        (item for item in datasets_list if item["name"] == dataset_folder),
                        None,
                    )
                    dataset_config = {**dataset_dict, **dataset_config}

                # Load the dataset into a pandas DataFrame
                if os.path.exists(os.path.join(dataset_path, "dataset.csv")):
                    dataset_df = pd.read_csv(os.path.join(dataset_path, "dataset.csv"))
                    data_format = "csv"
                elif os.path.exists(os.path.join(dataset_path, "dataset.json")):
                    dataset_df = pd.read_json(os.path.join(dataset_path, "dataset.json"), orient="records")
                    data_format = "json"
                else:
                    raise ValueError(f"No dataset found in {dataset_folder}.")

                datasets.append(
                    Dataset(
                        name=dataset_folder,
                        config=dataset_config,
                        df=dataset_df,
                        output_path=dataset_path,
                        data_format=data_format,
                    )
                )
        else:
            raise ValueError("No model found in the openlayer.json file. Cannot compute metric.")

        if not datasets:
            raise ValueError("No datasets found in the openlayer.json file. Cannot compute metric.")

        self.datasets = datasets

    def _compute_metrics(self, metrics: List[BaseMetric]) -> None:
        """Compute the metrics."""
        for metric in metrics:
            metric.compute(self.datasets)

    def _write_updated_datasets_to_output(self) -> None:
        """Write the updated datasets to the output location."""
        for dataset in self.datasets:
            if dataset.added_cols:
                self._write_updated_dataset_to_output(dataset)

    def _write_updated_dataset_to_output(self, dataset: Dataset) -> None:
        """Write the updated dataset to the output location."""

        # Determine the filename based on the dataset name and format
        filename = f"dataset.{dataset.data_format}"
        data_path = os.path.join(dataset.output_path, filename)

        # TODO: Read the dataset again and only include the added columns

        # Write the DataFrame to the file based on the specified format
        if dataset.data_format == "csv":
            dataset.df.to_csv(data_path, index=False)
        elif dataset.data_format == "json":
            dataset.df.to_json(data_path, orient="records", indent=4, index=False)
        else:
            raise ValueError("Unsupported format. Please choose 'csv' or 'json'.")

        print(f"Updated dataset {dataset.name} written to {data_path}")


class BaseMetric(abc.ABC):
    """Interface for the Base metric.

    Your metric's class should inherit from this class and implement the compute method.
    """

    def get_key(self) -> str:
        """Return the key of the metric. This should correspond to the folder name."""
        return os.path.basename(os.getcwd())

    @property
    def key(self) -> str:
        return self.get_key()

    def compute(self, datasets: List[Dataset]) -> None:
        """Compute the metric on the model outputs."""
        for dataset in datasets:
            # Check if the metric has already been computed
            if os.path.exists(os.path.join(dataset.output_path, "metrics", f"{self.key}.json")):
                print(f"Metric ({self.key}) already computed on {dataset.name}. " "Skipping.")
                continue

            try:
                metric_return = self.compute_on_dataset(dataset)
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error computing metric ({self.key}) on {dataset.name}:")
                print(traceback.format_exc())
                metric_return = MetricReturn(error=str(e), value=None)

            metric_value = metric_return.value
            if metric_return.unit:
                metric_value = f"{metric_value} {metric_return.unit}"
            print(f"Metric ({self.key}) value on {dataset.name}: {metric_value}")

            output_dir = os.path.join(dataset.output_path, "metrics")
            self._write_metric_return_to_file(metric_return, output_dir)

            # Add the added columns to the dataset
            if metric_return.added_cols:
                dataset.added_cols.update(metric_return.added_cols)

    @abc.abstractmethod
    def compute_on_dataset(self, dataset: Dataset) -> MetricReturn:
        """Compute the metric on a specific dataset."""
        pass

    def _write_metric_return_to_file(self, metric_return: MetricReturn, output_dir: str) -> None:
        """Write the metric return to a file."""

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Turn the metric return to a dict
        metric_return_dict = asdict(metric_return)
        # Convert the set to a list
        metric_return_dict["added_cols"] = list(metric_return.added_cols)

        with open(os.path.join(output_dir, f"{self.key}.json"), "w", encoding="utf-8") as f:
            json.dump(metric_return_dict, f, indent=4)
        print(f"Metric ({self.key}) value written to {output_dir}/{self.key}.json")

    def run(self) -> None:
        """Run the metric."""
        metric_runner = MetricRunner()
        metric_runner.run_metrics([self])
