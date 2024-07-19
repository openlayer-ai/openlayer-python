"""Module containing the BaseMetric definition for Openlayer."""

from __future__ import annotations

import abc
import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union, Set

import pandas as pd


@dataclass
class MetricReturn:
    """The return type of the `run` method in the BaseMetric."""

    value: Union[float, int, bool]
    """The value of the metric."""

    unit: Optional[str] = None
    """The unit of the metric."""

    meta: Dict[str, Any] = field(default_factory=dict)
    """Any useful metadata in a JSON serializable dict."""

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
        self.selected_metrics: Optional[List[str]] = None

    def run_metrics(self, metrics: List[BaseMetric]) -> None:
        """Run a list of metrics."""

        # Parse arguments from the command line
        self._parse_args()

        # Load the openlayer.json file
        self._load_openlayer_json()

        # Load the datasets from the openlayer.json file
        self._load_datasets()

        # TODO: Auto-load all the metrics in the current directory

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
            help="The path to your openlayer.json. Uses working dir if not provided.",
        )

        # Parse the arguments
        args = parser.parse_args()
        self.config_path = args.config_path

    def _load_openlayer_json(self) -> None:
        """Load the openlayer.json file."""

        if not self.config_path:
            openlayer_json_path = os.path.join(os.getcwd(), "openlayer.json")
        else:
            openlayer_json_path = self.config_path

        with open(openlayer_json_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Extract selected metrics
        if "metrics" in self.config and "settings" in self.config["metrics"]:
            self.selected_metrics = [
                metric["key"] for metric in self.config["metrics"]["settings"] if metric["selected"]
            ]

    def _load_datasets(self) -> None:
        """Compute the metric from the command line."""

        datasets: List[Dataset] = []

        # Check first for a model. If it exists, use the output of the model
        if "model" in self.config:
            model = self.config["model"]
            datasets_list = self.config["datasets"]
            dataset_names = [dataset["name"] for dataset in datasets_list]
            output_directory = model["outputDirectory"]
            # Read the outputs directory for dataset folders. For each, load
            # the config.json and the dataset.json files into a dict and a dataframe

            for dataset_folder in os.listdir(output_directory):
                if dataset_folder not in dataset_names:
                    continue
                dataset_path = os.path.join(output_directory, dataset_folder)
                config_path = os.path.join(dataset_path, "config.json")
                with open(config_path, "r", encoding="utf-8") as f:
                    dataset_config = json.load(f)

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
            if self.selected_metrics and metric.key not in self.selected_metrics:
                print(f"Skipping metric {metric.key} as it is not a selected metric.")
                continue
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

    @property
    def key(self) -> str:
        """Return the key of the metric."""
        return self.__class__.__name__

    def compute(self, datasets: List[Dataset]) -> None:
        """Compute the metric on the model outputs."""
        for dataset in datasets:
            metric_return = self.compute_on_dataset(dataset)
            metric_value = metric_return.value
            if metric_return.unit:
                metric_value = f"{metric_value} {metric_return.unit}"
            print(f"Metric ({self.key}) value for {dataset.name}: {metric_value}")

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

        with open(os.path.join(output_dir, f"{self.key}.json"), "w", encoding="utf-8") as f:
            json.dump(metric_return_dict, f, indent=4)
        print(f"Metric ({self.key}) value written to {output_dir}/{self.key}.json")
