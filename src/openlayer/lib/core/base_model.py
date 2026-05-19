"""Base class for an Openlayer model."""

import os
import abc
import json
import time
import asyncio
import inspect
import argparse
from typing import Any, Dict, List, Tuple, Optional
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

    ``run`` may be defined as either ``def run`` (called sequentially per row)
    or ``async def run``. When ``run`` is async, ``run_batch_from_df`` will drive
    rows concurrently with ``asyncio.gather``; pass ``max_workers > 1`` to enable
    concurrent execution. Use async-native I/O (``httpx``, ``openai-async``, etc.)
    inside an async ``run`` to actually benefit from concurrency.

    Refer to Openlayer's templates for examples of how to implement this class.
    """

    custom_args: dict = {}

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
        parser.add_argument(
            "--custom-args",
            type=str,
            required=False,
            help="Custom arguments in format 'key1=value1,key2=value2'",
        )
        parser.add_argument(
            "--max-workers",
            type=int,
            default=None,
            help=(
                "Max concurrent rows when run() is async. "
                "Defaults to 4 for async run, 1 for sync run."
            ),
        )

        # Parse the arguments
        args = parser.parse_args()

        # Parse custom arguments string
        custom_args = {}
        if args.custom_args:
            pairs = args.custom_args.split(",")
            for pair in pairs:
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    custom_args[key] = value
        self.custom_args = custom_args

        return self.batch(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
        )

    def batch(
        self, dataset_path: str, output_dir: str, max_workers: Optional[int] = None
    ) -> None:
        """Reads the dataset from a file and runs the model on it."""
        # Load the dataset into a pandas DataFrame
        fmt = "csv"
        if dataset_path.endswith(".csv"):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith(".json"):
            df = pd.read_json(dataset_path, orient="records")
            fmt = "json"
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")

        # Call the model's run_batch method, passing in the DataFrame
        output_df, config = self.run_batch_from_df(df, max_workers=max_workers)
        self.write_output_to_directory(output_df, config, output_dir, fmt)

    def run_batch_from_df(
        self, df: pd.DataFrame, max_workers: Optional[int] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """Function that runs the model and returns the result.

        If ``run`` is defined as ``async def run(...)``, rows are dispatched
        concurrently with ``asyncio.gather`` gated by ``asyncio.Semaphore(max_workers)``.
        ``max_workers`` defaults to 4 for an async ``run`` (writing `async def`
        is the opt-in signal that interleaving is safe). For a synchronous
        ``run``, rows are processed sequentially and ``max_workers`` must be 1.

        A row's exception propagates and aborts the batch. For the async path,
        ``asyncio.gather`` cancels in-flight siblings before re-raising.
        """
        run_signature = inspect.signature(self.run)
        valid_params = set(run_signature.parameters)
        is_async = inspect.iscoroutinefunction(self.run)

        if max_workers is None:
            max_workers = 4 if is_async else 1
        elif max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        if max_workers > 1 and not is_async:
            raise ValueError(
                "max_workers > 1 requires an async `run` method. "
                "Define `run` as `async def run(self, ...)` to enable "
                "concurrent execution."
            )

        for col in ("output", "steps", "latency", "cost", "tokens", "context"):
            if col not in df.columns:
                df[col] = None

        rows = [
            (
                idx,
                {k: v for k, v in row.to_dict().items() if k in valid_params},
            )
            for idx, row in df.iterrows()
        ]

        if is_async:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeError(
                    "run_batch_from_df was called from inside a running event "
                    "loop. Call `await self._run_rows_async(...)` directly "
                    "from async code."
                )
            results = asyncio.run(self._run_rows_async(rows, max_workers))
        else:
            results = [
                (idx, self.run(**kwargs), tracer.get_current_trace())
                for idx, kwargs in rows
            ]

        for index, output, trace in results:
            self._apply_row_result(df, index, output, trace)

        return df, self._build_config(run_signature, df)

    async def _run_rows_async(
        self,
        rows: List[Tuple[Any, Dict[str, Any]]],
        max_workers: int,
    ) -> List[Tuple[Any, RunReturn, Optional[Any]]]:
        """Drive an async ``run`` over all rows with bounded concurrency.

        The first row to raise causes ``asyncio.gather`` to cancel in-flight
        siblings and re-raise the original exception.
        """
        sem = asyncio.Semaphore(max_workers)

        async def _one(index: Any, kwargs: Dict[str, Any]):
            async with sem:
                output = await self.run(**kwargs)
                return index, output, tracer.get_current_trace()

        return await asyncio.gather(*(_one(i, k) for i, k in rows))

    def _apply_row_result(
        self,
        df: pd.DataFrame,
        index: Any,
        output: RunReturn,
        trace: Optional[Any],
    ) -> None:
        """Write a single row's output and trace fields into ``df`` in place."""
        df.at[index, "output"] = output.output

        for k, v in output.other_fields.items():
            if k not in df.columns:
                df[k] = None
            df.at[index, k] = v

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
                df.at[index, "context"] = processed_trace["context"]

    def _build_config(
        self, run_signature: inspect.Signature, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Build the config dict returned alongside the output DataFrame."""
        config: Dict[str, Any] = {
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

        for k, v in self.custom_args.items():
            config["metadata"][k] = v

        return config

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
