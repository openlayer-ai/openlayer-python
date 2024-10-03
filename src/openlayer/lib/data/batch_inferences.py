"""Upload a batch of inferences to the Openlayer platform."""

import time
import tempfile
from typing import Optional

import httpx
import pandas as pd
import pyarrow as pa

from . import StorageType, _upload
from ... import Openlayer
from ..._utils import maybe_transform
from ...types.inference_pipelines import data_stream_params
import asyncio


async def upload_batch_inferences_async(
    client: Openlayer,
    inference_pipeline_id: str,
    config: data_stream_params.Config,
    dataset_df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[str] = None,
    storage_type: Optional[StorageType] = None,
    merge: bool = False,
    verbose: bool = False,
) -> None:
    """Uploads a batch of inferences to the Openlayer platform."""
    if dataset_df is None and dataset_path is None:
        raise ValueError("Either dataset_df or dataset_path must be provided.")
    if dataset_df is not None and dataset_path is not None:
        raise ValueError("Only one of dataset_df or dataset_path should be provided.")

    uploader = _upload.Uploader(client, storage_type)
    object_name = f"batch_data_{time.time()}_{inference_pipeline_id}.arrow"

    # Fetch presigned url
    presigned_url_response = client.storage.presigned_url.create(
        object_name=object_name,
    )

    # Write dataset and config to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # If DataFrame is provided, convert it to Arrow Table and write it using IPC
        # writer
        if dataset_df is not None:
            temp_file_path = f"{tmp_dir}/dataset.arrow"
            if verbose:
                print("Converting DataFrame to pyarrow Table...")
            pa_table = pa.Table.from_pandas(dataset_df)
            pa_schema = pa_table.schema

            if verbose:
                print(
                    "Writing Arrow Table using RecordBatchStreamWriter to "
                    f"{temp_file_path}"
                )
            with pa.ipc.RecordBatchStreamWriter(temp_file_path, pa_schema) as writer:
                writer.write_table(pa_table, max_chunksize=16384)
        else:
            object_name = f"batch_data_{time.time()}_{inference_pipeline_id}.csv"
            temp_file_path = dataset_path

        # camelCase the config
        config = maybe_transform(config, data_stream_params.Config)

        # Upload tarball to storage
        if verbose:
            print("Uploading dataset to storage via presigned URL...")
        uploader.upload(
            file_path=temp_file_path,
            object_name=object_name,
            presigned_url_response=presigned_url_response,
        )

    # Notify the backend
    client.post(
        f"/inference-pipelines/{inference_pipeline_id}/data",
        cast_to=httpx.Response,
        body={
            "storageUri": presigned_url_response.storage_uri,
            "performDataMerge": merge,
            "config": config,
        },
    )


def upload_batch_inferences(
    client: Openlayer,
    inference_pipeline_id: str,
    config: data_stream_params.Config,
    dataset_df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[str] = None,
    storage_type: Optional[StorageType] = None,
    merge: bool = False,
    verbose: bool = False,
) -> None:
    asyncio.run(
        upload_batch_inferences_async(
            client,
            inference_pipeline_id,
            config,
            dataset_df,
            dataset_path,
            storage_type,
            merge,
            verbose,
        )
    )


def update_batch_inferences(
    client: Openlayer,
    inference_pipeline_id: str,
    dataset_df: pd.DataFrame,
    config: data_stream_params.Config,
    storage_type: Optional[StorageType] = None,
) -> None:
    """Updates a batch of inferences on the Openlayer platform."""
    if config["inference_id_column_name"] is None:
        raise ValueError("inference_id_column_name must be set in config")
    upload_batch_inferences(
        client=client,
        inference_pipeline_id=inference_pipeline_id,
        dataset_df=dataset_df,
        config=config,
        storage_type=storage_type,
        merge=True,
    )
