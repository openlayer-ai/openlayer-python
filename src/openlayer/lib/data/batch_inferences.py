"""Upload a batch of inferences to the Openlayer platform."""

import os
import tarfile
import tempfile
import time
from typing import Optional

import httpx
import pandas as pd

from ... import Openlayer
from ..._utils import maybe_transform
from ...types.inference_pipelines import data_stream_params
from .. import utils
from . import StorageType, _upload


def upload_batch_inferences(
    client: Openlayer,
    inference_pipeline_id: str,
    config: data_stream_params.Config,
    dataset_df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[str] = None,
    storage_type: Optional[StorageType] = None,
    merge: bool = False,
) -> None:
    """Uploads a batch of inferences to the Openlayer platform."""
    if dataset_df is None and dataset_path is None:
        raise ValueError("Either dataset_df or dataset_path must be provided.")
    if dataset_df is not None and dataset_path is not None:
        raise ValueError("Only one of dataset_df or dataset_path should be provided.")

    uploader = _upload.Uploader(client, storage_type)
    object_name = f"batch_data_{time.time()}_{inference_pipeline_id}.tar.gz"

    # Fetch presigned url
    presigned_url_response = client.storage.presigned_url.create(
        object_name=object_name,
    )

    # Write dataset and config to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        if dataset_df is not None:
            temp_file_path = f"{tmp_dir}/dataset.csv"
            dataset_df.to_csv(temp_file_path, index=False)
        else:
            temp_file_path = dataset_path

        # Copy relevant files to tmp dir
        config["label"] = "production"
        utils.write_yaml(
            maybe_transform(config, data_stream_params.Config),
            f"{tmp_dir}/dataset_config.yaml",
        )

        tar_file_path = os.path.join(tmp_dir, object_name)
        with tarfile.open(tar_file_path, mode="w:gz") as tar:
            tar.add(temp_file_path, arcname=os.path.basename("dataset.csv"))
            tar.add(
                f"{tmp_dir}/dataset_config.yaml",
                arcname=os.path.basename("dataset_config.yaml"),
            )

        # Upload to storage
        uploader.upload(
            file_path=tar_file_path,
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
        },
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
