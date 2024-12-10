"""Upload reference datasets to the Openlayer platform."""

import os
import tarfile
import tempfile
import time
from typing import Optional

import pandas as pd

from ... import Openlayer
from ..._utils import maybe_transform
from ...types.inference_pipelines import data_stream_params
from .. import utils
from . import StorageType, _upload


def upload_reference_dataframe(
    client: Openlayer,
    inference_pipeline_id: str,
    dataset_df: pd.DataFrame,
    config: data_stream_params.Config,
    storage_type: Optional[StorageType] = None,
) -> None:
    """Uploads a reference dataset to the Openlayer platform."""
    uploader = _upload.Uploader(client, storage_type)
    object_name = f"reference_dataset_{time.time()}_{inference_pipeline_id}.tar.gz"

    # Fetch presigned url
    presigned_url_response = client.storage.presigned_url.create(
        object_name=object_name,
    )

    # Write dataset and config to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_file_path = f"{tmp_dir}/dataset.csv"
        dataset_df.to_csv(temp_file_path, index=False)

        # Copy relevant files to tmp dir
        config["label"] = "reference"
        utils.write_yaml(
            maybe_transform(config, data_stream_params.Config),
            f"{tmp_dir}/dataset_config.yaml",
        )

        tar_file_path = os.path.join(tmp_dir, object_name)
        with tarfile.open(tar_file_path, mode="w:gz") as tar:
            tar.add(tmp_dir, arcname=os.path.basename("reference_dataset"))

        # Upload to storage
        uploader.upload(
            file_path=tar_file_path,
            object_name=object_name,
            presigned_url_response=presigned_url_response,
        )

    # Notify the backend
    client.inference_pipelines.update(
        inference_pipeline_id=inference_pipeline_id,
        reference_dataset_uri=presigned_url_response.storage_uri,
    )
