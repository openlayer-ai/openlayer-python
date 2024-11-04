"""Pushes a commit to the Openlayer platform."""

import os
import tarfile
import tempfile
from typing import Optional


from ... import Openlayer
from . import StorageType, _upload


def push(
    client: Openlayer,
    directory: str,
    project_id: str,
    message: str = "New commit",
    storage_type: Optional[StorageType] = None,
) -> None:
    """Push a new commit to the Openlayer platform.

    This is equivalent to running `openlayer push` from the Openlayer CLI."""
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_file_path = os.path.join(tmp_dir, "bundle.tar")
        with tarfile.open(tar_file_path, mode="w") as tar:
            tar.add(directory, arcname=os.path.basename(directory))

        # Upload tar storage
        uploader = _upload.Uploader(client, storage_type)
        object_name = "bundle.tar"
        presigned_url_response = client.storage.presigned_url.create(
            object_name=object_name,
        )
        uploader.upload(
            file_path=tar_file_path,
            object_name=object_name,
            presigned_url_response=presigned_url_response,
        )

    # Create the project version (commit)
    client.projects.commits.create(
        project_id=project_id,
        commit={"message": message, "source": "cli"},
        storage_uri=presigned_url_response.storage_uri,
    )
