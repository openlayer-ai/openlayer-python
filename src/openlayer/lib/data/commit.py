"""Pushes a commit to the Openlayer platform."""

import os
import tarfile
import tempfile
import time
from typing import Optional


from ... import Openlayer
from . import StorageType, _upload
from ...types.commit_retrieve_response import CommitRetrieveResponse


def push(
    client: Openlayer,
    directory: str,
    project_id: str,
    message: str = "New commit",
    storage_type: Optional[StorageType] = None,
    wait_for_completion: bool = False,
    verbose: bool = False,
) -> Optional[CommitRetrieveResponse]:
    """Push a new commit to the Openlayer platform.

    This is equivalent to running `openlayer push` from the Openlayer CLI.

    If `wait_for_completion` is True, the function will wait for the commit to be
    completed and return the commit object.

    Args:
        client: The Openlayer client.
        directory: The directory to push.
        project_id: The id of the project to push to.
        message: The commit message.
        storage_type: The storage type to use.
        wait_for_completion: Whether to wait for the commit to be completed.
        verbose: Whether to print verbose output.

    Returns:
        The commit object if `wait_for_completion` is True, otherwise None.
    """
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
    commit = client.projects.commits.create(
        project_id=project_id,
        commit={"message": message, "source": "cli"},
        storage_uri=presigned_url_response.storage_uri,
    )

    if wait_for_completion:
        return wait_for_commit_completion(
            client=client,
            project_version_id=commit.id,
            verbose=verbose,
        )

    return None


def wait_for_commit_completion(
    client: Openlayer, project_version_id: str, verbose: bool = True
) -> CommitRetrieveResponse:
    """Wait for a commit to be processed by the Openlayer platform.

    Waits until the commit status is "completed" or "failed".

    Args:
        client: The Openlayer client.
        project_version_id: The id of the project version (commit) to wait for.
        verbose: Whether to print verbose output.

    Returns:
        The commit object.
    """
    while True:
        commit = client.commits.retrieve(project_version_id=project_version_id)
        if commit.status == "completed":
            if verbose:
                print(f"Commit {project_version_id} completed successfully.")
            return commit
        elif commit.status == "failed":
            raise Exception(
                f"Commit {project_version_id} failed with status message:"
                f" {commit.status_message}"
            )
        else:
            if verbose:
                print(
                    f"Commit {project_version_id} is still processing (status:"
                    f" {commit.status})..."
                )
            time.sleep(1)
