"""
Utility functions for exporting data from inference pipelines.

This module provides high-level utility functions that encapsulate the export workflow,
making it easier for users to export inference pipeline data without dealing with 
the low-level API details.
"""

import time
from datetime import datetime
from typing import Optional, Literal

from ... import Openlayer, AsyncOpenlayer
from ...types.inference_pipelines.export_data_response import ExportDataResponse
from ...types.inference_pipelines.task_status_response import TaskStatusResponse


def export_inference_pipeline_data(
    client: Openlayer,
    inference_pipeline_id: str,
    start_datetime: datetime,
    end_datetime: datetime,
    format_type: Literal["json", "csv"] = "json",
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 5,
) -> str:
    """
    Export data from an inference pipeline and wait for completion.

    This is a high-level utility function that handles the entire export workflow:
    1. Initiates the export request
    2. Polls for task completion
    3. Returns the storage URI when ready

    Args:
        client: The Openlayer client instance
        inference_pipeline_id: The ID of the inference pipeline to export data from
        start_datetime: Start datetime for the export range
        end_datetime: End datetime for the export range
        format_type: Export format, either "json" or "csv"
        timeout_seconds: Maximum time to wait for export completion (default: 5 minutes)
        poll_interval_seconds: Time between status checks (default: 5 seconds)

    Returns:
        The storage URI of the exported data

    Raises:
        ValueError: If the export task fails
        TimeoutError: If the task doesn't complete within the timeout period
        Exception: For other API errors

    Example:
        ```python
        import openlayer
        from datetime import datetime, timedelta
        from openlayer.lib.data.export_utils import export_inference_pipeline_data

        client = openlayer.Openlayer()
        
        # Export last 24 hours of data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
        storage_uri = export_inference_pipeline_data(
            client=client,
            inference_pipeline_id="your-pipeline-id",
            start_datetime=start_time,
            end_datetime=end_time,
            format_type="json"
        )
        
        print(f"Data exported to: {storage_uri}")
        ```
    """
    # Convert datetime objects to Unix timestamps
    start_timestamp = int(start_datetime.timestamp())
    end_timestamp = int(end_datetime.timestamp())

    # Start the export task
    export_response = client.inference_pipelines.export_data(
        inference_pipeline_id=inference_pipeline_id,
        start=start_timestamp,
        end=end_timestamp,
        fmt=format_type,
    )

    # Poll for task completion
    max_attempts = timeout_seconds // poll_interval_seconds
    attempt = 0

    while attempt < max_attempts:
        task_status = client.inference_pipelines.get_task_status(
            task_result_url=export_response.task_result_url
        )

        if task_status.complete:
            if task_status.error:
                raise ValueError(f"Export task failed: {task_status.error}")

            if task_status.outputs and task_status.outputs.storage_uri:
                return task_status.outputs.storage_uri
            else:
                raise ValueError("Export completed but no storage URI available")

        time.sleep(poll_interval_seconds)
        attempt += 1

    raise TimeoutError(
        f"Export task did not complete within {timeout_seconds} seconds"
    )


async def export_inference_pipeline_data_async(
    client: AsyncOpenlayer,
    inference_pipeline_id: str,
    start_datetime: datetime,
    end_datetime: datetime,
    format_type: Literal["json", "csv"] = "json",
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 5,
) -> str:
    """
    Asynchronously export data from an inference pipeline and wait for completion.

    This is the async version of export_inference_pipeline_data.

    Args:
        client: The AsyncOpenlayer client instance
        inference_pipeline_id: The ID of the inference pipeline to export data from
        start_datetime: Start datetime for the export range
        end_datetime: End datetime for the export range
        format_type: Export format, either "json" or "csv"
        timeout_seconds: Maximum time to wait for export completion (default: 5 minutes)
        poll_interval_seconds: Time between status checks (default: 5 seconds)

    Returns:
        The storage URI of the exported data

    Raises:
        ValueError: If the export task fails
        TimeoutError: If the task doesn't complete within the timeout period
        Exception: For other API errors

    Example:
        ```python
        import asyncio
        import openlayer
        from datetime import datetime, timedelta
        from openlayer.lib.data.export_utils import export_inference_pipeline_data_async

        async def main():
            async with openlayer.AsyncOpenlayer() as client:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=1)
                
                storage_uri = await export_inference_pipeline_data_async(
                    client=client,
                    inference_pipeline_id="your-pipeline-id",
                    start_datetime=start_time,
                    end_datetime=end_time,
                    format_type="json"
                )
                
                print(f"Data exported to: {storage_uri}")

        asyncio.run(main())
        ```
    """
    import asyncio

    # Convert datetime objects to Unix timestamps
    start_timestamp = int(start_datetime.timestamp())
    end_timestamp = int(end_datetime.timestamp())

    # Start the export task
    export_response = await client.inference_pipelines.export_data(
        inference_pipeline_id=inference_pipeline_id,
        start=start_timestamp,
        end=end_timestamp,
        fmt=format_type,
    )

    # Poll for task completion
    max_attempts = timeout_seconds // poll_interval_seconds
    attempt = 0

    while attempt < max_attempts:
        task_status = await client.inference_pipelines.get_task_status(
            task_result_url=export_response.task_result_url
        )

        if task_status.complete:
            if task_status.error:
                raise ValueError(f"Export task failed: {task_status.error}")

            if task_status.outputs and task_status.outputs.storage_uri:
                return task_status.outputs.storage_uri
            else:
                raise ValueError("Export completed but no storage URI available")

        await asyncio.sleep(poll_interval_seconds)
        attempt += 1

    raise TimeoutError(
        f"Export task did not complete within {timeout_seconds} seconds"
    )


def get_download_url(
    client: Openlayer,
    storage_uri: str,
) -> str:
    """
    Get a presigned download URL for exported data.

    Args:
        client: The Openlayer client instance
        storage_uri: The storage URI returned from an export operation

    Returns:
        The presigned download URL

    Example:
        ```python
        import openlayer
        from openlayer.lib.data.export_utils import export_inference_pipeline_data, get_download_url

        client = openlayer.Openlayer()
        
        # Export data
        storage_uri = export_inference_pipeline_data(...)
        
        # Get download URL
        download_url = get_download_url(client, storage_uri)
        print(f"Download your data from: {download_url}")
        ```
    """
    presigned_response = client.storage.presigned_url.create(
        object_name=storage_uri
    )
    return presigned_response.download_url


async def get_download_url_async(
    client: AsyncOpenlayer,
    storage_uri: str,
) -> str:
    """
    Asynchronously get a presigned download URL for exported data.

    Args:
        client: The AsyncOpenlayer client instance
        storage_uri: The storage URI returned from an export operation

    Returns:
        The presigned download URL

    Example:
        ```python
        import asyncio
        import openlayer
        from openlayer.lib.data.export_utils import export_inference_pipeline_data_async, get_download_url_async

        async def main():
            async with openlayer.AsyncOpenlayer() as client:
                # Export data
                storage_uri = await export_inference_pipeline_data_async(...)
                
                # Get download URL
                download_url = await get_download_url_async(client, storage_uri)
                print(f"Download your data from: {download_url}")

        asyncio.run(main())
        ```
    """
    presigned_response = await client.storage.presigned_url.create(
        object_name=storage_uri
    )
    return presigned_response.download_url

