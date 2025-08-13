#!/usr/bin/env python3
"""
Example: Export data from inference pipelines using the Openlayer Python SDK.

This example demonstrates how to:
1. Export data from an inference pipeline for a specified time range
2. Poll for task completion
3. Retrieve the presigned URL for downloading the exported data
4. Handle different export formats (JSON, CSV)
5. Work with both synchronous and asynchronous clients

Requirements:
- OPENLAYER_API_KEY environment variable
- OPENLAYER_PIPELINE_ID environment variable
"""

import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import openlayer


def export_pipeline_data_sync(
    pipeline_id: str,
    start_datetime: datetime,
    end_datetime: datetime,
    format_type: str = "json",
    api_key: Optional[str] = None,
) -> str:
    """
    Export data from inference pipeline synchronously.

    Args:
        pipeline_id: The inference pipeline ID
        start_datetime: Start datetime for export range
        end_datetime: End datetime for export range
        format_type: Export format ('json' or 'csv')
        api_key: Optional API key (uses environment variable if not provided)

    Returns:
        The storage URI of the exported data

    Raises:
        ValueError: If the export task fails
        TimeoutError: If the task doesn't complete within the timeout period
    """
    # Initialize the client
    client = openlayer.Openlayer(api_key=api_key)

    # Convert datetime objects to Unix timestamps
    start_timestamp = int(start_datetime.timestamp())
    end_timestamp = int(end_datetime.timestamp())

    print(f"Exporting data from {start_datetime} to {end_datetime}")
    print(f"Time range: {start_timestamp} - {end_timestamp} (Unix timestamps)")
    print(f"Format: {format_type}")

    # Start the export task
    export_response = client.inference_pipelines.export_data(
        inference_pipeline_id=pipeline_id,
        start=start_timestamp,
        end=end_timestamp,
        fmt=format_type,
    )

    print(f"Export task started. Task URL: {export_response.task_result_url}")

    # Poll for task completion
    max_attempts = 60  # Maximum 5 minutes with 5-second intervals
    attempt = 0

    while attempt < max_attempts:
        # Check task status
        task_status = client.inference_pipelines.get_task_status(
            task_result_url=export_response.task_result_url
        )

        print(f"Attempt {attempt + 1}: Task complete = {task_status.complete}")

        if task_status.complete:
            if task_status.error:
                raise ValueError(f"Export task failed: {task_status.error}")

            if task_status.outputs and task_status.outputs.storage_uri:
                print(f"Export completed successfully!")
                print(f"Storage URI: {task_status.outputs.storage_uri}")
                return task_status.outputs.storage_uri
            else:
                raise ValueError("Export completed but no storage URI available")

        # Wait before next attempt
        time.sleep(5)
        attempt += 1

    raise TimeoutError("Export task did not complete within the timeout period")


async def export_pipeline_data_async(
    pipeline_id: str,
    start_datetime: datetime,
    end_datetime: datetime,
    format_type: str = "json",
    api_key: Optional[str] = None,
) -> str:
    """
    Export data from inference pipeline asynchronously.

    Args:
        pipeline_id: The inference pipeline ID
        start_datetime: Start datetime for export range
        end_datetime: End datetime for export range
        format_type: Export format ('json' or 'csv')
        api_key: Optional API key (uses environment variable if not provided)

    Returns:
        The storage URI of the exported data

    Raises:
        ValueError: If the export task fails
        TimeoutError: If the task doesn't complete within the timeout period
    """
    # Initialize the async client
    async with openlayer.AsyncOpenlayer(api_key=api_key) as client:
        # Convert datetime objects to Unix timestamps
        start_timestamp = int(start_datetime.timestamp())
        end_timestamp = int(end_datetime.timestamp())

        print(f"[Async] Exporting data from {start_datetime} to {end_datetime}")
        print(f"[Async] Time range: {start_timestamp} - {end_timestamp} (Unix timestamps)")
        print(f"[Async] Format: {format_type}")

        # Start the export task
        export_response = await client.inference_pipelines.export_data(
            inference_pipeline_id=pipeline_id,
            start=start_timestamp,
            end=end_timestamp,
            fmt=format_type,
        )

        print(f"[Async] Export task started. Task URL: {export_response.task_result_url}")

        # Poll for task completion
        max_attempts = 60  # Maximum 5 minutes with 5-second intervals
        attempt = 0

        while attempt < max_attempts:
            # Check task status
            task_status = await client.inference_pipelines.get_task_status(
                task_result_url=export_response.task_result_url
            )

            print(f"[Async] Attempt {attempt + 1}: Task complete = {task_status.complete}")

            if task_status.complete:
                if task_status.error:
                    raise ValueError(f"Export task failed: {task_status.error}")

                if task_status.outputs and task_status.outputs.storage_uri:
                    print(f"[Async] Export completed successfully!")
                    print(f"[Async] Storage URI: {task_status.outputs.storage_uri}")
                    return task_status.outputs.storage_uri
                else:
                    raise ValueError("Export completed but no storage URI available")

            # Wait before next attempt
            await asyncio.sleep(5)
            attempt += 1

        raise TimeoutError("Export task did not complete within the timeout period")


def get_presigned_url_for_download(storage_uri: str, api_key: Optional[str] = None) -> str:
    """
    Get a presigned URL for downloading the exported data.

    Args:
        storage_uri: The storage URI returned from the export task
        api_key: Optional API key (uses environment variable if not provided)

    Returns:
        The presigned URL for downloading the data
    """
    client = openlayer.Openlayer(api_key=api_key)

    # Get presigned URL for the storage URI
    presigned_response = client.storage.presigned_url.create(
        object_name=storage_uri
    )

    print(f"Presigned URL obtained: {presigned_response.download_url}")
    return presigned_response.download_url


def main():
    """Main function demonstrating export functionality."""
    # Get configuration from environment variables
    pipeline_id = os.getenv("OPENLAYER_PIPELINE_ID")
    api_key = os.getenv("OPENLAYER_API_KEY")

    if not pipeline_id:
        raise ValueError("OPENLAYER_PIPELINE_ID environment variable is required")

    if not api_key:
        raise ValueError("OPENLAYER_API_KEY environment variable is required")

    # Example: Export data from the last 24 hours
    end_datetime = datetime.now()
    start_datetime = end_datetime - timedelta(days=1)

    print("=== Synchronous Export Example ===")
    try:
        # Export as JSON
        storage_uri_json = export_pipeline_data_sync(
            pipeline_id=pipeline_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            format_type="json",
            api_key=api_key,
        )

        # Get presigned URL for download
        download_url_json = get_presigned_url_for_download(
            storage_uri=storage_uri_json,
            api_key=api_key,
        )

        print(f"JSON export download URL: {download_url_json}")

    except Exception as e:
        print(f"Synchronous export failed: {e}")

    print("\n=== CSV Export Example ===")
    try:
        # Export as CSV
        storage_uri_csv = export_pipeline_data_sync(
            pipeline_id=pipeline_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            format_type="csv",
            api_key=api_key,
        )

        # Get presigned URL for download
        download_url_csv = get_presigned_url_for_download(
            storage_uri=storage_uri_csv,
            api_key=api_key,
        )

        print(f"CSV export download URL: {download_url_csv}")

    except Exception as e:
        print(f"CSV export failed: {e}")

    print("\n=== Asynchronous Export Example ===")
    try:
        # Example with async client
        storage_uri_async = asyncio.run(
            export_pipeline_data_async(
                pipeline_id=pipeline_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                format_type="json",
                api_key=api_key,
            )
        )

        # Get presigned URL for download
        download_url_async = get_presigned_url_for_download(
            storage_uri=storage_uri_async,
            api_key=api_key,
        )

        print(f"Async export download URL: {download_url_async}")

    except Exception as e:
        print(f"Asynchronous export failed: {e}")


def example_with_specific_dates():
    """Example with specific date ranges."""
    pipeline_id = os.getenv("OPENLAYER_PIPELINE_ID")
    api_key = os.getenv("OPENLAYER_API_KEY")

    if not pipeline_id or not api_key:
        print("Skipping specific dates example - missing environment variables")
        return

    # Example: Export data for a specific date range
    start_dt = datetime(2024, 1, 1, 0, 0, 0)
    end_dt = datetime(2024, 1, 2, 0, 0, 0)

    print(f"\n=== Specific Date Range Export ===")
    print(f"Exporting data from {start_dt} to {end_dt}")

    try:
        storage_uri = export_pipeline_data_sync(
            pipeline_id=pipeline_id,
            start_datetime=start_dt,
            end_datetime=end_dt,
            format_type="json",
            api_key=api_key,
        )

        download_url = get_presigned_url_for_download(
            storage_uri=storage_uri,
            api_key=api_key,
        )

        print(f"Specific date range export download URL: {download_url}")

    except Exception as e:
        print(f"Specific date range export failed: {e}")


if __name__ == "__main__":
    # Example: current datetime and timestamp conversions
    dt_object = datetime.now()
    timestamp_float = dt_object.timestamp()
    timestamp_integer = int(timestamp_float)

    print(f"Datetime object: {dt_object}")
    print(f"Timestamp (float): {timestamp_float}")
    print(f"Timestamp (integer): {timestamp_integer}")
    print("=" * 50)

    # Run the main examples
    main()

    # Run example with specific dates
    example_with_specific_dates()

    print("\n" + "=" * 50)
    print("Export examples completed!")
