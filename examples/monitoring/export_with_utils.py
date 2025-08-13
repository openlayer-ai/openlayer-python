#!/usr/bin/env python3
"""
Example using the high-level utility functions for exporting inference pipeline data.

This example demonstrates the easiest way to export data using the utility functions
that handle all the complexity of polling and error handling.
"""

import os
import asyncio
from datetime import datetime, timedelta

import openlayer
from openlayer.lib.data import (
    export_inference_pipeline_data,
    export_inference_pipeline_data_async,
    get_download_url,
    get_download_url_async,
)


def simple_export_example():
    """Simple synchronous export using utility functions."""
    print("=== Simple Export with Utility Functions ===")
    
    # Get configuration from environment
    pipeline_id = os.getenv("OPENLAYER_PIPELINE_ID")
    api_key = os.getenv("OPENLAYER_API_KEY")

    if not pipeline_id or not api_key:
        print("Missing OPENLAYER_PIPELINE_ID or OPENLAYER_API_KEY environment variables")
        return

    # Initialize client
    client = openlayer.Openlayer(api_key=api_key)

    # Define export timeframe (last 24 hours)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)

    print(f"Exporting data from {start_time} to {end_time}")

    try:
        # Export data (this handles all the polling automatically)
        storage_uri = export_inference_pipeline_data(
            client=client,
            inference_pipeline_id=pipeline_id,
            start_datetime=start_time,
            end_datetime=end_time,
            format_type="json",
            timeout_seconds=300,  # 5 minutes timeout
        )

        print(f"✅ Export completed! Storage URI: {storage_uri}")

        # Get download URL
        download_url = get_download_url(client, storage_uri)
        print(f"📥 Download URL: {download_url}")

    except Exception as e:
        print(f"❌ Export failed: {e}")


async def async_export_example():
    """Asynchronous export using utility functions."""
    print("\n=== Async Export with Utility Functions ===")
    
    # Get configuration from environment
    pipeline_id = os.getenv("OPENLAYER_PIPELINE_ID")
    api_key = os.getenv("OPENLAYER_API_KEY")

    if not pipeline_id or not api_key:
        print("Missing OPENLAYER_PIPELINE_ID or OPENLAYER_API_KEY environment variables")
        return

    # Use async context manager
    async with openlayer.AsyncOpenlayer(api_key=api_key) as client:
        # Define export timeframe (last week)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)

        print(f"[Async] Exporting data from {start_time} to {end_time}")

        try:
            # Export data asynchronously
            storage_uri = await export_inference_pipeline_data_async(
                client=client,
                inference_pipeline_id=pipeline_id,
                start_datetime=start_time,
                end_datetime=end_time,
                format_type="csv",  # Try CSV format
                timeout_seconds=600,  # 10 minutes timeout for larger export
            )

            print(f"✅ [Async] Export completed! Storage URI: {storage_uri}")

            # Get download URL asynchronously
            download_url = await get_download_url_async(client, storage_uri)
            print(f"📥 [Async] Download URL: {download_url}")

        except Exception as e:
            print(f"❌ [Async] Export failed: {e}")


def multiple_exports_example():
    """Example of exporting multiple date ranges."""
    print("\n=== Multiple Exports Example ===")
    
    # Get configuration from environment
    pipeline_id = os.getenv("OPENLAYER_PIPELINE_ID")
    api_key = os.getenv("OPENLAYER_API_KEY")

    if not pipeline_id or not api_key:
        print("Missing OPENLAYER_PIPELINE_ID or OPENLAYER_API_KEY environment variables")
        return

    client = openlayer.Openlayer(api_key=api_key)

    # Export data for each day of the last week
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for i in range(7):
        start_time = base_date - timedelta(days=i+1)
        end_time = base_date - timedelta(days=i)
        
        print(f"Exporting day {i+1}: {start_time.date()}")

        try:
            storage_uri = export_inference_pipeline_data(
                client=client,
                inference_pipeline_id=pipeline_id,
                start_datetime=start_time,
                end_datetime=end_time,
                format_type="json",
                timeout_seconds=120,  # Shorter timeout for smaller exports
            )

            print(f"  ✅ Day {i+1} exported: {storage_uri}")

        except Exception as e:
            print(f"  ❌ Day {i+1} failed: {e}")


def main():
    """Run all examples."""
    print("🚀 Starting inference pipeline data export examples...")
    
    # Run synchronous example
    simple_export_example()
    
    # Run asynchronous example
    asyncio.run(async_export_example())
    
    # Run multiple exports example
    multiple_exports_example()
    
    print("\n🎉 All examples completed!")


if __name__ == "__main__":
    main()
