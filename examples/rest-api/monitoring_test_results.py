import os

from openlayer import Openlayer

inference_pipeline_id = "YOUR_OPENLAYER_INFERENCE_PIPELINE_ID_HERE"


client = Openlayer(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENLAYER_API_KEY"),
)
response = client.inference_pipelines.test_results.list(inference_pipeline_id=inference_pipeline_id)

print(response.items)
