import os

from openlayer import Openlayer

# Let's say we want to stream the following row, which represents a model prediction:
data = {"user_query": "what's the meaning of life?", "output": "42", "tokens": 7, "cost": 0.02, "timestamp": 1620000000}

client = Openlayer(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENLAYER_API_KEY"),
)

# Prepare the config for the data, which depends on your project's task type. In this
# case, we have an LLM project:
from openlayer.types.inference_pipelines import data_stream_params

config = data_stream_params.ConfigLlmData(
    input_variable_names=["user_query"],
    output_column_name="output",
    num_of_token_column_name="tokens",
    cost_column_name="cost",
    timestamp_column_name="timestamp",
    prompt=[{"role": "user", "content": "{{ user_query }}"}],
)


data_stream_response = client.inference_pipelines.data.stream(
    id="YOUR_INFERENCE_PIPELINE_ID",
    rows=[data],
    config=config,
)
