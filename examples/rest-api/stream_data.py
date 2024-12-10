import os

from openlayer import Openlayer

client = Openlayer(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENLAYER_API_KEY"),
)

# Let's say we want to stream the following row, which represents a tabular
# classification model prediction, with features and a prediction:
data = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Male",
    "Age": 42,
    "Tenure": 5,
    "Balance": 100000,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000,
    "AggregateRate": 0.5,
    "Year": 2020,
    "Prediction": 1,
}

# Prepare the config for the data, which depends on your project's task type. In this
# case, we have an Tabular Classification project:
from openlayer.types.inference_pipelines import data_stream_params

config = data_stream_params.ConfigTabularClassificationData(
    categorical_feature_names=["Gender", "Geography"],
    class_names=["Retained", "Exited"],
    feature_names=[
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "AggregateRate",
        "Year",
    ],
    predictions_column_name="Prediction",
)

# Now, you can stream the data to the inference pipeline:
data_stream_response = client.inference_pipelines.data.stream(
    inference_pipeline_id="YOUR_INFERENCE_PIPELINE_ID",
    rows=[data],
    config=config,
)
