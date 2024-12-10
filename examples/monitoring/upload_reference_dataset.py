import os

import pandas as pd
from openlayer import Openlayer
from openlayer.lib import data
from openlayer.types.inference_pipelines import data_stream_params

os.environ["OPENLAYER_API_KEY"] = "YOUR_API_KEY"
pipeline_id = "YOUR_INFERENCE_PIPELINE_ID"

df = pd.DataFrame(
    {
        "CreditScore": [600],
        "Geography": ["France"],
        "Gender": ["Male"],
        "Age": [40],
        "Tenure": [5],
        "Balance": [100000],
        "NumOfProducts": [1],
        "HasCrCard": [1],
        "IsActiveMember": [1],
        "EstimatedSalary": [50000],
        "AggregateRate": [0.5],
        "Year": [2020],
        "Exited": [0],
    }
)

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
    label_column_name="Exited",
)

data.upload_reference_dataframe(
    client=Openlayer(),
    inference_pipeline_id=pipeline_id,
    dataset_df=df,
    config=config,
)
