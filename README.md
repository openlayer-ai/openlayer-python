# Unbox AI | Python SDK

## Installation

```console
pip install -e .
```

## Usage

```python
import unboxapi

client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')
```

## Models

```python
from unboxapi.models import ModelType

# Predict function
def predict(model, text_list):
    return model.predict(text_list)

# Package your model and upload it to Unbox
client.add_model(
    function=predict,
    model=model,
    model_type=ModelType.sklearn,
    class_names=['negative', 'positive'],
    name='My First Model',
    description='Sentiment analyzer for tweets',
    requirements_txt_file='./requirements.txt',
    **kwargs # specify additional kwargs for your predict function
)
```

## Datasets

```python
# Upload your dataset csv to Unbox
client.add_dataset(
    file_path='path/to/dataset.csv', 
    class_names=['negative', 'positive'], # Notice it matches the model class names
    label_column_name='polarity',
    text_column_name='text',
    name='My First Dataset',
    description='My sentiment analysis validation dataset',
)

# Alternatively, upload your pandas dataframe to Unbox
client.add_dataframe(
    df=dataframe,
    class_names=['negative', 'positive'], # Notice it matches the model class names
    label_column_name='polarity',
    text_column_name='text',
    name='My Second Dataset',
    description='My sentiment analysis validation pandas dataframe',
)
```

## Customer Onboarding

When creating a wheel for customers, make sure the following global variables are set as appropriate below:

In `__init__.py`

```python
DEPLOYMENT = DeploymentType.AWS # If using AWS
DEPLOYMENT = DeploymentType.ONPREM # If using local trial
```

In `api.py`

```python
UNBOX_ENDPOINT = "https://api-staging.unbox.ai/api" # If using AWS
UNBOX_ENDPOINT = "http://localhost:8080/api" # If using local trial
```

1. To create a wheel, run:

```bash
python setup.py bdist_wheel
```

The file should be here: `./dist/unboxapi-{version}-py3-none-any.whl`.

2. Select appropriate sample notebooks from `examples/` and move them into a new folder. Zip that folder and send it over Slack to customers during onboarding
