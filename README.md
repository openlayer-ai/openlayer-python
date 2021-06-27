# Unbox AI | Python SDK

## Installation

```console
$ pip install -e .
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

# Package a model as a bento service and upload to Firebase
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
# Upload a dataset to Firebase
client.add_dataset(
    file_path='path/to/dataset.csv', 
    class_names=['negative', 'positive'], # Notice it matches the model class names
    label_column_name='polarity',
    text_column_name='text',
    name='My First Dataset',
    description='My sentiment analysis validation dataset',
)

# Upload a pandas data frame to Firebase
client.add_dataframe(
    df=dataframe,
    class_names=['negative', 'positive'], # Notice it matches the model class names
    label_column_name='polarity',
    text_column_name='text',
    name='My Second Dataset',
    description='My sentiment analysis validation pandas dataframe',
)
```