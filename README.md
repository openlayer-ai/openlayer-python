# Unbox AI | Python SDK

## Installation

```console
$ pip install -e .
```

## Usage

```python
import unboxapi
client = unboxapi.UnboxClient(email='you@domain.com', password='your_password')

# Package a model as a bento service and upload to Firebase
client.add_model(function=predict_function, model=any_model)

# Upload a dataset to Firebase
client.add_dataset(file_path='path/to/dataset.csv', name='dataset_name')

# Upload a pandas data frame to Firebase
client.add_dataframe(df='dataframe_object', name='dataset_name')
```
