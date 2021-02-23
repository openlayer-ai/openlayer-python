# Unbox AI | Python SDK

## Installation

```console
$ pip install -e .
```

## Usage

```python
import unboxapi
client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

# Package as a bento service and upload to Firebase Storage
client.add(function=predict_function, model=any_model)
```
