.. _api.upload:

========================
Upload models / datasets
========================
.. currentmodule:: unboxapi

Client constructor 
------------------
.. autosummary:: 
   :toctree: api/
   :template: class.rst

   UnboxClient

Upload functions
----------------
.. autosummary:: 
   :toctree: api/
   :template: class.rst
   
   UnboxClient.add_model   
   UnboxClient.add_dataset
   UnboxClient.add_dataframe

Model types
-----------
.. autosummary:: 
   :toctree: api/
   :template: class.rst

   ModelType
   ModelType.fasttext
   ModelType.sklearn
   ModelType.pytorch
   ModelType.tensorflow
   ModelType.transformers
   ModelType.keras
   ModelType.rasa
   ModelType.custom

Task types
-----------
.. autosummary:: 
   :toctree: api/
   :template: class.rst
   
   TaskType
   TaskType.TextClassification
   TaskType.TabularClassification
   TaskType.TabularRegression

Objects
-------
.. autosummary:: 
   :toctree: api/
   :template: class.rst
   
   Model
   Model.to_dict
   Dataset
   Dataset.to_dict
