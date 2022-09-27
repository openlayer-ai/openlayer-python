.. _api.upload:

========================
Upload models / datasets
========================
.. currentmodule:: openlayer

Client constructor 
------------------
.. autosummary:: 
   :toctree: api/
   :template: class.rst

   OpenlayerClient

Project creation and loading
----------------------------
.. autosummary:: 
   :toctree: api/
   :template: class.rst
   
   OpenlayerClient.create_project
   OpenlayerClient.load_project

Upload functions
----------------
.. autosummary:: 
   :toctree: api/
   :template: class.rst
   
   OpenlayerClient.add_model   
   OpenlayerClient.add_dataset
   OpenlayerClient.add_dataframe

Model / Task types
------------------
.. autosummary:: 
   :toctree: api/

   ModelType
   TaskType

Objects
-------
.. autosummary:: 
   :toctree: api/
   :template: class.rst
   
   Model
   Dataset
   Model.to_dict
   Dataset.to_dict
