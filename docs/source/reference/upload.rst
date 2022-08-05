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

Project creation and loading
----------------------------
.. autosummary:: 
   :toctree: api/
   :template: class.rst
   
   UnboxClient.create_project
   UnboxClient.load_project

Upload functions
----------------
.. autosummary:: 
   :toctree: api/
   :template: class.rst
   
   UnboxClient.add_model   
   UnboxClient.add_dataset
   UnboxClient.add_dataframe

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
