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
   OpenlayerClient.create_or_load_project

Add models and datasets
-----------------------
.. autosummary:: 
   :toctree: api/
   :template: class.rst
   
   OpenlayerClient.add_model   
   OpenlayerClient.add_dataset
   OpenlayerClient.add_dataframe
   OpenlayerClient.add_baseline_model

Version control flow
--------------------
.. autosummary:: 
   :toctree: api/
   :template: class.rst
   
   OpenlayerClient.commit   
   OpenlayerClient.push
   OpenlayerClient.status
   OpenlayerClient.restore

Dataset / Task types
--------------------
.. autosummary:: 
   :toctree: api/

   DatasetType
   TaskType

