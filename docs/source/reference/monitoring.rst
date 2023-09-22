.. _api.monitoring:

==========
Monitoring
==========
.. currentmodule:: openlayer

The monitoring mode of a project helps you keep track of model health in production and
set up alert for when your model is not performing as expected.
You will use the methods described on this page to create an inference pipeline, publish
production data, and upload reference datasets.

To use these methods, you must have:

1. Authenticated, using :obj:`openlayer.OpenlayerClient`

2. Created a project, using :obj:`openlayer.OpenlayerClient.create_project`

**Related guide**: `How to set up monitoring <https://docs.openlayer.com/docs/set-up-monitoring>`_.


Creating and loading inference pipelines
----------------------------------------
The inference pipeline represents a model deployed in production. It is part of an
Openlayer project is what enables the monitoring mode.

.. autosummary::
   :toctree: api/
   :template: class.rst

   Project.create_inference_pipeline
   Project.load_inference_pipeline

Publishing production data
----------------------------
Production data is published to an inference pipeline on the Openlayer platform using
the methods below.

.. autosummary::
   :toctree: api/
   :template: class.rst

   InferencePipeline.publish_batch_data
   InferencePipeline.publish_ground_truths

Uploading reference datasets
----------------------------
Reference datasets can be uploaded to an inference pipeline to enable data drift goals.
The production data will be compared to the reference dataset to measure
drift.

.. autosummary::
   :toctree: api/
   :template: class.rst

   InferencePipeline.upload_reference_dataset
   InferencePipeline.upload_reference_dataframe

