.. _api.development:

===========
Development
===========
.. currentmodule:: openlayer

The development mode of a project helps you as you iterate on your models and datasets.
You will use the methods described on this page to add models and datasets to your
development project,

To use these methods, you must have:

1. Authenticated, using :obj:`openlayer.OpenlayerClient`

2. Created a project, using :obj:`openlayer.OpenlayerClient.create_project`

**Related guide**: `How to upload datasets and models for development <https://docs.openlayer.com/documentation/how-to-guides/upload-datasets-and-models>`_.


Staging area
------------
The upload of models and datasets to a project on Openlayer follows a similar flow
to the one for uploading files to a version control system like Git.

The ``add_*`` methods, add models and datasets to the local staging area.
As you add resources to the staging area, you can check its status using the
``status`` method.


Finally, the resources on the staging area are committed and pushed to the Openlayer
platform using the ``commit`` and ``push`` methods.


Datasets
--------
Datasets stored as Pandas dataframes or csv files can be easily added to a project's
staging area with the methods below.

.. autosummary::
   :toctree: api/
   :template: class.rst

   Project.add_dataset
   Project.add_dataframe

Models
------
Models are added to the staging area using the ``add_model`` method.

.. autosummary::
   :toctree: api/
   :template: class.rst

   Project.add_model

Committing and pushing
----------------------
After adding resources to the staging area, you can commit and push them to Openlayer.

.. autosummary::
   :toctree: api/
   :template: class.rst

   Project.commit
   Project.push

Other methods to interact with the staging area
-----------------------------------------------
Additional methods used to interact with the staging area.

.. autosummary::
   :toctree: api/
   :template: class.rst

   Project.status
   Project.restore
   Project.export

Checking a project version's goal statuses
------------------------------------------
To programatically check the status of a project version's goals, use the
``ProjectVersion`` object, which can be obtained using the ``load_project_version`` method.

.. autosummary::
   :toctree: api/
   :template: class.rst

   ProjectVersion
   OpenlayerClient.load_project_version





