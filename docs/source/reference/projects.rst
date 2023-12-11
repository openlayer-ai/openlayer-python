.. _api.projects:

========
Projects
========
.. currentmodule:: openlayer


A project is the logical unit on the Openlayer platform that houses models, datasets,
and goals. You can create projects for any of the task types defined
in :class:`tasks.TaskType`.

**Related guide**: `How to create and load projects <https://docs.openlayer.com/docs/how-to-guides/creating-and-loading-projects>`_.

Project creation and loading
----------------------------

Create projects on the Openlayer platform or load an existing project.

.. autosummary::
   :toctree: api/
   :template: class.rst

   OpenlayerClient.create_project
   OpenlayerClient.load_project
   OpenlayerClient.create_or_load_project

Project task types
------------------

Each project has a task type, which defines the type of ML problem
that the project is designed to solve.


.. autosummary::
   :toctree: api/

   tasks.TaskType


