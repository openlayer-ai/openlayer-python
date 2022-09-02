# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

* Make project `description` optional.

## [0.3.0]

### Added

* A `Project` helper class.
* A convenience method `create_or_load_project` which loads in a project if it is already created.
* Accepts AZURE as a `DeploymentType`.

### Changed

* Compatibility with Unbox API OpenAPI refactor.
* Models and datasets must be added to projects.
* Deprecates `categorical_features_map` in favor of `categorical_feature_names` for model and dataset uploads.
* Moved `TaskType` attribute from the `Model` level to the `Project` level. Creating a `Project` now requires specifying the `TaskType`.
* Removed `name` from `add_dataset`.
* Changed `description` to `commit_message` from `add_dataset`, `add_dataframe` and `add_model`.
* `requirements_txt_file` no longer optional for model uploads.
* NLP dataset character limit is now 1000 characters.

### Fixed

* More comprehensive model and dataset upload validation.
* Bug with duplicate feature names for NLP datasets if uploading same dataset twice.
* Added `protobuf==3.2.0` to requirements to fix bug with model deployment.

## [0.1.2] - 2022-05-22

### Fixed

* Default Unbox server URL (<https://api.unbox.ai/>).

## [0.1.1] - 2022-05-22

### Added

* Can specify `ModelType.keras`.
* API documentation.

### Changed

* Unbox server URL (<https://api.unbox.ai/>).
* Parameter ordering for `add_model`, `add_dataset`, `add_dataframe` for clarity's sake.

### Fixed

* Fix bug when predict function is a list with numpy objects.
* Better error message if return type isn't a list or numpy array.
* Prevents dataset upload when `label_column_name` is also in `feature_names`.

## [0.1.0] - 2022-04-17

### Added

* Accepts GCP as a `DeploymentType`.
