# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

# Changed

* Models and datasets must be added to projects. Added a `Project` helper class.
* Deprecates `categorical_features_map` in favor of `categorical_feature_names` for model and dataset uploads.

## [0.2.0a1]

### Fixed

* Fail early if `custom_model_code`, `dependent_dir` or `requirements_txt_file` are `None` when model type is `ModelType.custom`.
* Fail early if `model` is not `None` when model type is `ModelType.custom`.

## [0.2.0a0]

### Added

* Accepts AZURE as a `DeploymentType`.

### Changed

* Compatibility with Unbox backend storage and data refactor.

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
