# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

* Fix warnings related to not closing requests sessions
* Loading bar no longer jumps from 0% to 100% for large uploads

### Added

* Added the project's `export` method, which exports the resources in the project's staging area to a specified location.
* Added `llm` as a supported `architectureType` for models.
* Added `protobuf<3.20` to requirements to fix compatibility issue with Tensorflow.
* Warnings if the dependencies from the `requirement_txt_file` and current environment are inconsistent.
* Paths to custom SSL certificates can now be modified by altering `openlayer.api.VERIFY_REQUESTS`. The value can either be True (default), False, or a path to a certificate.
* Ability to check for goal statuses through the API.

### Changed

* Renamed conda environment created by the model runner from `new-openlayer` to `model-runner-env-%m-%d-%H-%M-%S-%f`. 
* Modified the zero-index integer checks for `predictionsColumnName` and `labelColumnName` to support dataset uploads with only a sample of the classes.
* Renamed `predictionsColumnName` argument from the datasets' configuration YAML to `predictionScoresColumnName`. 
* Migrated package name from [openlayer](https://pypi.org/project/openlayer/) to [openlayer](https://pypi.org/project/openlayer/) due to a company name change.
* Required Python version `>=3.7` and `<3.9`.
