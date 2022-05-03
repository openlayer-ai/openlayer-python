# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* Can specify `ModelType.keras`.

### Changed

* Prevents dataset upload when `label_column_name` is also in `feature_names`.
* Unbox server URL.

## [0.1.0] - 2022-04-17

### Added

* Accepts GCP as a `DeploymentType`.
