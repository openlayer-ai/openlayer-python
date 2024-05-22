# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
* Added support for OpenAI assistants. The `llm_monitor` now supports monitoring OpenAI assistant runs with the function `monitor_run`.
* Added the ability to use the `llm_monitor.OpenAIMonitor` as a context manager.
* Added `openlayer_inference_pipeline_id` as an optional parameter to the `OpenAIMonitor`. This is an alternative to `openlayer_inference_pipeline_name` and `openlayer_inference_project_name` parameters for identifying the inference pipeline on the platform.
* Added `monitor_output_only` as an argument to the OpenAI `llm_monitor`. If set to `True`, the monitor will only record the output of the model, and not the input.
* Added `costColumnName` as an optional field in the config for LLM data.

### Changed
* `llm_monitor` for OpenAI models now records the `cost` estimate and uploads it.

### Removed
* Deprecated and removed `publish_ground_truths` method. Use `update_data` instead.

## 0.1.0-alpha.3 (2024-05-22)

Full Changelog: [v0.1.0-alpha.2...v0.1.0-alpha.3](https://github.com/openlayer-ai/openlayer-python/compare/v0.1.0-alpha.2...v0.1.0-alpha.3)

### Features

* **api:** OpenAPI spec update via Stainless API ([#207](https://github.com/openlayer-ai/openlayer-python/issues/207)) ([0a806f1](https://github.com/openlayer-ai/openlayer-python/commit/0a806f1be1042caeefcebb2bf17636190abb4685))
* **api:** OpenAPI spec update via Stainless API ([#209](https://github.com/openlayer-ai/openlayer-python/issues/209)) ([da14f38](https://github.com/openlayer-ai/openlayer-python/commit/da14f383fd48523a7e79431dd50ff7c6baac370b))
* **api:** OpenAPI spec update via Stainless API ([#210](https://github.com/openlayer-ai/openlayer-python/issues/210)) ([9a261c6](https://github.com/openlayer-ai/openlayer-python/commit/9a261c6b3bdada872bd221d5bbd311d5e3d12fcf))

## 0.1.0-alpha.2 (2024-05-20)

Full Changelog: [v0.1.0-alpha.1...v0.1.0-alpha.2](https://github.com/openlayer-ai/openlayer-python/compare/v0.1.0-alpha.1...v0.1.0-alpha.2)

### Features

* fix: remove openlayer/ directory ([1faaf2f](https://github.com/openlayer-ai/openlayer-python/commit/1faaf2fa91947706be32783c76807fc98020fc3d))

## 0.1.0-alpha.1 (2024-05-20)

Full Changelog: [v0.0.1-alpha.0...v0.1.0-alpha.1](https://github.com/openlayer-ai/openlayer-python/compare/v0.0.1-alpha.0...v0.1.0-alpha.1)

### Features

* various codegen changes ([002b857](https://github.com/openlayer-ai/openlayer-python/commit/002b85774bc4170d9115a4df9e4185ddd2d19b05))


### Bug Fixes

* s3 storage type ([af91766](https://github.com/openlayer-ai/openlayer-python/commit/af917668a06be1c61f7b9f29d97b5b976a54ae79))

## [0.1.0a20]

### Added
* Added `prompt` as an optional field in the config for LLM production data.
* `llm_monitor` for OpenAI ChatCompletion models records the `prompt` used and uploads it.
