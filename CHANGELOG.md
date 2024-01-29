# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
* Added `openlayer_inference_pipeline_id` as an optional parameter to the `OpenAIMonitor`. This is an alternative to `openlayer_inference_pipeline_name` and `openlayer_inference_project_name` parameters for identifying the inference pipeline on the platform.
* Added `monitor_output_only` as an argument to the OpenAI `llm_monitor`. If set to `True`, the monitor will only record the output of the model, and not the input.
* Added `costColumnName` as an optional field in the config for LLM data.

### Changed
* `llm_monitor` for OpenAI models now records the `cost` estimate and uploads it.

### Removed
* Deprecated and removed `publish_ground_truths` method. Use `update_data` instead.

## [0.1.0a20]

### Added
* Added `prompt` as an optional field in the config for LLM production data.
* `llm_monitor` for OpenAI ChatCompletion models records the `prompt` used and uploads it.
