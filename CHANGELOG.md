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

## 0.2.0-alpha.6 (2024-06-28)

Full Changelog: [v0.2.0-alpha.5...v0.2.0-alpha.6](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.5...v0.2.0-alpha.6)

### Features

* **api:** update via SDK Studio ([#246](https://github.com/openlayer-ai/openlayer-python/issues/246)) ([ed77b5b](https://github.com/openlayer-ai/openlayer-python/commit/ed77b5b0870f11856cf534fa4ad24a0989b2a10c))
* feat(WIP): add support for custom metrics ([6c1cf1d](https://github.com/openlayer-ai/openlayer-python/commit/6c1cf1d7c4937776a31caf0e05d73aa8cf622791))

## 0.2.0-alpha.5 (2024-06-26)

Full Changelog: [v0.2.0-alpha.4...v0.2.0-alpha.5](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.4...v0.2.0-alpha.5)

### Chores

* **internal:** version bump ([#243](https://github.com/openlayer-ai/openlayer-python/issues/243)) ([7f06eeb](https://github.com/openlayer-ai/openlayer-python/commit/7f06eeb753c1c33070e52bdce002b22416aaeac6))

## 0.2.0-alpha.4 (2024-06-25)

Full Changelog: [v0.2.0-alpha.3...v0.2.0-alpha.4](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.3...v0.2.0-alpha.4)

### Features

* feat: Add Anthropic tracer ([25792c5](https://github.com/openlayer-ai/openlayer-python/commit/25792c5abec407fd8b44c24997579e143ff25a2d))


### Chores

* **internal:** version bump ([#239](https://github.com/openlayer-ai/openlayer-python/issues/239)) ([24057f9](https://github.com/openlayer-ai/openlayer-python/commit/24057f9b390cc32a117618b77313aba8d60783d4))

## 0.2.0-alpha.3 (2024-06-20)

Full Changelog: [v0.2.0-alpha.2...v0.2.0-alpha.3](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.2...v0.2.0-alpha.3)

### Features

* feat: python async function tracing in dev mode, closing OPEN-6157 ([7cb1a07](https://github.com/openlayer-ai/openlayer-python/commit/7cb1a0768ddd9f2d49b50d4a0b30544bd4c28cc2))

## 0.2.0-alpha.2 (2024-06-11)

Full Changelog: [v0.2.0-alpha.1...v0.2.0-alpha.2](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.1...v0.2.0-alpha.2)

### Features

* fix: include pandas as requirement ([733ee7e](https://github.com/openlayer-ai/openlayer-python/commit/733ee7e7c21dbc80c014e137036896b0000b798a))

## 0.2.0-alpha.1 (2024-06-10)

Full Changelog: [v0.1.0-alpha.5...v0.2.0-alpha.1](https://github.com/openlayer-ai/openlayer-python/compare/v0.1.0-alpha.5...v0.2.0-alpha.1)

### Chores

* update Colab URLs for notebook examples ([5c822fa](https://github.com/openlayer-ai/openlayer-python/commit/5c822fa380f20ebcb93e8a6998e2b8e00958dd54))
* update SDK settings ([#224](https://github.com/openlayer-ai/openlayer-python/issues/224)) ([e4afabb](https://github.com/openlayer-ai/openlayer-python/commit/e4afabb2354859bc372e8b08b96c07a0f275dd4f))
* update SDK settings ([#227](https://github.com/openlayer-ai/openlayer-python/issues/227)) ([1b56601](https://github.com/openlayer-ai/openlayer-python/commit/1b566012d18b6e1baafa5fedd3e265e1dba477bd))

## 0.1.0-alpha.5 (2024-06-05)

Full Changelog: [v0.1.0-alpha.4...v0.1.0-alpha.5](https://github.com/openlayer-ai/openlayer-python/compare/v0.1.0-alpha.4...v0.1.0-alpha.5)

### Features

* completes OPEN-6020 Refactor manual part of the Python SDK ([9cb9cc1](https://github.com/openlayer-ai/openlayer-python/commit/9cb9cc1fd18e7051d53ba7f95f669a2d70fa0b27))


### Chores

* apply formatting to custom files ([3414c66](https://github.com/openlayer-ai/openlayer-python/commit/3414c66705e08185746caacfdcc6fc3682884a57))
* update examples with new SDK syntax ([4bc92a5](https://github.com/openlayer-ai/openlayer-python/commit/4bc92a5775b7d0c0f9f9b2ad08f7001ac97c5098))
* update SDK settings ([#219](https://github.com/openlayer-ai/openlayer-python/issues/219)) ([0668954](https://github.com/openlayer-ai/openlayer-python/commit/0668954d989a74fa9a8021445c17dae26f043a12))
* update SDK settings ([#221](https://github.com/openlayer-ai/openlayer-python/issues/221)) ([600247b](https://github.com/openlayer-ai/openlayer-python/commit/600247ba9f6eccef57038e79413bf8260b398079))

## 0.1.0-alpha.4 (2024-05-24)

Full Changelog: [v0.1.0-alpha.3...v0.1.0-alpha.4](https://github.com/openlayer-ai/openlayer-python/compare/v0.1.0-alpha.3...v0.1.0-alpha.4)

### Chores

* configure new SDK language ([#213](https://github.com/openlayer-ai/openlayer-python/issues/213)) ([a6450d7](https://github.com/openlayer-ai/openlayer-python/commit/a6450d7530b0ce06a949e0011bb7a5228866b179))

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
