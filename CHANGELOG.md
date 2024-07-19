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

## 0.2.0-alpha.10 (2024-07-19)

Full Changelog: [v0.2.0-alpha.9...v0.2.0-alpha.10](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.9...v0.2.0-alpha.10)

### Features

* **api:** OpenAPI spec update via Stainless API ([#265](https://github.com/openlayer-ai/openlayer-python/issues/265)) ([58a602f](https://github.com/openlayer-ai/openlayer-python/commit/58a602f3fa3ab61466b90bcfe1a1ce8db4a83fb9))
* feat: add new columns to dataset when running custom metrics ([9c0d94c](https://github.com/openlayer-ai/openlayer-python/commit/9c0d94c1ab79ab8d3f94aa21f8c460e4d7e029f7))

## 0.2.0-alpha.9 (2024-07-17)

Full Changelog: [v0.2.0-alpha.8...v0.2.0-alpha.9](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.8...v0.2.0-alpha.9)

### Features

* **api:** codegen updates ([006edb5](https://github.com/openlayer-ai/openlayer-python/commit/006edb56e4cd3ec6e2ba8e3d79b326b3f08526db))
* **api:** OpenAPI spec update via Stainless API ([#261](https://github.com/openlayer-ai/openlayer-python/issues/261)) ([b8bcee3](https://github.com/openlayer-ai/openlayer-python/commit/b8bcee347e9355dcb904b9d531be766bd787285e))
* **api:** update via SDK Studio ([#262](https://github.com/openlayer-ai/openlayer-python/issues/262)) ([b8718de](https://github.com/openlayer-ai/openlayer-python/commit/b8718de4e1bd37e3c44180523bd46928579f64a0))
* **api:** update via SDK Studio ([#263](https://github.com/openlayer-ai/openlayer-python/issues/263)) ([6852bd4](https://github.com/openlayer-ai/openlayer-python/commit/6852bd4a0b9b64edd41ff6ea9eec24d396fe9528))

## 0.2.0-alpha.8 (2024-07-08)

Full Changelog: [v0.2.0-alpha.7...v0.2.0-alpha.8](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.7...v0.2.0-alpha.8)

### Features

* **api:** OpenAPI spec update via Stainless API ([#256](https://github.com/openlayer-ai/openlayer-python/issues/256)) ([af3d1ee](https://github.com/openlayer-ai/openlayer-python/commit/af3d1ee07dd9102f743157d117cbd355f485dc94))
* **api:** OpenAPI spec update via Stainless API ([#257](https://github.com/openlayer-ai/openlayer-python/issues/257)) ([38ac5ff](https://github.com/openlayer-ai/openlayer-python/commit/38ac5fff100fb0cfadd87b27f1b81ed23b7eba51))
* **api:** update via SDK Studio ([#254](https://github.com/openlayer-ai/openlayer-python/issues/254)) ([ea55198](https://github.com/openlayer-ai/openlayer-python/commit/ea55198158b95c3c32bc7f9361ebd4ae2a15b1ff))
* **api:** update via SDK Studio ([#258](https://github.com/openlayer-ai/openlayer-python/issues/258)) ([2b4eb5d](https://github.com/openlayer-ai/openlayer-python/commit/2b4eb5d340298559b2660d1a04456b8cc3edab3d))


### Chores

* go live ([#259](https://github.com/openlayer-ai/openlayer-python/issues/259)) ([ee2f102](https://github.com/openlayer-ai/openlayer-python/commit/ee2f1029f246ef9b70176b974d085166f7d9a322))
* move cost estimation logic to the backend ([b9e1134](https://github.com/openlayer-ai/openlayer-python/commit/b9e113481e570101ba8e9512ee5ebb49e5a5732c))

## 0.2.0-alpha.7 (2024-07-04)

Full Changelog: [v0.2.0-alpha.6...v0.2.0-alpha.7](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.6...v0.2.0-alpha.7)

### Features

* **api:** update via SDK Studio ([#250](https://github.com/openlayer-ai/openlayer-python/issues/250)) ([89330f7](https://github.com/openlayer-ai/openlayer-python/commit/89330f72a36008aba53df89ba3e3114036efe4a0))
* **api:** update via SDK Studio ([#252](https://github.com/openlayer-ai/openlayer-python/issues/252)) ([b205e14](https://github.com/openlayer-ai/openlayer-python/commit/b205e146dd4af68232d3d97fbda4583a56431594))

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
