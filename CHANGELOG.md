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

## 0.2.0-alpha.20 (2024-08-19)

Full Changelog: [v0.2.0-alpha.19...v0.2.0-alpha.20](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.19...v0.2.0-alpha.20)

### Features

* fix: add pyyaml to requirements ([94626f0](https://github.com/openlayer-ai/openlayer-python/commit/94626f0329cadc2f18219c13eea89da3825823eb))


### Chores

* **examples:** minor formatting changes ([#307](https://github.com/openlayer-ai/openlayer-python/issues/307)) ([9060e31](https://github.com/openlayer-ai/openlayer-python/commit/9060e3173a21ecb66116b906eaacb533f28dabc1))

## 0.2.0-alpha.19 (2024-08-13)

Full Changelog: [v0.2.0-alpha.18...v0.2.0-alpha.19](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.18...v0.2.0-alpha.19)

### Features

* feat: allow specification of context column name when using tracers ([05c5df5](https://github.com/openlayer-ai/openlayer-python/commit/05c5df55a10eaed48b5d54c4b7fe4f5406b8ae39))
* feat: support Vertex AI models via LangChain callback handler ([0e53043](https://github.com/openlayer-ai/openlayer-python/commit/0e5304358869b400d54b9abe5bd0158dd5a94bf0))

## 0.2.0-alpha.18 (2024-08-12)

Full Changelog: [v0.2.0-alpha.17...v0.2.0-alpha.18](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.17...v0.2.0-alpha.18)

### Chores

* **ci:** bump prism mock server version ([#299](https://github.com/openlayer-ai/openlayer-python/issues/299)) ([c97393c](https://github.com/openlayer-ai/openlayer-python/commit/c97393cd131112cb8f2038fef57513f9c5774064))
* **internal:** codegen related update ([#296](https://github.com/openlayer-ai/openlayer-python/issues/296)) ([4025f65](https://github.com/openlayer-ai/openlayer-python/commit/4025f65af981a377bee7887d1ef71d2a16f2edeb))
* **internal:** ensure package is importable in lint cmd ([#300](https://github.com/openlayer-ai/openlayer-python/issues/300)) ([8033a12](https://github.com/openlayer-ai/openlayer-python/commit/8033a1291ce6f3c6db18ec51e228b5b45976bd80))
* **internal:** remove deprecated ruff config ([#298](https://github.com/openlayer-ai/openlayer-python/issues/298)) ([8d2604b](https://github.com/openlayer-ai/openlayer-python/commit/8d2604bec7d5d1489a7208211c0be9e2a78dc465))

## 0.2.0-alpha.17 (2024-08-12)

Full Changelog: [v0.2.0-alpha.16...v0.2.0-alpha.17](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.16...v0.2.0-alpha.17)

### Features

* feat: support Ollama models via LangChain callback handler ([2865b34](https://github.com/openlayer-ai/openlayer-python/commit/2865b34e70f2f2437bcd2459520a1ee0f7985925))

## 0.2.0-alpha.16 (2024-07-31)

Full Changelog: [v0.2.0-alpha.15...v0.2.0-alpha.16](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.15...v0.2.0-alpha.16)

### Features

* fix: uploading batch data was broken ([d16eee4](https://github.com/openlayer-ai/openlayer-python/commit/d16eee4c3d7d5f474b25033d2cff08c322581077))

## 0.2.0-alpha.15 (2024-07-31)

Full Changelog: [v0.2.0-alpha.14...v0.2.0-alpha.15](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.14...v0.2.0-alpha.15)

### Features

* improvement: allow specifying dataset as path for uploads ([a4d126f](https://github.com/openlayer-ai/openlayer-python/commit/a4d126f2c0b3bdf67fefbb06fb3ffa9107ea1387))
* improvement: include method to update batch of inferences ([a8f3d82](https://github.com/openlayer-ai/openlayer-python/commit/a8f3d8246c75ff8ebff8f5e92212044fd3433d47))


### Chores

* **internal:** add type construction helper ([#287](https://github.com/openlayer-ai/openlayer-python/issues/287)) ([39fbda1](https://github.com/openlayer-ai/openlayer-python/commit/39fbda1bcaacbd8546926e7d32b7fc2ae1ad058e))
* **internal:** version bump ([#284](https://github.com/openlayer-ai/openlayer-python/issues/284)) ([73c3067](https://github.com/openlayer-ai/openlayer-python/commit/73c30676b1e49e2355cffd232305c5aab1a0b309))
* **tests:** update prism version ([#285](https://github.com/openlayer-ai/openlayer-python/issues/285)) ([3c0fcbb](https://github.com/openlayer-ai/openlayer-python/commit/3c0fcbbe9199b68ef5bc92247df751bfd4ae3649))

## 0.2.0-alpha.14 (2024-07-29)

Full Changelog: [v0.2.0-alpha.13...v0.2.0-alpha.14](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.13...v0.2.0-alpha.14)

### Features

* feat: allow inference_pipeline_id to be specified as a kwarg for tracing ([e2b9ace](https://github.com/openlayer-ai/openlayer-python/commit/e2b9ace1225db6630b7ab6546c542176567673ca))


### Chores

* **tests:** update prism version ([#279](https://github.com/openlayer-ai/openlayer-python/issues/279)) ([e2fe88f](https://github.com/openlayer-ai/openlayer-python/commit/e2fe88f8722769ca4e849596b78e983b82f36ac1))

## 0.2.0-alpha.13 (2024-07-23)

Full Changelog: [v0.2.0-alpha.12...v0.2.0-alpha.13](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.12...v0.2.0-alpha.13)

### Features

* upload a batch of inferences ([fa3eb50](https://github.com/openlayer-ai/openlayer-python/commit/fa3eb5003223b02c36bda486018e8e90349c862c))
* upload a reference dataset ([eff6bf0](https://github.com/openlayer-ai/openlayer-python/commit/eff6bf0a1d3a7e68b851c822c85db472660484d8))

## 0.2.0-alpha.12 (2024-07-23)

Full Changelog: [v0.2.0-alpha.11...v0.2.0-alpha.12](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.11...v0.2.0-alpha.12)

### Features

* **api:** update via SDK Studio ([#272](https://github.com/openlayer-ai/openlayer-python/issues/272)) ([dc7ef78](https://github.com/openlayer-ai/openlayer-python/commit/dc7ef78f40cccfb1b5254a3c13217b237a09fa48))
* **api:** update via SDK Studio ([#274](https://github.com/openlayer-ai/openlayer-python/issues/274)) ([2e703d3](https://github.com/openlayer-ai/openlayer-python/commit/2e703d3240b1273e4a5914afaccd4082752eae1d))

## 0.2.0-alpha.11 (2024-07-22)

Full Changelog: [v0.2.0-alpha.10...v0.2.0-alpha.11](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.10...v0.2.0-alpha.11)

### Features

* **api:** update via SDK Studio ([#270](https://github.com/openlayer-ai/openlayer-python/issues/270)) ([b5d333b](https://github.com/openlayer-ai/openlayer-python/commit/b5d333bc6c654cbe0d0952f949da0bfd9bc91cf4))


### Chores

* **internal:** refactor release doctor script ([#269](https://github.com/openlayer-ai/openlayer-python/issues/269)) ([11a5605](https://github.com/openlayer-ai/openlayer-python/commit/11a5605b48310b1bc9fa840865e375a74c93e55b))
* **internal:** version bump ([#267](https://github.com/openlayer-ai/openlayer-python/issues/267)) ([932aac4](https://github.com/openlayer-ai/openlayer-python/commit/932aac43080f81ac5f5e3725f068bb4a628d8c88))

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
