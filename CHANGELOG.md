# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## 0.2.0-alpha.40 (2024-12-10)

Full Changelog: [v0.2.0-alpha.39...v0.2.0-alpha.40](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.39...v0.2.0-alpha.40)

### Bug Fixes

* **client:** compat with new httpx 0.28.0 release ([#394](https://github.com/openlayer-ai/openlayer-python/issues/394)) ([c05fb39](https://github.com/openlayer-ai/openlayer-python/commit/c05fb39d3ce2f54b01f1f4536f612f73f5511b69))


### Chores

* **internal:** codegen related update ([#396](https://github.com/openlayer-ai/openlayer-python/issues/396)) ([6d0d530](https://github.com/openlayer-ai/openlayer-python/commit/6d0d5309210d82076f31df5c13feefaa71ee7e44))
* **internal:** codegen related update ([#399](https://github.com/openlayer-ai/openlayer-python/issues/399)) ([5927ddc](https://github.com/openlayer-ai/openlayer-python/commit/5927ddc54cfbf56ef5b1c85f23ace9ae4aa54505))
* **internal:** exclude mypy from running on tests ([#392](https://github.com/openlayer-ai/openlayer-python/issues/392)) ([2ce3de0](https://github.com/openlayer-ai/openlayer-python/commit/2ce3de0cdd36063bffd68ef34cb4062e675c9fe6))
* make the `Omit` type public ([#398](https://github.com/openlayer-ai/openlayer-python/issues/398)) ([f8aaafa](https://github.com/openlayer-ai/openlayer-python/commit/f8aaafa2ba06516ef986407be382caf8ec141ed8))

## 0.2.0-alpha.39 (2024-11-26)

Full Changelog: [v0.2.0-alpha.38...v0.2.0-alpha.39](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.38...v0.2.0-alpha.39)

### Bug Fixes

* add missing dependencies (tqdm and numpy&lt;2) ([298eedb](https://github.com/openlayer-ai/openlayer-python/commit/298eedb4861ac74859da3b167390cd4897c5ad32))


### Chores

* **internal:** codegen related update ([#388](https://github.com/openlayer-ai/openlayer-python/issues/388)) ([2dec899](https://github.com/openlayer-ai/openlayer-python/commit/2dec8992b9bc0003af4d61a4972ca4c9eac0d8ea))
* remove now unused `cached-property` dep ([#389](https://github.com/openlayer-ai/openlayer-python/issues/389)) ([c6e03c8](https://github.com/openlayer-ai/openlayer-python/commit/c6e03c84fa2f1dd564c19f45e1addba74b7540e8))

## 0.2.0-alpha.38 (2024-11-19)

Full Changelog: [v0.2.0-alpha.37...v0.2.0-alpha.38](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.37...v0.2.0-alpha.38)

### Bug Fixes

* pin pyarrow version to avoid installation issues with latest versions ([37af76c](https://github.com/openlayer-ai/openlayer-python/commit/37af76c534ac831469e488f964b7949df72a3a93))
* update to pyarrow==14.0.1 to avoid dependabot issues ([a226ca2](https://github.com/openlayer-ai/openlayer-python/commit/a226ca2c18b75232099f628246b3ae2158e97cb2))


### Chores

* rebuild project due to codegen change ([#384](https://github.com/openlayer-ai/openlayer-python/issues/384)) ([b6873de](https://github.com/openlayer-ai/openlayer-python/commit/b6873de3f5de327b1db17451ab328d93e0ee214f))

## 0.2.0-alpha.37 (2024-11-13)

Full Changelog: [v0.2.0-alpha.36...v0.2.0-alpha.37](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.36...v0.2.0-alpha.37)

### Chores

* add Vertex AI example ([b668aeb](https://github.com/openlayer-ai/openlayer-python/commit/b668aeb58f7b78f85136c3635d1c8959df5bec21))
* **internal:** version bump ([#375](https://github.com/openlayer-ai/openlayer-python/issues/375)) ([fcd0205](https://github.com/openlayer-ai/openlayer-python/commit/fcd0205203eb54776bf7d3b361db82c2681816ff))
* rebuild project due to codegen change ([#378](https://github.com/openlayer-ai/openlayer-python/issues/378)) ([01ba806](https://github.com/openlayer-ai/openlayer-python/commit/01ba806143e8cb0e2d718501226e62e55cb7a1de))
* rebuild project due to codegen change ([#379](https://github.com/openlayer-ai/openlayer-python/issues/379)) ([a6fc82b](https://github.com/openlayer-ai/openlayer-python/commit/a6fc82b48729044f8a00d2947b751414f4b423af))

## 0.2.0-alpha.36 (2024-11-04)

Full Changelog: [v0.2.0-alpha.35...v0.2.0-alpha.36](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.35...v0.2.0-alpha.36)

### Chores

* **internal:** version bump ([#373](https://github.com/openlayer-ai/openlayer-python/issues/373)) ([1fe6227](https://github.com/openlayer-ai/openlayer-python/commit/1fe6227f705fb1f3e8b31e16813a1b1e21f23caf))

## 0.2.0-alpha.35 (2024-11-04)

Full Changelog: [v0.2.0-alpha.34...v0.2.0-alpha.35](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.34...v0.2.0-alpha.35)

### Features

* feat(data): add function to push a commit to the platform ([7b5a29e](https://github.com/openlayer-ai/openlayer-python/commit/7b5a29e7622fec7185b6eb9eec705ac298888d5e))


### Chores

* **internal:** version bump ([#370](https://github.com/openlayer-ai/openlayer-python/issues/370)) ([5b3bd38](https://github.com/openlayer-ai/openlayer-python/commit/5b3bd3887d10dea9371ea1c7e417e32e047a7462))

## 0.2.0-alpha.34 (2024-11-01)

Full Changelog: [v0.2.0-alpha.33...v0.2.0-alpha.34](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.33...v0.2.0-alpha.34)

### Chores

* **internal:** version bump ([#368](https://github.com/openlayer-ai/openlayer-python/issues/368)) ([4559716](https://github.com/openlayer-ai/openlayer-python/commit/4559716e585852866ecec7413da146503b324717))

## 0.2.0-alpha.33 (2024-10-31)

Full Changelog: [v0.2.0-alpha.32...v0.2.0-alpha.33](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.32...v0.2.0-alpha.33)

### Features

* **api:** manual updates ([#364](https://github.com/openlayer-ai/openlayer-python/issues/364)) ([f14669b](https://github.com/openlayer-ai/openlayer-python/commit/f14669be5f6790af961657b4d7c8f8dca2371f30))


### Bug Fixes

* **internal:** remove stale files ([52247af](https://github.com/openlayer-ai/openlayer-python/commit/52247affd27056cbda7a8b8da1d7ca0b9f9253a9))

## 0.2.0-alpha.32 (2024-10-31)

Full Changelog: [v0.2.0-alpha.31...v0.2.0-alpha.32](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.31...v0.2.0-alpha.32)

### Features

* **api:** manual updates ([#360](https://github.com/openlayer-ai/openlayer-python/issues/360)) ([4641235](https://github.com/openlayer-ai/openlayer-python/commit/4641235bf842a5d6d132870517aa1ac523867fc9))


### Bug Fixes

* **docs:** remove old examples from next branch ([534b732](https://github.com/openlayer-ai/openlayer-python/commit/534b73224f9adb3b287fac1f4abd285eed65c047))
* **docs:** ruff linting issues ([728a7dc](https://github.com/openlayer-ai/openlayer-python/commit/728a7dc71ddb0edb1f8cfa7c0d6889801d1486a0))

## 0.2.0-alpha.31 (2024-10-07)

Full Changelog: [v0.2.0-alpha.30...v0.2.0-alpha.31](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.30...v0.2.0-alpha.31)

### Features

* fix: adjust storage upload error code range ([867b3d2](https://github.com/openlayer-ai/openlayer-python/commit/867b3d2a193bc5c6626056ac5782e2e8f5b30ae0))

## 0.2.0-alpha.30 (2024-10-05)

Full Changelog: [v0.2.0-alpha.29...v0.2.0-alpha.30](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.29...v0.2.0-alpha.30)

### Features

* fix: remove async uploads ([28e24a5](https://github.com/openlayer-ai/openlayer-python/commit/28e24a5c6c1fcac010362c970c3901207687e5fc))

## 0.2.0-alpha.29 (2024-10-03)

Full Changelog: [v0.2.0-alpha.28...v0.2.0-alpha.29](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.28...v0.2.0-alpha.29)

### Features

* feat: add async batch uploads & improve client-side upload latency ([7e7261d](https://github.com/openlayer-ai/openlayer-python/commit/7e7261d9c8eab2ee0f781500502483f316009a1e))
* improvement: make data stream example about tabular classification ([03f1f31](https://github.com/openlayer-ai/openlayer-python/commit/03f1f316bedb9c6fef39e2fbe853eed53266c1f2))

## 0.2.0-alpha.28 (2024-09-25)

Full Changelog: [v0.2.0-alpha.27...v0.2.0-alpha.28](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.27...v0.2.0-alpha.28)

### Features

* chore: show how to log context in RAG notebook example ([5610593](https://github.com/openlayer-ai/openlayer-python/commit/5610593bc124d601c0dda0c2e507cf9bfafdfd77))
* fix: make sure that context logging works in development mode ([11f5267](https://github.com/openlayer-ai/openlayer-python/commit/11f526701591ee36d8f6e56b651397360ef589f1))

## 0.2.0-alpha.27 (2024-09-12)

Full Changelog: [v0.2.0-alpha.26...v0.2.0-alpha.27](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.26...v0.2.0-alpha.27)

### Chores

* **internal:** codegen related update ([#333](https://github.com/openlayer-ai/openlayer-python/issues/333)) ([ad7b567](https://github.com/openlayer-ai/openlayer-python/commit/ad7b56761fed6576424bdaf6f49cb4ae604936bc))
* **internal:** codegen related update ([#340](https://github.com/openlayer-ai/openlayer-python/issues/340)) ([4bd2cb2](https://github.com/openlayer-ai/openlayer-python/commit/4bd2cb2a601b20f2673206031acf3cef0190de4a))

## 0.2.0-alpha.26 (2024-08-29)

Full Changelog: [v0.2.0-alpha.25...v0.2.0-alpha.26](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.25...v0.2.0-alpha.26)

### Features

* feat: add Groq tracer ([bdf3f36](https://github.com/openlayer-ai/openlayer-python/commit/bdf3f368da9e1608cc6b56233563cce57d9b7af7))


### Chores

* **internal:** codegen related update ([#333](https://github.com/openlayer-ai/openlayer-python/issues/333)) ([e1e2237](https://github.com/openlayer-ai/openlayer-python/commit/e1e223797c569a7db65f8a0fdb08bc480200788b))

## 0.2.0-alpha.25 (2024-08-29)

Full Changelog: [v0.2.0-alpha.24...v0.2.0-alpha.25](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.24...v0.2.0-alpha.25)

### Features

* fix: batch uploads to VMs broken when using filesystem storage ([31e4195](https://github.com/openlayer-ai/openlayer-python/commit/31e4195f6626d0f789ad6d8f9eeee7b371b144fa))


### Chores

* **internal:** codegen related update ([#333](https://github.com/openlayer-ai/openlayer-python/issues/333)) ([ad43d95](https://github.com/openlayer-ai/openlayer-python/commit/ad43d954c6066f0d0a7518054739cb20cf90ac19))

## 0.2.0-alpha.24 (2024-08-29)

Full Changelog: [v0.2.0-alpha.23...v0.2.0-alpha.24](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.23...v0.2.0-alpha.24)

### Features

* **api:** OpenAPI spec update via Stainless API ([#325](https://github.com/openlayer-ai/openlayer-python/issues/325)) ([24230df](https://github.com/openlayer-ai/openlayer-python/commit/24230dffda1fe7e37068fd98d59647bf085bda54))
* **api:** update via SDK Studio ([#323](https://github.com/openlayer-ai/openlayer-python/issues/323)) ([0090a06](https://github.com/openlayer-ai/openlayer-python/commit/0090a0691d6c3eb988bf669ca8869913ffc57d24))
* feat: add tracer for Mistral AI ([a1b8729](https://github.com/openlayer-ai/openlayer-python/commit/a1b8729773bb2b78ae73c4900d4020c5a09ea42e))

## 0.2.0-alpha.23 (2024-08-26)

Full Changelog: [v0.2.0-alpha.22...v0.2.0-alpha.23](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.22...v0.2.0-alpha.23)

### Features

* improvement: updates to custom metric runner
* improvement: skip metrics if already computed, surface errors for each metric
* feat: add --dataset flag so custom metrics can be forced to run on only specific datasets

## 0.2.0-alpha.22 (2024-08-21)

Full Changelog: [v0.2.0-alpha.21...v0.2.0-alpha.22](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.21...v0.2.0-alpha.22)

### Bug Fixes

* add missing dependency for Anthropic notebook example ([eddc160](https://github.com/openlayer-ai/openlayer-python/commit/eddc160a8d40478655c241d682cfe12afa851d91))


### Chores

* **ci:** also run pydantic v1 tests ([#319](https://github.com/openlayer-ai/openlayer-python/issues/319)) ([6959e23](https://github.com/openlayer-ai/openlayer-python/commit/6959e230ac798a1ad3b8a00e0483000962bece93))
* **client:** fix parsing union responses when non-json is returned ([#318](https://github.com/openlayer-ai/openlayer-python/issues/318)) ([1b18e64](https://github.com/openlayer-ai/openlayer-python/commit/1b18e646a353d20ccfd4d2ba98f6f855c6e4aa3a))

## 0.2.0-alpha.21 (2024-08-19)

Full Changelog: [v0.2.0-alpha.20...v0.2.0-alpha.21](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.20...v0.2.0-alpha.21)

### Features

* fix: add missing dependencies for LangChain notebook example ([fa382eb](https://github.com/openlayer-ai/openlayer-python/commit/fa382eb455c1e7f629314b06f0ddf2e6dc0fccc6))


### Chores

* **internal:** use different 32bit detection method ([#311](https://github.com/openlayer-ai/openlayer-python/issues/311)) ([389516d](https://github.com/openlayer-ai/openlayer-python/commit/389516d55843bc0e765cde855afa4759d67b5820))

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
