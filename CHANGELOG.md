# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## 0.2.0-alpha.66 (2025-07-09)

Full Changelog: [v0.2.0-alpha.65...v0.2.0-alpha.66](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.65...v0.2.0-alpha.66)

### Features

* **tracer:** enhance tracing functionality with helper methods for input extraction and logging finalization ([c2908d0](https://github.com/openlayer-ai/openlayer-python/commit/c2908d0f5516a21b8c25a830b7cd98a4df797ac6))
* **tracer:** implement lazy initialization for Openlayer client ([2daf847](https://github.com/openlayer-ai/openlayer-python/commit/2daf847518779c070e0cb9b82ff6a8267dd6b965))
* **tracer:** refactor step creation and logging for improved clarity and maintainability ([243a7f9](https://github.com/openlayer-ai/openlayer-python/commit/243a7f9011f66a38af8bc60fbe8687395a73c222))


### Bug Fixes

* update client retrieval for LangChain callback handler ([7fb7cbe](https://github.com/openlayer-ai/openlayer-python/commit/7fb7cbed5c7781bf1266655e0dbc0caff5b80c00))


### Chores

* format file ([16429ac](https://github.com/openlayer-ai/openlayer-python/commit/16429ac260e1c458af12ed7c5508d9af9e4722bb))


### Documentation

* add LangGraph notebook example ([cb8838c](https://github.com/openlayer-ai/openlayer-python/commit/cb8838c0d0f6bd983e295eaf990eb35ecf9a48e7))


### Refactors

* **tracer:** simplify async step creation by consolidating functions ([d61888c](https://github.com/openlayer-ai/openlayer-python/commit/d61888c4e23b8b592022e0ee766bab87d79d7e13))
* **tracer:** streamline code formatting and improve readability ([bada5eb](https://github.com/openlayer-ai/openlayer-python/commit/bada5eb23c1979b0ba76f0e1c4ff3f991d54cb40))

## 0.2.0-alpha.65 (2025-07-09)

Full Changelog: [v0.2.0-alpha.64...v0.2.0-alpha.65](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.64...v0.2.0-alpha.65)

### Features

* adds openai agents sdk trace processor ([da53c53](https://github.com/openlayer-ai/openlayer-python/commit/da53c534e3e9969fa4b2bb7e1ba571caa80a78aa))
* **client:** add support for aiohttp ([977528d](https://github.com/openlayer-ai/openlayer-python/commit/977528d63ccc1d9c9ad534c2c84f490dcfd8fa2c))
* **examples:** add OpenAI Agents tracing notebook for multi-agent conversation monitoring ([dbeb9f4](https://github.com/openlayer-ai/openlayer-python/commit/dbeb9f4f8f267b02434bae4a6ab56f9f8d2843af))
* implement remaining methods for LangChain callback handler ([cd6d303](https://github.com/openlayer-ai/openlayer-python/commit/cd6d30373859a432d91d36fcd56294906e9b52aa))
* **openai-agents:** enhance OpenAI Agents tracing with structured span data extraction ([46d0852](https://github.com/openlayer-ai/openlayer-python/commit/46d08528ba036ace5fdf45a35f813c2494e1ae1f))


### Bug Fixes

* **ci:** correct conditional ([f616411](https://github.com/openlayer-ai/openlayer-python/commit/f6164110ff27782f0df72c486d2c45c66f3a6cb5))
* **ci:** release-doctor — report correct token name ([e42727c](https://github.com/openlayer-ai/openlayer-python/commit/e42727caf8c7ac350874d9195487da19df7f0081))
* context list handling ([#474](https://github.com/openlayer-ai/openlayer-python/issues/474)) ([1ef1a1e](https://github.com/openlayer-ai/openlayer-python/commit/1ef1a1e917675646ec62275101d19e595ba6c2cf))
* **tests:** fix: tests which call HTTP endpoints directly with the example parameters ([ab7ef6b](https://github.com/openlayer-ai/openlayer-python/commit/ab7ef6b12437afc6bc07b1839cdd5fb70d4c3628))
* update pyarrow version ([f4feadf](https://github.com/openlayer-ai/openlayer-python/commit/f4feadfa95a07a71d79b6184795e79c44644947b))


### Chores

* **ci:** change upload type ([49cdc9c](https://github.com/openlayer-ai/openlayer-python/commit/49cdc9c1c246051fcd78722eab8896fc3398a555))
* **ci:** enable for pull requests ([07c86b5](https://github.com/openlayer-ai/openlayer-python/commit/07c86b5080d0c910e373b6f50b966ea56794e734))
* **ci:** only run for pushes and fork pull requests ([fbf9c05](https://github.com/openlayer-ai/openlayer-python/commit/fbf9c05081172a447968c7c4ed011a364239ac7a))
* **internal:** bump pinned h11 dep ([ddef8c8](https://github.com/openlayer-ai/openlayer-python/commit/ddef8c848fd1abb9a884b6fa0a42b5e9f2be0412))
* **internal:** codegen related update ([f514ca3](https://github.com/openlayer-ai/openlayer-python/commit/f514ca32ebd1068d9b91b85d1788de560da14a08))
* **internal:** update conftest.py ([af83c82](https://github.com/openlayer-ai/openlayer-python/commit/af83c828c31f99537e8b57074a325d0ec8dec13e))
* **package:** mark python 3.13 as supported ([e663ce9](https://github.com/openlayer-ai/openlayer-python/commit/e663ce9a6b27739878efac099e0c253cc616190c))
* **readme:** update badges ([2c30786](https://github.com/openlayer-ai/openlayer-python/commit/2c30786b6870f003f4c6c2a9f68136eff15d2ebf))
* refactor LangChain callback handler ([858285d](https://github.com/openlayer-ai/openlayer-python/commit/858285dc4387088001a50ebde6c1cf34ffb5374c))
* remove unused imports, break long lines, and formatting cleanup ([753c317](https://github.com/openlayer-ai/openlayer-python/commit/753c31705958f2c16ed27092e33f97aa87854230))
* **tests:** add tests for httpx client instantiation & proxies ([55a2e38](https://github.com/openlayer-ai/openlayer-python/commit/55a2e38b32dd755ac27b36c7b1ebffe0ef41d3f2))
* **tests:** skip some failing tests on the latest python versions ([ef12a3a](https://github.com/openlayer-ai/openlayer-python/commit/ef12a3a6487d67e0add70f168a5954fb49c0f47b))


### Documentation

* **client:** fix httpx.Timeout documentation reference ([ad5d7c0](https://github.com/openlayer-ai/openlayer-python/commit/ad5d7c000f6ffb885d176192a98a740ff1251bd4))


### Refactors

* **integrations:** update Openlayer integration imports ([ac78c1c](https://github.com/openlayer-ai/openlayer-python/commit/ac78c1c6c4dce5c6f822263ad9b168cd2d414c13))

## 0.2.0-alpha.64 (2025-06-16)

Full Changelog: [v0.2.0-alpha.63...v0.2.0-alpha.64](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.63...v0.2.0-alpha.64)

### Bug Fixes

* **client:** correctly parse binary response | stream ([8fe8ec0](https://github.com/openlayer-ai/openlayer-python/commit/8fe8ec0159021248987a6557c9a75f9a49a02512))
* **tracer:** pull ground truth from root step only when it is defined ([29b5f56](https://github.com/openlayer-ai/openlayer-python/commit/29b5f5672d4e2180cc5f5ae140af395b7ad1f847))


### Chores

* **tests:** run tests in parallel ([140bf6e](https://github.com/openlayer-ai/openlayer-python/commit/140bf6e8e6ee523dc7ee64d99e0b4433607d00e9))


### Documentation

* add Pydantic AI notebook example ([65f9b15](https://github.com/openlayer-ai/openlayer-python/commit/65f9b1540fa4225e01dd9e5ade3e995b00b5618f))

## 0.2.0-alpha.63 (2025-06-03)

Full Changelog: [v0.2.0-alpha.62...v0.2.0-alpha.63](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.62...v0.2.0-alpha.63)

### Features

* add MLflow notebook example ([149e85f](https://github.com/openlayer-ai/openlayer-python/commit/149e85f075db80c9800fd8dff58b277341a3384c))
* add OpenLIT notebook example ([f71c668](https://github.com/openlayer-ai/openlayer-python/commit/f71c66895d38b0245f8a5da4c000e6bf747ef4c8))
* **client:** add follow_redirects request option ([87d8986](https://github.com/openlayer-ai/openlayer-python/commit/87d89863dd9c4f700b8a8910ce14d2a961404336))


### Bug Fixes

* **package:** support direct resource imports ([8407753](https://github.com/openlayer-ai/openlayer-python/commit/84077531a8491bc48c8fe5d67a9076a27ba21fce))


### Chores

* **ci:** fix installation instructions ([d7d4fd2](https://github.com/openlayer-ai/openlayer-python/commit/d7d4fd2e5464f87660a30edd1067aef930b2249a))
* **ci:** upload sdks to package manager ([0aadb0a](https://github.com/openlayer-ai/openlayer-python/commit/0aadb0a4deed48d46981fd44b308fba5bbc5a3c1))
* **docs:** grammar improvements ([27794bc](https://github.com/openlayer-ai/openlayer-python/commit/27794bc2ff2f34c10c1635fcf14677e0711a8af0))
* **docs:** remove reference to rye shell ([9f8db4a](https://github.com/openlayer-ai/openlayer-python/commit/9f8db4a42a79af923d55ec636e43bf49ce80bc50))
* **internal:** avoid errors for isinstance checks on proxies ([3de384b](https://github.com/openlayer-ai/openlayer-python/commit/3de384be80ba27ba97a6079a78b75cdeadf55e5f))
* **internal:** codegen related update ([120114a](https://github.com/openlayer-ai/openlayer-python/commit/120114ad9d40ce7c41112522f2951dd92be61eaf))
* **internal:** codegen related update ([f990977](https://github.com/openlayer-ai/openlayer-python/commit/f990977209f13f02b1b87ab98bef5eef50414ea9))
* link to OpenLLMetry integration guide ([ffcd085](https://github.com/openlayer-ai/openlayer-python/commit/ffcd085e1ad58e2b88fac6f739b6a9a12ba05844))
* remove MLflow example ([17256c9](https://github.com/openlayer-ai/openlayer-python/commit/17256c96873cef5b085400ad64af860c35de4cf4))
* sync repo ([caa47dc](https://github.com/openlayer-ai/openlayer-python/commit/caa47dc5b9d671046dca4dd5378a72018ed5d334))

## 0.2.0-alpha.62 (2025-04-29)

Full Changelog: [v0.2.0-alpha.61...v0.2.0-alpha.62](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.61...v0.2.0-alpha.62)

### Bug Fixes

* **openai tracer:** Azure OpenAI chat completion step duplicated ([23ee128](https://github.com/openlayer-ai/openlayer-python/commit/23ee1280f621f695aa1606b1a729e94c3dbaa783))
* **openai tracer:** object async_generator can't be used in 'await' expression ([ce13918](https://github.com/openlayer-ai/openlayer-python/commit/ce13918f523355b957f9d0f7a0371bb11367a7c6))


### Chores

* **lib:** expose async tracing methods ([af49b20](https://github.com/openlayer-ai/openlayer-python/commit/af49b2007bb80718ed0cd72ae13c56f532058f0e))


### Documentation

* update docstring ([b248a52](https://github.com/openlayer-ai/openlayer-python/commit/b248a52b842a558e2717d922fb84b351c47f6320))

## 0.2.0-alpha.61 (2025-04-25)

Full Changelog: [v0.2.0-alpha.60...v0.2.0-alpha.61](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.60...v0.2.0-alpha.61)

### Features

* feat: add convenience function that copies tests from one project to another ([d59dfe0](https://github.com/openlayer-ai/openlayer-python/commit/d59dfe023b6d6e164c6e272cc410dc6b5f4bcec8))

## 0.2.0-alpha.60 (2025-04-25)

Full Changelog: [v0.2.0-alpha.59...v0.2.0-alpha.60](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.59...v0.2.0-alpha.60)

### Features

* **api:** api update ([fbce7ca](https://github.com/openlayer-ai/openlayer-python/commit/fbce7ca28fd5a013126533dc95535f202aa1de1b))

## 0.2.0-alpha.59 (2025-04-25)

Full Changelog: [v0.2.0-alpha.58...v0.2.0-alpha.59](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.58...v0.2.0-alpha.59)

### Features

* **api:** api update ([fb9c6ee](https://github.com/openlayer-ai/openlayer-python/commit/fb9c6ee1555b764a00c313ef0cd0520782de2e09))
* **api:** api update ([1a25da2](https://github.com/openlayer-ai/openlayer-python/commit/1a25da24c4c3c0fd589348718425d4b61d1d1298))
* **api:** expose test update endpoint ([ef1427e](https://github.com/openlayer-ai/openlayer-python/commit/ef1427ebc91a1f569b68f4b853758cdc7adac586))

## 0.2.0-alpha.58 (2025-04-24)

Full Changelog: [v0.2.0-alpha.57...v0.2.0-alpha.58](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.57...v0.2.0-alpha.58)

### Features

* **api:** api update ([dc2b7e5](https://github.com/openlayer-ai/openlayer-python/commit/dc2b7e51dbd22bb0f990f1d67a6ff58b103811af))
* **api:** expose test retrieval endpoint ([0bb2160](https://github.com/openlayer-ai/openlayer-python/commit/0bb2160a1079e8d9892a7977da8851ca41cd3f71))

## 0.2.0-alpha.57 (2025-04-24)

Full Changelog: [v0.2.0-alpha.56...v0.2.0-alpha.57](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.56...v0.2.0-alpha.57)

### Features

* **api:** api update ([660a2ce](https://github.com/openlayer-ai/openlayer-python/commit/660a2ce213ba8aefb4fb4f02f74532fa0baba346))
* **api:** expose test retrieval endpoint ([9762024](https://github.com/openlayer-ai/openlayer-python/commit/9762024ea999dd0fdb7d3c05636422125b1860d7))


### Bug Fixes

* **pydantic v1:** more robust ModelField.annotation check ([1624ca6](https://github.com/openlayer-ai/openlayer-python/commit/1624ca6da5760b8c849749be1fb150071b14e9ae))


### Chores

* broadly detect json family of content-type headers ([39d78ac](https://github.com/openlayer-ai/openlayer-python/commit/39d78ac984c9f8c726fa8e7c8debec418476cebc))
* **ci:** add timeout thresholds for CI jobs ([1093391](https://github.com/openlayer-ai/openlayer-python/commit/10933919d99b4e4045ce37e95ffe01eae17ea5c7))
* **ci:** only use depot for staging repos ([bafdcd8](https://github.com/openlayer-ai/openlayer-python/commit/bafdcd8cd926966f0347f0d8ad6283897f21dac3))
* **internal:** codegen related update ([8c10e35](https://github.com/openlayer-ai/openlayer-python/commit/8c10e3532cc04d0dff74e7047a580acc3544c0ac))
* **internal:** fix list file params ([312f532](https://github.com/openlayer-ai/openlayer-python/commit/312f5325acca7f11912abfd514e4d5ada640452c))
* **internal:** import reformatting ([4f944c7](https://github.com/openlayer-ai/openlayer-python/commit/4f944c71bba568da8c25468cc3f729669e5562f9))
* **internal:** refactor retries to not use recursion ([5a2c154](https://github.com/openlayer-ai/openlayer-python/commit/5a2c1542c0b2ca22eaa6a4c843de04234f677965))

## 0.2.0-alpha.56 (2025-04-21)

Full Changelog: [v0.2.0-alpha.55...v0.2.0-alpha.56](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.55...v0.2.0-alpha.56)

### Features

* **api:** add test creation endpoint ([f9c02bf](https://github.com/openlayer-ai/openlayer-python/commit/f9c02bfd25604f82b0663acdd9ef3a7a57270c59))

## 0.2.0-alpha.55 (2025-04-19)

Full Changelog: [v0.2.0-alpha.54...v0.2.0-alpha.55](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.54...v0.2.0-alpha.55)

### Features

* **api:** api update ([b40ca02](https://github.com/openlayer-ai/openlayer-python/commit/b40ca0253f502e9d249c901e7f878b7f9461a0c1))


### Chores

* **internal:** base client updates ([9afcd88](https://github.com/openlayer-ai/openlayer-python/commit/9afcd88c21786e5903f04227e314164699aeddea))
* **internal:** bump pyright version ([0301486](https://github.com/openlayer-ai/openlayer-python/commit/03014864bcb6e69d5040435521cfdc76f3189641))
* **internal:** update models test ([97be493](https://github.com/openlayer-ai/openlayer-python/commit/97be4939dc8a3d16f3316cc513a5cad8d2311d41))

## 0.2.0-alpha.54 (2025-04-15)

Full Changelog: [v0.2.0-alpha.53...v0.2.0-alpha.54](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.53...v0.2.0-alpha.54)

### Features

* fix: default value for OPENLAYER_VERIFY_SSL env var ([a4557de](https://github.com/openlayer-ai/openlayer-python/commit/a4557dec1751a34b2894c605dfd0a54787157923))

## 0.2.0-alpha.53 (2025-04-15)

Full Changelog: [v0.2.0-alpha.52...v0.2.0-alpha.53](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.52...v0.2.0-alpha.53)

### Features

* fix: verify SSL by default and disable it via env var ([92f8b70](https://github.com/openlayer-ai/openlayer-python/commit/92f8b7055c4721edc8a6ec1ab9e678ff6bf18e97))


### Chores

* **client:** minor internal fixes ([cb7cdf2](https://github.com/openlayer-ai/openlayer-python/commit/cb7cdf29f19b6131dcfb0a47dcbfd20f1b6659b6))
* **internal:** update pyright settings ([0e70ac7](https://github.com/openlayer-ai/openlayer-python/commit/0e70ac7853b7c2a353da7021e7454096c0ea6524))

## 0.2.0-alpha.52 (2025-04-14)

Full Changelog: [v0.2.0-alpha.51...v0.2.0-alpha.52](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.51...v0.2.0-alpha.52)

### Features

* feat: allow publish without ssl verification ([24dbdef](https://github.com/openlayer-ai/openlayer-python/commit/24dbdef53ccb988e6cd807094ae2a15a4e40fa7f))


### Bug Fixes

* **perf:** optimize some hot paths ([badc2bb](https://github.com/openlayer-ai/openlayer-python/commit/badc2bb1b915c70045a4f9150792746788a61b79))
* **perf:** skip traversing types for NotGiven values ([afb0108](https://github.com/openlayer-ai/openlayer-python/commit/afb01083b15f4b4f4878176f2d34a74c72ef3c57))


### Chores

* **internal:** expand CI branch coverage ([121cc4c](https://github.com/openlayer-ai/openlayer-python/commit/121cc4cf1e7276aba8fde9ca216db17242b641ed))
* **internal:** reduce CI branch coverage ([05f20c8](https://github.com/openlayer-ai/openlayer-python/commit/05f20c8ff1b471a9a3f3d6f688d0cc7d78cf680b))
* **internal:** slight transform perf improvement ([#448](https://github.com/openlayer-ai/openlayer-python/issues/448)) ([3c5cd0a](https://github.com/openlayer-ai/openlayer-python/commit/3c5cd0a60b3d33248568075ccb3576536d5cfe7e))
* **tests:** improve enum examples ([#449](https://github.com/openlayer-ai/openlayer-python/issues/449)) ([3508728](https://github.com/openlayer-ai/openlayer-python/commit/350872865c9f574048c4d6acb112ee72f81e5046))

## 0.2.0-alpha.51 (2025-04-04)

Full Changelog: [v0.2.0-alpha.50...v0.2.0-alpha.51](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.50...v0.2.0-alpha.51)

### Chores

* **internal:** remove trailing character ([#445](https://github.com/openlayer-ai/openlayer-python/issues/445)) ([6ccac8e](https://github.com/openlayer-ai/openlayer-python/commit/6ccac8e6d3eee06c4f1241f4dc0a9104a48d1841))

## 0.2.0-alpha.50 (2025-04-02)

Full Changelog: [v0.2.0-alpha.49...v0.2.0-alpha.50](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.49...v0.2.0-alpha.50)

### Features

* feat: add async openai tracer ([6d8bc02](https://github.com/openlayer-ai/openlayer-python/commit/6d8bc020c41cdbd43fc47127b0bb34b72e449fd9))


### Chores

* fix typos ([#441](https://github.com/openlayer-ai/openlayer-python/issues/441)) ([987d427](https://github.com/openlayer-ai/openlayer-python/commit/987d42797440477a7fe113e9ac5de1ee686e097b))

## 0.2.0-alpha.49 (2025-03-21)

Full Changelog: [v0.2.0-alpha.48...v0.2.0-alpha.49](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.48...v0.2.0-alpha.49)

### Features

* chore: add OpenLLMetry tracing example ([ff13020](https://github.com/openlayer-ai/openlayer-python/commit/ff13020ee4c7ea9cadd4cc0af0604debe706b599))
* chore: add Semantic Kernel tracing example ([98ada7f](https://github.com/openlayer-ai/openlayer-python/commit/98ada7f7993b3163844c80604a81a75f37d30616))

## 0.2.0-alpha.48 (2025-03-18)

Full Changelog: [v0.2.0-alpha.47...v0.2.0-alpha.48](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.47...v0.2.0-alpha.48)

### Features

* feat: add option to wait for commit completion to push function ([b3b4afd](https://github.com/openlayer-ai/openlayer-python/commit/b3b4afd998c28df816f4223fc0eebc2ab0882b8b))
* feat: add wait_for_commit_completion convenience method ([f71e29a](https://github.com/openlayer-ai/openlayer-python/commit/f71e29af2602d5eb08a88de02f834a5f654aeec8))

## 0.2.0-alpha.47 (2025-03-17)

Full Changelog: [v0.2.0-alpha.46...v0.2.0-alpha.47](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.46...v0.2.0-alpha.47)

### Bug Fixes

* **ci:** remove publishing patch ([#433](https://github.com/openlayer-ai/openlayer-python/issues/433)) ([c30bf64](https://github.com/openlayer-ai/openlayer-python/commit/c30bf64ebb1e47d754aed02ca256cd9bec71542b))


### Chores

* **internal:** codegen related update ([#432](https://github.com/openlayer-ai/openlayer-python/issues/432)) ([98ac8ac](https://github.com/openlayer-ai/openlayer-python/commit/98ac8ac29f78f3847a859b474b073667f677bc22))

## 0.2.0-alpha.46 (2025-03-15)

Full Changelog: [v0.2.0-alpha.45...v0.2.0-alpha.46](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.45...v0.2.0-alpha.46)

### Features

* **api:** api update ([10f1de0](https://github.com/openlayer-ai/openlayer-python/commit/10f1de0a71b489ec6e479af5fd8c33bc4f2cc63a))
* **client:** allow passing `NotGiven` for body ([6a582f7](https://github.com/openlayer-ai/openlayer-python/commit/6a582f73748f4c628cd55dd4781792f8ba82426b))
* **client:** send `X-Stainless-Read-Timeout` header ([919377e](https://github.com/openlayer-ai/openlayer-python/commit/919377ee8e73ad8ca39d5cead7f85c3e934b7bc1))


### Bug Fixes

* asyncify on non-asyncio runtimes ([1aa358a](https://github.com/openlayer-ai/openlayer-python/commit/1aa358aefbee3ddb9c401eb3e6838b063ba26f1e))
* **client:** mark some request bodies as optional ([6a582f7](https://github.com/openlayer-ai/openlayer-python/commit/6a582f73748f4c628cd55dd4781792f8ba82426b))
* **tests:** correctly generate examples with writeOnly fields ([aefb7d9](https://github.com/openlayer-ai/openlayer-python/commit/aefb7d93a78f972467a3f70a17c06d9e451817b8))
* **types:** handle more discriminated union shapes ([#431](https://github.com/openlayer-ai/openlayer-python/issues/431)) ([3a8b9c1](https://github.com/openlayer-ai/openlayer-python/commit/3a8b9c104e28589248d3208f92d8cda3bee1364e))


### Chores

* **internal:** bummp ruff dependency ([a85525a](https://github.com/openlayer-ai/openlayer-python/commit/a85525a6cc9e3ac81ba1cd5fb534e120c1580067))
* **internal:** bump rye to 0.44.0 ([#430](https://github.com/openlayer-ai/openlayer-python/issues/430)) ([9fe86fe](https://github.com/openlayer-ai/openlayer-python/commit/9fe86fef481775181a52d3e4f9249c4405d4bb24))
* **internal:** change default timeout to an int ([32452f0](https://github.com/openlayer-ai/openlayer-python/commit/32452f0ac8f3a321a81fb7bd340fa6ced4c5c648))
* **internal:** codegen related update ([dfd7861](https://github.com/openlayer-ai/openlayer-python/commit/dfd7861657bbd5f761649b5f956cb9c85e9bd1e4))
* **internal:** codegen related update ([c87c92d](https://github.com/openlayer-ai/openlayer-python/commit/c87c92ded5591542b9c939c775fa2d09fb0885c5))
* **internal:** codegen related update ([#425](https://github.com/openlayer-ai/openlayer-python/issues/425)) ([ec47eb9](https://github.com/openlayer-ai/openlayer-python/commit/ec47eb9f03007a5efa8c194ab98d0aa1377720b9))
* **internal:** codegen related update ([#429](https://github.com/openlayer-ai/openlayer-python/issues/429)) ([395275b](https://github.com/openlayer-ai/openlayer-python/commit/395275b0f996f2b4eb49857530e72f9fe64b853a))
* **internal:** fix devcontainers setup ([9bc507d](https://github.com/openlayer-ai/openlayer-python/commit/9bc507d3197627087b7139ee3c2f9e28c4075c95))
* **internal:** fix type traversing dictionary params ([df06aaa](https://github.com/openlayer-ai/openlayer-python/commit/df06aaa91ee17410b96b28e897c5559f67cbc829))
* **internal:** fix workflows ([1946b4f](https://github.com/openlayer-ai/openlayer-python/commit/1946b4f202142fe9a58c11d5f74870def6582d9b))
* **internal:** minor type handling changes ([a920965](https://github.com/openlayer-ai/openlayer-python/commit/a92096519c3a1d2ecaad5595029231faeafb09ed))
* **internal:** properly set __pydantic_private__ ([0124a23](https://github.com/openlayer-ai/openlayer-python/commit/0124a2338534da8f0d707d9c6d6f5e5576d6999f))
* **internal:** remove extra empty newlines ([#428](https://github.com/openlayer-ai/openlayer-python/issues/428)) ([7111d6d](https://github.com/openlayer-ai/openlayer-python/commit/7111d6d4a8a8524aadbc402ea4761dba2b377170))
* **internal:** update client tests ([c7a8995](https://github.com/openlayer-ai/openlayer-python/commit/c7a899524ea9b3ff1218a0e03868a8647ee46a08))

## 0.2.0-alpha.45 (2025-03-13)

Full Changelog: [v0.2.0-alpha.44...v0.2.0-alpha.45](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.44...v0.2.0-alpha.45)

### Features

* **api:** add endpoint to retrieve commit by id ([#421](https://github.com/openlayer-ai/openlayer-python/issues/421)) ([d7c8489](https://github.com/openlayer-ai/openlayer-python/commit/d7c84892a258c15b23fac3dedd2c074357595613))

## 0.2.0-alpha.44 (2025-02-26)

Full Changelog: [v0.2.0-alpha.43...v0.2.0-alpha.44](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.43...v0.2.0-alpha.44)

### Features

* feat(tracing): completes OPEN-6538 Surface root step metadata at the request level ([1bcedcf](https://github.com/openlayer-ai/openlayer-python/commit/1bcedcf57d509064f89e2a5fae3fb39f22da5920))

## 0.2.0-alpha.43 (2025-02-24)

Full Changelog: [v0.2.0-alpha.42...v0.2.0-alpha.43](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.42...v0.2.0-alpha.43)

### Features

* chore: update download URL for context file ([6835d38](https://github.com/openlayer-ai/openlayer-python/commit/6835d389fd250546bfa13bb054843d7d6c769ebd))

## 0.2.0-alpha.42 (2024-12-18)

Full Changelog: [v0.2.0-alpha.41...v0.2.0-alpha.42](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.41...v0.2.0-alpha.42)

### Features

* **api:** api update ([#412](https://github.com/openlayer-ai/openlayer-python/issues/412)) ([f6ca1fc](https://github.com/openlayer-ai/openlayer-python/commit/f6ca1fcbc7ed85d6e3bdc635b8f7a4796c943e2a))


### Chores

* **internal:** codegen related update ([#406](https://github.com/openlayer-ai/openlayer-python/issues/406)) ([3360b9e](https://github.com/openlayer-ai/openlayer-python/commit/3360b9e6f6037c7bc9ce877f7ae430ca249e9b95))
* **internal:** codegen related update ([#408](https://github.com/openlayer-ai/openlayer-python/issues/408)) ([9bab516](https://github.com/openlayer-ai/openlayer-python/commit/9bab5168085e325ac7b8b4f07643f39ef564d78d))
* **internal:** codegen related update ([#409](https://github.com/openlayer-ai/openlayer-python/issues/409)) ([f59c50e](https://github.com/openlayer-ai/openlayer-python/commit/f59c50ebd7b298536f0a6a92437630551074e172))
* **internal:** codegen related update ([#410](https://github.com/openlayer-ai/openlayer-python/issues/410)) ([7e4304a](https://github.com/openlayer-ai/openlayer-python/commit/7e4304a87d8330fc15b099a078412f0dbab78842))
* **internal:** fix some typos ([#414](https://github.com/openlayer-ai/openlayer-python/issues/414)) ([1009b11](https://github.com/openlayer-ai/openlayer-python/commit/1009b11b627a4236137c76543e2a09cc4fc78557))
* **internal:** updated imports ([#411](https://github.com/openlayer-ai/openlayer-python/issues/411)) ([90c6218](https://github.com/openlayer-ai/openlayer-python/commit/90c6218e0a9929f8672da20f1871f20aab9bb500))


### Documentation

* **readme:** example snippet for client context manager ([#413](https://github.com/openlayer-ai/openlayer-python/issues/413)) ([4ef9f75](https://github.com/openlayer-ai/openlayer-python/commit/4ef9f75dfea53f198af9768414b51027ec9bd553))

## 0.2.0-alpha.41 (2024-12-13)

Full Changelog: [v0.2.0-alpha.40...v0.2.0-alpha.41](https://github.com/openlayer-ai/openlayer-python/compare/v0.2.0-alpha.40...v0.2.0-alpha.41)

### Chores

* **internal:** add support for TypeAliasType ([#404](https://github.com/openlayer-ai/openlayer-python/issues/404)) ([42da61a](https://github.com/openlayer-ai/openlayer-python/commit/42da61a02c4db5b87b326b1a2b3a1e0df3757d59))
* **internal:** bump pyright ([#402](https://github.com/openlayer-ai/openlayer-python/issues/402)) ([a2fe31a](https://github.com/openlayer-ai/openlayer-python/commit/a2fe31a2aff4d7cd18014d4f135fa137a8649e00))

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
