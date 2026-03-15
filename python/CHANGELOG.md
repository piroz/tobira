# Changelog

## [0.1.0](https://github.com/piroz/tobira/compare/tobira-python-v0.0.1...tobira-python-v0.1.0) (2026-03-15)


### Features

* **adversarial:** add homoglyph and invisible Unicode detectors ([#49](https://github.com/piroz/tobira/issues/49)) ([04116e6](https://github.com/piroz/tobira/commit/04116e6fa8c6e53944f93420e5118edd035e0881))
* **backends:** add BackendProtocol and FastTextBackend implementation ([b54d87d](https://github.com/piroz/tobira/commit/b54d87d1c93db6400350291d459d33ea3a02348f)), closes [#3](https://github.com/piroz/tobira/issues/3)
* **backends:** add BertBackend implementation ([1a10267](https://github.com/piroz/tobira/commit/1a10267fd0b2ff1cbb57ec943b804c2b0a6e57cb))
* **backends:** add ensemble backend ([65c9c5b](https://github.com/piroz/tobira/commit/65c9c5b62b6ad0bb277e4bb3fe58e85d71cad50e))
* **backends:** add multiclass spam classification (8 subcategories) ([#55](https://github.com/piroz/tobira/issues/55)) ([abbe6ff](https://github.com/piroz/tobira/commit/abbe6ffa917235a051c3fd76c1badaa00ad9503c))
* **backends:** add Ollama and OpenAI-compatible LLM API backends ([8f93597](https://github.com/piroz/tobira/commit/8f93597074d32e54d4889472cef85c4e950a1788))
* **backends:** add ONNX export and OnnxBackend ([6931db0](https://github.com/piroz/tobira/commit/6931db0a8bd9cf4604685af9dacad66cc4970a9e))
* **backends:** add TwoStageBackend for two-stage filtering ([3af957d](https://github.com/piroz/tobira/commit/3af957de99cbef67785ddb57dfe54b606f08811b))
* **cli:** add CLI framework with tobira serve subcommand ([bad1eb7](https://github.com/piroz/tobira/commit/bad1eb7f1b2e9e00ceffb03d00845b46233aa141))
* **cli:** add tobira doctor subcommand ([d814dcb](https://github.com/piroz/tobira/commit/d814dcbac8deb2a2bdcde221baeec89eeb39dc35))
* **cli:** add tobira evaluate subcommand  ([add9efb](https://github.com/piroz/tobira/commit/add9efb91151fc16bf3cea56c0dbe7ce9fb1a1aa))
* **cli:** add tobira init subcommand ([ef3980d](https://github.com/piroz/tobira/commit/ef3980d3ce3e432fd4e40aa6071fbb7d34136115))
* **cli:** add tobira monitor subcommand ([4c1c812](https://github.com/piroz/tobira/commit/4c1c812fa3003188f899eaf7fd4bbdd949d7ae6e))
* **evaluation:** add benchmark suite for backend comparison ([#54](https://github.com/piroz/tobira/issues/54)) ([3f6fa0a](https://github.com/piroz/tobira/commit/3f6fa0a8992c79bd6e5d3ec08d03759bdde198ac))
* **evaluation:** add evaluation module ([5866562](https://github.com/piroz/tobira/commit/5866562aa4734e3667a97227baf77756e7da63cd))
* **hub:** add HuggingFace Hub model upload and download support ([#53](https://github.com/piroz/tobira/issues/53)) ([9848acb](https://github.com/piroz/tobira/commit/9848acb17d1a9d2875a565ef94bbb604cac0ec01))
* **monitoring:** add concept drift detection with Redis statistics ([#50](https://github.com/piroz/tobira/issues/50)) ([d7c9e8f](https://github.com/piroz/tobira/commit/d7c9e8f27954ba0c855447aba55f18ba5ea0ac45))
* **monitoring:** add prediction metrics logging middleware ([5dac908](https://github.com/piroz/tobira/commit/5dac908c63534e490504200e97bca729fe47cd27))
* **multilingual:** add language detection and multilingual support ([#52](https://github.com/piroz/tobira/issues/52)) ([83e8143](https://github.com/piroz/tobira/commit/83e814349befca4ebc885c25c087c77a6164ada2))
* **pipeline:** add feedback loop endpoint and JSONL store ([#51](https://github.com/piroz/tobira/issues/51)) ([f44ae3b](https://github.com/piroz/tobira/commit/f44ae3baf371dff8f2ff0c30ed5fd2eef2b57449))
* **preprocessing,data:** add PII anonymization and synthetic data generation ([4637443](https://github.com/piroz/tobira/commit/46374434e7e34e002b2f244852cabf286325c0c7))
* **serving:** add FastAPI server with POST /predict and GET /health ([bcc4a9b](https://github.com/piroz/tobira/commit/bcc4a9b0014295f618df2c363716747ef97700af))
* **serving:** add request size limit and OpenAPI schema improvements ([b681157](https://github.com/piroz/tobira/commit/b6811578ed25b4b9d3bdc0a1069621b1b23f3800))


### Bug Fixes

* **backends:** select highest-score label in FastTextBackend.predict() ([4154cfc](https://github.com/piroz/tobira/commit/4154cfc4892481f02ad6770d6355c56a71b9c3a4))


### Code Refactoring

* **backends:** make backend deps optional and improve error handling ([6879566](https://github.com/piroz/tobira/commit/6879566664acad71058527229df6cf9730738931))


### Tests

* **python:** expand test coverage and introduce pytest-cov ([57f1d17](https://github.com/piroz/tobira/commit/57f1d17141e07d50aa7a319a59a3023ea782af81))


### Documentation

* add bilingual naming with door emoji to README and GitHub Pages ([946c3d9](https://github.com/piroz/tobira/commit/946c3d9bbeabe9d0ba2852487017642517eb2d1b))


### CI

* add GitHub Actions workflows for Python and JS testing ([1afd80f](https://github.com/piroz/tobira/commit/1afd80f8a5c00c26973498e47bbc0903e852d47d))


### Miscellaneous

* **python:** add mypy type checking ([7b7ae6b](https://github.com/piroz/tobira/commit/7b7ae6b864d5b73bce39406144cc2daa00287e08)), closes [#14](https://github.com/piroz/tobira/issues/14)
