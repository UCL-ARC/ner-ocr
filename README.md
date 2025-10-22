# NER OCR

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE.md)

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/macknix/ner-ocr/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/macknix/ner-ocr/actions/workflows/tests.yml
[linting-badge]:            https://github.com/macknix/ner-ocr/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/macknix/ner-ocr/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/macknix/ner-ocr/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/macknix/ner-ocr/actions/workflows/docs.yml
[license-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

A pipeline for NER using OCR

This project is developed in collaboration with the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University
College London.

## About

### Project Team

Mack Nixon ([mack.nixon@ucl.ac.uk](mailto:mack.nixon@ucl.ac.uk))

### Research Software Engineering Contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`ner-ocr` requires Python 3.13&ndash;3.11.

### Installation

<!-- How to build or install the application. -->

### Installing `uv`

[uv](https://docs.astral.sh/uv) is used for Python dependency management and managing virtual environments. You can install uv either using pipx or the uv installer script:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installing Dependencies

Once uv is installed, install dependencies:

```sh
uv sync
```

### Activate your Python environment

```sh
source .venv/bin/activate
```

### Installing pre-commit hooks

Install `pre-commit` locally (in your activated `venv`) to aid code consistency (if you're looking to contribute).

```sh
pre-commit install
```

## Roadmap

- [ ] Initial Research
- [ ] Minimum viable product <-- You are Here
- [ ] Alpha Release
- [ ] Feature-Complete Release
