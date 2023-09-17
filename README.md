# pydanttention

[![PyPI](https://img.shields.io/pypi/v/pydanttention?logo=python&logoColor=%23cccccc)](https://pypi.org/project/pydanttention)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/lmmx/pydanttention/master.svg)](https://results.pre-commit.ci/latest/github/lmmx/pydanttention/master)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/pydanttention.svg)](https://pypi.org/project/pydanttention)

<!-- [![build status](https://github.com/lmmx/pydanttention/actions/workflows/master.yml/badge.svg)](https://github.com/lmmx/pydanttention/actions/workflows/master.yml) -->

Transformer model attention in Pydantic.

Adapted from the source by Theia Vogel (MIT licensed, included here as `vogel_manual_transformer.py`):

- [I made a transformer by hand (no training!)](https://vgel.me/posts/handmade-transformer/) (2023)

In turn using model ops from [picoGPT](https://github.com/jaymody/picoGPT/blob/main/gpt2.py) (MIT license)

## Motivation

Rewriting AI model source code as Pydantic data models is an interesting exercise. I'd note the following benefits.

- All operations can be subclassed from an arbitrary `Operation` model (see `.models.ops.base`),
  i.e. they all expect their first argument to be a numpy array `x`. This naturally allows you to
  factor your code around a category of 'operations'.

- Since all functions get turned into a class (a Pydantic data model with type-annotated fields for
  input state rather than funcdef kw/args), and classes are conventionally named in `PascalCase` whereas functions
  (like all other Python variables) are conventionally named in `snake_case`, you can easily observe from case alone
  where significant operations are called, as well as where the data model is referenced (by `self.{field}`) making
  these 2 types of data access distinct from the intermediate variables. This gives a better sense
  at a glance of data flow through your program.

- State can be configured at runtime but also given defaults at import time through use of fields in
  the data model. The original source code hardcoded values in the config as module globals
  (similarly to using class variables), it was not possible to configure component parts at runtime.
  This was appropriate to author an expository demo, but made it difficult to approach as a reader
  wishing to modify and experiment (likewise code is easier to test if easier to configure at runtime).

- Clear and consolidated declarations of input data (i.e. not scattered across many sites of declaration)
  without losing the ability to decompose into structured components. The original code used primitive types
  (lists of dictionaries) for the attention blocks, which became model field defaults in a self-contained module (see `.models.config`).
  Since Pydantic allows you to load ("validate") typed data models from these primitive types, we
  could supply the original dictionary primitive to `AttentionBlock.model_validate` and it'd still work
  (but doing so is actually more verbose than just constructing the model class directly).

## Installation

```py
pip install pydanttention
```

## Usage

```
from pydanttention import Transformer

model = Transformer(report=True)
model.run()
```

This will reproduce the results of the demo from the original blog post:

```
...
b (1): next=a (0) probs=[1. 0.] logits=[1.024e+03 1.000e+00]
a (0): next=a (0) probs=[1. 0.] logits=[1025.    0.]
ACCURACY: 100.0% (27 / 27)
```

See `.main:Transformer` and `models.config:Config` for the available options that can be set on `Transformer`.


## Development

- To set up pre-commit hooks (to keep the CI bot happy) run `pre-commit install-hooks` so all git
  commits trigger the pre-commit checks. I use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
  This runs `black`, `flake8`, `autopep8`, `pyupgrade`, etc.

- To set up a dev env, I first create a new conda environment and use it in PDM with `which python > .pdm-python`.
  To use `virtualenv` environment instead of conda, skip that. Run `pdm install` and a `.venv` will be created if no
  Python binary path is found in `.pdm-python`.

- To run tests, run `pdm run python -m pytest` and the PDM environment will be used to run the test suite.
