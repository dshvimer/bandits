# Multi-armed Bandits

## Setup

Python's poetry package manager is required to run this project. On macOS install it with

```
brew install poetry
```

Installing project dependencies:

```
poetry install
```

In order to perform analysis, we need to run the experiments. Each experiment will create a CSV of data we can examine.

To run the experiments:

```
poetry run testbed
poetry run ads
```

To run the notebooks:

```
poetry run jupyter notebook
```
