[![Build Status](https://travis-ci.com/ftrojan/e2epipeline.svg?branch=master)](https://travis-ci.com/ftrojan/e2epipeline)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

# e2epipeline
E2EPipeline is a generalisation of sklearn Pipeline to allow for more flexible mapping of input and output parameters.

## Introduction

Scikit-learn is excellent package and its `Pipeline` implementation is widely used among data scientists. It is easy to use yet powerful for the most of the machine learning use cases. There are however use cases which cannot be implemented by the standard scikit-learn pipeline. Examples of such use cases are:

- We need a transformer which removes observations from the training dataset with missing values, like in this [Stackoverflow question](https://stackoverflow.com/questions/25539311/custom-transformer-for-sklearn-pipeline-that-alters-both-x-and-y).
- We need to filter out infrequent classes for our multiclass problem, like in this [Stackoverflow question](https://stackoverflow.com/questions/55948491/scikitlearn-remove-less-frequent-categorical-classes/55949408).
- We want to perform oversampling or undersampling of our imbalanced training dataset, like in this [Stackoverflow question](https://stackoverflow.com/questions/54118076/how-to-resample-text-imbalanced-groups-in-a-pipeline).
- We want to include [propositionalization](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_680) (in my community we use term flattening which is easier to pronounce) into the pipeline. In this scenario the input to the pipeline is not tuple of `X, y`, but a relational database from which we can calculate the `X` and `y` by a data transformation.

Currently the Scikit-learn cannot handle those issues in pipeline - see [Scikit-learn issue](https://github.com/scikit-learn/scikit-learn/issues/3855) for the evidence that it is unresolved. This is why the [`sklearn.utils.resample`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html) is a utility function and not the proper transformer.
The reason for which the standard Scikit-learn `Pipeline` fails on those use cases lies in its core design principle, which can be summarised as follows.

During the `Pipeline.fit(p1, p2)` execution, in each step both `fit(X, y)` and `transform(X, y)` methods are called with 
- `X` being the result of the `transform` from the preceding step (or `p1` for the first step), and
- `y` being the original second positional argument `p2`.

With this pipeline logic it is impossible to transform both `X` and `y`.
The E2EPipeline addresses this issue by implementing a more flexible pipeline logic which keeps the simplicity of `sklearn.Pipeline`, but allows for customisation in terms of how inputs and outputs of pipeline steps are mapped to each other.

## Design Principles

E2EPipeline was designed to be simple to use. The design principle #1 is that any Scikit-learn pipeline must work also as E2EPipeline directly as is. This means that a user familiar with Scikit-learn can take his pipeline and just the class name the pipeline must fit and predict in the very same way.

The design principle #2 is the state. Each E2EPipeline has a state, which represents the data flowing through the pipeline. The state is changed at each step of the pipeline in the `transform` method. It is a Python dictionary with keys of two kinds:
- integer keys for positional arguments to transformers and estimators,
- string keys for named arguments to transformers and estimators.

When a transformer or an estimator is called in a pipeline step, we analyse the method signature and match the input arguments by name and by position from the current state. The output from a transformer is parsed to the state by the following rules:
- if the output is a tuple, it is parsed element by element with integer keys starting from 0,
- if the output is a dictionary, it is parsed element by element with the same keys,
- otherwise the output is parsed as `state[0]`.

In the `fit`, the state is initialised by the input arguments, depending whether they are positional or named. In the `predict`, the state is initialised also by the input arguments in the very same way.   

Having said that, we can easily see what happens with standard Scikit-learn pipeline during the standard `E2EPipeline.fit(p1, p2)` execution. The state is initialised as `state = {0: p1, 1: p2}` and in each step both `fit(X, y)` and `transform(X, y)` methods are called with 
- `X` being the `state[0]`, which is result of the `transform` from the preceding step (or `p1` for the first step), and
- `y` being the `state[1]`, which is always the original second positional argument `p2`.

The design principle #3 is extensibility. To achieve the desired flexibility you can either make your own transformers (wrap an existing transformer), or extend an existing transformer by introducing a simple preprocessing and/or postprocessing mapping. When you are working with a third party transformer, in most cases you should not need to wrap it, but simple pre/postprocessing should be enough. More on how to use pre/postprocessing is in the tutorial.

## Tutorial

### Installation and Import

Install the package directly from Github using the following shell command
```bash
pip install git+https://github.com/ftrojan/e2epipeline.git
```

Basic recommended import statement is
```python
from e2epipeline import E2EPipeline
```

You might find useful to import also class `Step` and functions `rename`, `name`, `reposition`, `position` for more advanced usage. You will learn about those later in this tutorial, but for the basic functionality they are not necessary.

### Standard Scikit-learn Pipeline

TBD

### Filter out Infrequent Classes

TBD

### Add Confidence to the Output

TBD

### Keep Intermediate Results By Name

TBD

### Keep Intermediate Results By Position

TBD