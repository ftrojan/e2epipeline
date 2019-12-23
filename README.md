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

Currently the Scikit-learn cannot handle those issues in pipeline - see [Scikit-learn issue](https://github.com/scikit-learn/scikit-learn/issues/3855) for the evidence that it is unresolved.
The reason for which the standard Scikit-learn `Pipeline` fails on those use cases lies in its core design principle, which can be summarised as follows.

During the `Pipeline.fit(p1, p2)` execution, in each step both `fit(X, y)` and `transform(X, y)` methods are called with 
- `X` being the result of the `transform` from the preceding step (or `p1` for the first step), and
- `y` being the original second positional argument `p2`.

With this pipeline logic it is impossible to transform both `X` and `y`. This is also reason why the [`sklearn.utils.resample`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html) is a utility function and not the proper transformer.

The E2EPipeline addresses this issue by implementing a more flexible pipeline logic which keeps the simplicity of `sklearn.Pipeline`, but allows for customisation in terms of how inputs and outputs of pipeline steps are mapped to each other.

## Design Principles

TBD

## Tutorial

TBD