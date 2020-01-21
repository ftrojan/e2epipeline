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

We start with showing the design principle #1. You do not need to worry replacing your Scikit-learn pipeline with the `E2EPipeline`. They both work in the same way and provide the same results. Let us demonstrate this on a task of text classification. The dataset [`all_tickets_906.csv`](data/all_tickets_906.csv) contains 906 customer tickets classified into a product hierarchy. The text body of each ticket is stored in the column named `body` and the hierarchical product path (the target) is stored in the column `target_path`. To fit a simple linear model we can use for example the following Scikit-learn pipeline:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

df = pd.read_csv('all_tickets_906.csv')
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        sublinear_tf=True,
        min_df=5,
        norm='l2',
        ngram_range=(1, 2),
        stop_words='english',
    )),
    ('svd', TruncatedSVD(
        n_components=100,
        random_state=11,
    )),
    ('svc', CalibratedClassifierCV(
        base_estimator=svm.LinearSVC(random_state=33),
        cv=5,
    )),
])
x = df['body']
y = df['target_path']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
``` 

Now let us do the same, just instead of `Pipeline` imported from `sklearn.pipeline` use `E2EPipeline` imported from `e2epipeline`. We will also set up logging as we may have been used to. The new Python code is following:

```python
import logging
from e2epipeline import E2EPipeline

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
pipeline_e2e = E2EPipeline([
    ('tfidf', TfidfVectorizer(
        sublinear_tf=True,
        min_df=5,
        norm='l2',
        ngram_range=(1, 2),
        stop_words='english',
    )),
    ('svd', TruncatedSVD(
        n_components=100,
        random_state=11,
    )),
    ('svc', CalibratedClassifierCV(
        base_estimator=svm.LinearSVC(random_state=33),
        cv=5,
    )),
])
pipeline_e2e.fit(x_train, y_train)
y_e2e = pipeline_e2e.predict(x_test)
```

You can copy and paste the code into a Jupyter notebook and run it to see what happens if we compare the two predictions.

```python
all(y_pred == y_e2e)
```
True

The predictions are exactly the same, because the all the transformers and estimators we have used are standard Scikit-learn components and the design principle #1 holds.

There is one little difference though. While the Scikit-learn does not support the standard Python logging (it is the second oldest unresolved [issue #78](https://github.com/scikit-learn/scikit-learn/issues/78) in their issue list), the E2EPipeline produces log with good balance between too much detail and too little information. The log from our text classifier looks like following:

```python
2020-01-03 19:09:51,347 - DEBUG - e2epipeline.fit - started, args=tuple2 = (pd.Series(724), pd.Series(724)), kwargs=dict0 = {}
2020-01-03 19:09:51,349 - DEBUG - e2epipeline.fit - step 0 (tfidf): dict2 = {0: pd.Series(724), 1: pd.Series(724)}
2020-01-03 19:09:51,351 - DEBUG - e2epipeline.fit - step 0 (tfidf): fit_args=list(2) fit_kwargs=dict0 = {}
2020-01-03 19:09:51,454 - DEBUG - e2epipeline.fit - step 0 (tfidf): dict2 = {0: scipy.sparse(724, 1097), 1: pd.Series(724)}
2020-01-03 19:09:51,454 - DEBUG - e2epipeline.fit - step 1 (svd): dict2 = {0: scipy.sparse(724, 1097), 1: pd.Series(724)}
2020-01-03 19:09:51,455 - DEBUG - e2epipeline.fit - step 1 (svd): fit_args=list(2) fit_kwargs=dict0 = {}
2020-01-03 19:09:51,501 - DEBUG - e2epipeline.fit - step 1 (svd): dict2 = {0: np.array(724, 100), 1: pd.Series(724)}
2020-01-03 19:09:51,501 - DEBUG - e2epipeline.fit - step 2 (svc): dict2 = {0: np.array(724, 100), 1: pd.Series(724)}
2020-01-03 19:09:51,502 - DEBUG - e2epipeline.fit - step 2 (svc): fit_args=list(2) fit_kwargs=dict0 = {}
2020-01-03 19:09:51,729 - DEBUG - e2epipeline.fit - step 2 (svc): dict2 = {0: np.array(724,), 1: pd.Series(724)}
2020-01-03 19:09:51,729 - DEBUG - e2epipeline.fit - finished
2020-01-03 19:09:51,730 - DEBUG - e2epipeline.predict - started
2020-01-03 19:09:51,732 - DEBUG - e2epipeline.predict - step 0 (tfidf): dict1 = {0: pd.Series(182)}
2020-01-03 19:09:51,749 - DEBUG - e2epipeline.predict - step 0 (tfidf): dict1 = {0: scipy.sparse(182, 1097)}
2020-01-03 19:09:51,749 - DEBUG - e2epipeline.predict - step 1 (svd): dict1 = {0: scipy.sparse(182, 1097)}
2020-01-03 19:09:51,751 - DEBUG - e2epipeline.predict - step 1 (svd): dict1 = {0: np.array(182, 100)}
2020-01-03 19:09:51,752 - DEBUG - e2epipeline.predict - step 2 (svc): dict1 = {0: np.array(182, 100)}
2020-01-03 19:09:51,756 - DEBUG - e2epipeline.predict - step 2 (svc): dict1 = {0: np.array(182,)}
2020-01-03 19:09:51,760 - DEBUG - e2epipeline.predict - finished
```

### Filter out Infrequent Classes

TBD

### Add Confidence to the Output

TBD

### Keep Intermediate Results By Name

TBD

### Keep Intermediate Results By Position

TBD