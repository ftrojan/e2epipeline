# Filter out Infrequent Classes

In this example we will discover the limits of the Scikit-learn pipeline and learn how to solve the problem using `E2EPipeline`. Let us fit our Scikit-learn pipeline from the [example 1](#standard-scikit-learn-pipeline) on a different dataset. Let us use the [`all_tickets_1000.csv`](../data/all_tickets_1000.csv) now. This dataset contains additional observations with small classes (low number of examples).

```python
df = pd.read_csv('all_tickets_1000.csv')
pipeline.fit(x_train, y_train)
```

The fit fails with the 

```ValueError: Requesting 5-fold cross-validation but provided less than 5 examples for at least one class.```

Indeed, the `CalibratedClassifierCV` cannot calibrate the probability if a class is too small. We might want to exclude those small classes from the training dataset. To achieve that, we want to introduce a new transformer as the very first step of our pipeline. This transformer will exclude all the examples of small classes (below a threshold) in the `fit` method, while keep all the examples in the `transform` method. It will be a custom transformer which removes the selected examples from both `X` and `y`. The definition is following.

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

class InfrequentClassFilter(BaseEstimator):

    def __init__(self, size_threshold: int):
        self.size_threshold = size_threshold
        self.classes_to_keep_ = None

    def fit(self, X, y):
        y_array = y.values if isinstance(y, pd.Series) else y
        unique, counts = np.unique(y_array, return_counts=True)
        self.classes_to_keep_ = unique[counts >= self.size_threshold]
        return self

    def transform(self, X, y=None):
        if y is None:
            return (X, )
        else:
            assert self.classes_to_keep_ is not None, "transformer is not fitted yet"
            y_array = y.values if isinstance(y, pd.Series) else y
            boolean_index = np.isin(y_array, self.classes_to_keep_)
            if X is None:
                return None, y[boolean_index]
            else:
                assert X.shape[0] == y.shape[0]
                return X[boolean_index], y[boolean_index]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
```

Note that the `transform` method has two branches. The `y is None` branch will be called in the `pipeline.predict(x_test)` when there is no `y` argument supplied. The `y is not None` branch will be called in the `pipeline.fit` when the `y` argument is supplied. The output will be always a tuple, either with just one element (if just `X` is supplied) or two elements (if both `X` and `y` are supplied).

Try the new transformer on our train dataset and see what happens.

```python
icf = InfrequentClassFilter(size_threshold=50)
res = icf.fit_transform(x_train, y_train)
print(f"res=({res[0].shape}, {res[1].shape})")
```

`res=((466,), (466,))`

The output shows that with `size_threshold=50` the new transformer reduces both `x_train` and `y_train` from original 724 examples down to 466 examples, keeping only the classes with at least 50 examples each.

Let us put now the transformer as the very first step into our Scikit-learn pipeline.

```python
pipeline = Pipeline([
    ('icf', InfrequentClassFilter(
        size_threshold=50,
    )),
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
        cv=5
    )),
])
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
```

If you run the code, you will observe that `pipeline.fit` fails with `AttributeError: 'Series' object has no attribute 'lower'`. What happened?

The transformed tuple `(X, y)` from the `icf` transformer was mapped to the input of `tfidf` in a wrong way. Instead of mapping output `X` to input `raw_documents` and output `y` to input `y`, the tuple `(X, y)` was mapped to input `raw_documents` and the input `y` was supplied as the original `y_train`, without any examples removed. The tokenizer in the `icf` started iterating over the `raw_documents` and realized that the first element (which is the output `X` as `pd.Series`) cannot be lower-cased (which is the first step in the tokenization). This issue is broadly discussed in Scikit-learn community (see [issue #3855](https://github.com/scikit-learn/scikit-learn/issues/3855)) since 2014, but it is very difficult to change it, because it would not be backward compatible. The issue is not resolved yet as of 2020.

Let us see if we are lucky when we replace the `Pipeline` with the `E2EPipeline`:

```python
pipeline_e2e = E2EPipeline([
    ('icf', InfrequentClassFilter(
        size_threshold=50,
    )),
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
        cv=5
    )),
])
pipeline_e2e.fit(x_train, y_train)
y_e2e = pipeline_e2e.predict(x_test)
```

There is no error now and the pipeline works in the way we wanted. The mapping between the `icf` output and the `tfidf` input is positional, so the output `X` is mapped to input `raw_documents` and the output `y` is mapped to the input `y`. The rest of the pipeline works in the very same way as in [example 1](example1.md). The output log which documents what happened is the following.

```text
2020-01-03 19:09:04,957 - DEBUG - e2epipeline.fit - started, args=tuple2 = (pd.Series(800), pd.Series(800)), kwargs=dict0 = {}
2020-01-03 19:09:04,958 - DEBUG - e2epipeline.fit - step 0 (icf): dict2 = {0: pd.Series(800), 1: pd.Series(800)}
2020-01-03 19:09:04,960 - DEBUG - e2epipeline.fit - step 0 (icf): fit_args=list(2) fit_kwargs=dict0 = {}
2020-01-03 19:09:04,967 - DEBUG - e2epipeline.fit - step 0 (icf): dict2 = {0: pd.Series(466), 1: pd.Series(466)}
2020-01-03 19:09:04,968 - DEBUG - e2epipeline.fit - step 1 (tfidf): dict2 = {0: pd.Series(466), 1: pd.Series(466)}
2020-01-03 19:09:04,971 - DEBUG - e2epipeline.fit - step 1 (tfidf): fit_args=list(2) fit_kwargs=dict0 = {}
2020-01-03 19:09:05,044 - DEBUG - e2epipeline.fit - step 1 (tfidf): dict2 = {0: scipy.sparse(466, 469), 1: pd.Series(466)}
2020-01-03 19:09:05,045 - DEBUG - e2epipeline.fit - step 2 (svd): dict2 = {0: scipy.sparse(466, 469), 1: pd.Series(466)}
2020-01-03 19:09:05,045 - DEBUG - e2epipeline.fit - step 2 (svd): fit_args=list(2) fit_kwargs=dict0 = {}
2020-01-03 19:09:05,083 - DEBUG - e2epipeline.fit - step 2 (svd): dict2 = {0: np.array(466, 100), 1: pd.Series(466)}
2020-01-03 19:09:05,084 - DEBUG - e2epipeline.fit - step 3 (svc): dict2 = {0: np.array(466, 100), 1: pd.Series(466)}
2020-01-03 19:09:05,085 - DEBUG - e2epipeline.fit - step 3 (svc): fit_args=list(2) fit_kwargs=dict0 = {}
2020-01-03 19:09:05,206 - DEBUG - e2epipeline.fit - step 3 (svc): dict2 = {0: np.array(466,), 1: pd.Series(466)}
2020-01-03 19:09:05,206 - DEBUG - e2epipeline.fit - finished
2020-01-03 19:09:05,207 - DEBUG - e2epipeline.predict - started
2020-01-03 19:09:05,210 - DEBUG - e2epipeline.predict - step 0 (icf): dict1 = {0: pd.Series(200)}
2020-01-03 19:09:05,212 - DEBUG - e2epipeline.predict - step 0 (icf): dict1 = {0: pd.Series(200)}
2020-01-03 19:09:05,213 - DEBUG - e2epipeline.predict - step 1 (tfidf): dict1 = {0: pd.Series(200)}
2020-01-03 19:09:05,232 - DEBUG - e2epipeline.predict - step 1 (tfidf): dict1 = {0: scipy.sparse(200, 469)}
2020-01-03 19:09:05,232 - DEBUG - e2epipeline.predict - step 2 (svd): dict1 = {0: scipy.sparse(200, 469)}
2020-01-03 19:09:05,234 - DEBUG - e2epipeline.predict - step 2 (svd): dict1 = {0: np.array(200, 100)}
2020-01-03 19:09:05,235 - DEBUG - e2epipeline.predict - step 3 (svc): dict1 = {0: np.array(200, 100)}
2020-01-03 19:09:05,241 - DEBUG - e2epipeline.predict - step 3 (svc): dict1 = {0: np.array(200,)}
2020-01-03 19:09:05,242 - DEBUG - e2epipeline.predict - finished
```

For those who are curious what is the accuracy of the classifier there is following calculation.

```python
accuracy = np.mean(y_test == y_e2e)
print(f"accuracy={accuracy:.3f}")
```

`accuracy=0.365`
