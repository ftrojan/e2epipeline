# Standard Scikit-learn Pipeline

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

```text
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
