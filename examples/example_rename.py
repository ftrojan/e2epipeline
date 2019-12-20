"""
Keep different states of pipeline using rename and name.

If we add preprocessing and postprocessing to pipeline steps, we can play with state and capture specific
inputs and outputs as separate elements of the state. In this example the final state elements are:
- x: initial dataframe with two columns (subject and body)
- y: target with infrequent classes filtered out after pruning
- subj: sparse matrix of TF-IDF from subject column
- body: SVD of TF-IDF of email body truncated to 100 most significant components
- X: subj and body merged together using MergeSubjectBody transformer, the input to SVC classifier
- pred: predictions on the training dataset
"""
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator

from e2epipeline import E2EPipeline, Step, name, rename, object_info
from custom_transformers import PruneHierarchicalTarget, InfrequentClassFilter


class MergeSubjectBody(BaseEstimator):

    def __init__(self):
        self.features_ = None

    def fit(self, subject, body):
        features_subj = [f"subject_{i+1:04d}" for i in range(subject.shape[1])]
        features_body = [f"body_{i+1:04d}" for i in range(body.shape[1])]
        self.features_ = features_subj + features_body
        return self

    def transform(self, subject, body):
        merged = np.concatenate([subject.todense(), body], axis=1)
        assert merged.shape[1] == len(self.features_)
        return merged


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s#%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
df = pd.read_csv('data/all_tickets_1000.csv')
pipeline = E2EPipeline([
    Step('pruner', PruneHierarchicalTarget(
        size_threshold=50
    ), postprocessing=name(['X', 'y'])),
    Step('icf', InfrequentClassFilter(
        size_threshold=100
    ), postprocessing=name(['X', 'y'])),
    Step('tfsubj', TfidfVectorizer(
        sublinear_tf=True,
        min_df=5,
        max_features=10,
        norm='l2',
        ngram_range=(1, 2),
        stop_words='english',
    ), preprocessing=lambda s: {'raw_documents': s['X']['title'], 'y': s.get('y', None)},
         postprocessing=name('title')),
    Step('tfbody', TfidfVectorizer(
        sublinear_tf=True,
        min_df=5,
        norm='l2',
        ngram_range=(1, 2),
        stop_words='english',
    ), preprocessing=lambda s: {'raw_documents': s['X']['body'], 'y': s.get('y', None)},
         postprocessing=name('body')),
    Step('svd', TruncatedSVD(
        n_components=100
    ), preprocessing=rename({'body': 'X'}), postprocessing=name('body')),
    Step('merge', MergeSubjectBody(),
         preprocessing=rename({'title': 'subject', 'body': 'body'}),
         postprocessing=name('X')),
    Step('svc', CalibratedClassifierCV(
        base_estimator=svm.LinearSVC(),
        cv=5
    ), postprocessing=name('pred')),
])
x = df[['title', 'body']]
y = df['target_path']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
pipeline.fit(x=x_train, y=y_train)
logging.info(f"pipeline.state={object_info(pipeline.state)}")
y_pred = pipeline.predict(x=x_test)
logging.info(f"y_pred={object_info(y_pred)}")
logging.debug(f"features={pipeline.named_steps['merge'].features_}")
logging.debug("completed")
