"""
Keep different states of pipeline using reposition and position.

If we add preprocessing and postprocessing to pipeline steps, we can play with state and capture specific
inputs and outputs as separate elements of the state. In this example the final state elements are:
- 0: initial dataframe with two columns (subject and body)
- 1: target with infrequent classes filtered out after pruning
- 3: SVD of TF-IDF of email body truncated to 100 most significant components
- 4: subj and body merged together using MergeSubjectBody transformer, the input to SVC classifier
- 5: predictions on the training dataset
- 6: sparse matrix of TF-IDF from subject column

The element with index 2 is not present, because it would enter the SVC classifier as
the parameter sample_weights, which is not desired.
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

from src.e2epipeline import E2EPipeline, Step, position, reposition, object_info
from src.custom_transformers import PruneHierarchicalTarget, InfrequentClassFilter


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
    ), postprocessing=position([4, 1])),
    Step('icf', InfrequentClassFilter(
        size_threshold=100
    ), postprocessing=position([4, 1])),
    Step('tfsubj', TfidfVectorizer(
        sublinear_tf=True,
        min_df=5,
        max_features=10,
        norm='l2',
        ngram_range=(1, 2),
        stop_words='english',
    ), preprocessing=lambda s: {'raw_documents': s[4]['title'], 'y': s.get(1, None)},
         postprocessing=position(6)),
    Step('tfbody', TfidfVectorizer(
        sublinear_tf=True,
        min_df=5,
        norm='l2',
        ngram_range=(1, 2),
        stop_words='english',
    ), preprocessing=lambda s: {'raw_documents': s[4]['body'], 'y': s.get(1, None)},
         postprocessing=position(3)),
    Step('svd', TruncatedSVD(
        n_components=100
    ), preprocessing=reposition({3: 0}), postprocessing=position(3)),
    Step('merge', MergeSubjectBody(),
         preprocessing=reposition({6: 0, 3: 1}),
         postprocessing=position(4)),
    Step('svc', CalibratedClassifierCV(
        base_estimator=svm.LinearSVC(),
        cv=5
    ), preprocessing=reposition({4: 0, 1: 1}), postprocessing=position(5)),
])
x = df[['title', 'body']]
y = df['target_path']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
pipeline.fit(x_train, y_train)
logging.info(f"pipeline.state={object_info(pipeline.state)}")
y_pred = pipeline.predict(x_test)
logging.info(f"y_pred={object_info(y_pred)}")
logging.debug(f"features={pipeline.named_steps['merge'].features_}")
logging.debug("completed")
