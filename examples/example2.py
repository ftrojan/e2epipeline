"""
Example2 is demo of E2EPipeline which adds two custom transformers with xy binding as the first two steps.

Those two custom transformers are PruneHierarchicalTarget and InfrequentClassFilter.
"""

import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

from e2epipeline import E2EPipeline
from custom_transformers import PruneHierarchicalTarget, InfrequentClassFilter

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logging.debug("run pipeline start")
df = pd.read_csv('data/all_tickets_1000.csv')
pipeline = E2EPipeline([
    ('pruner', PruneHierarchicalTarget(
        size_threshold=50
    )),
    ('icf', InfrequentClassFilter(
        size_threshold=100
    )),
    ('tfidf', TfidfVectorizer(
        sublinear_tf=True,
        min_df=5,
        norm='l2',
        encoding='latin-1',
        ngram_range=(1, 2),
        stop_words='english',
    )),
    ('svd', TruncatedSVD(
        n_components=100
    )),
    ('svc', CalibratedClassifierCV(
        base_estimator=svm.LinearSVC(),
        cv=5
    )),
])
x = df['body']
y = df['target_path']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
