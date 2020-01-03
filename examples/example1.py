"""
Example1 is demo of E2EPipeline which mimics a standard sklearn pipeline with no custom transformers or estimators.
PruneHierarchicalTarget is performed before the pipeline.
"""

import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

from src.e2epipeline import E2EPipeline
from src.custom_transformers import PruneHierarchicalTarget, InfrequentClassFilter

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logging.debug("start")
df = pd.read_csv('data/all_tickets_1000.csv')
pipeline = E2EPipeline([
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
        base_estimator=svm.LinearSVC(random_state=33),
        cv=5
    )),
])
x = df['body']
y = df['target_path']
pruner = PruneHierarchicalTarget(size_threshold=50)
pruner.fit(x, y)
_, y_pruned = pruner.transform(x, y)
icf = InfrequentClassFilter(size_threshold=10)
icf.fit(x, y_pruned)
x_filt, y_filt = icf.transform(x, y_pruned)
x_train, x_test, y_train, y_test = train_test_split(x_filt, y_filt, test_size=0.2, random_state=42)
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
logging.debug("completed")
