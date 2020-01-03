import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

from src.e2epipeline import E2EPipeline, Step, name, rename, position, reposition
from src.custom_transformers import (
    PruneHierarchicalTarget,
    InfrequentClassFilter,
    MergeSubjectBody,
    TransformerX,
    TransformerXY,
    TransformerY,
)


df = pd.read_csv('data/all_tickets_1000.csv')


def test_example1():
    """
    Example1 is demo of E2EPipeline which mimics a standard sklearn pipeline with no custom transformers or estimators.
    PruneHierarchicalTarget and InfrequentClassFilter is performed before the pipeline.
    """
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
            n_components=50
        )),
        ('svc', CalibratedClassifierCV(
            base_estimator=svm.LinearSVC(),
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
    assert y_pred.shape == y_test.shape


def test_example2():
    """
    Example2 is demo of E2EPipeline which adds two custom transformers with xy binding as the first two steps.

    Those two custom transformers are PruneHierarchicalTarget and InfrequentClassFilter.
    """
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
            n_components=50
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
    assert y_pred.shape == y_test.shape


def test_example3():
    """
    Example3 is demo of E2EPipeline with transformers nonstandard from the sklearn perspective.
    """
    X = np.array([
        [4, 2, 3],
        [3, 5, 7],
        [5, 8, 4]
    ])
    y = np.array([8, 4, 7])
    pipeline = E2EPipeline([
        ('only_x', TransformerX()),
        ('only_y', TransformerY()),
        ('both_xy', TransformerXY()),
    ])
    pipeline.fit(X=X, y=y)
    new_x, new_y = pipeline.predict(X=X, y=y)
    expected_new_x = np.array([
        [6,  4,  5],
        [5,  7,  9],
        [7, 10,  6],
    ])
    expected_new_y = np.array([12,  8, 11])
    assert np.array_equal(new_x, expected_new_x)
    assert np.array_equal(new_y, expected_new_y)


def test_preprocessing_rename():
    """Keep different states of pipeline using rename and name."""
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
        ), postprocessing=None),
    ])
    x = df[['title', 'body']]
    y = df['target_path']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pipeline.fit(x=x_train, y=y_train)
    y_pred = pipeline.predict(x=x_test)
    assert y_pred.shape == y_test.shape


def test_preprocessing_reposition():
    """Keep different states of pipeline using reposition and position."""
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
        ), preprocessing=reposition({3: 0})
             , postprocessing=position(3)),
        Step('merge', MergeSubjectBody(),
             preprocessing=reposition({6: 0, 3: 1}),
             postprocessing=position(4)),
        Step('svc', CalibratedClassifierCV(
            base_estimator=svm.LinearSVC(),
            cv=5
        ), preprocessing=reposition({4: 0}),
             postprocessing=None),
    ])
    x = df[['title', 'body']]
    y = df['target_path']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    assert y_pred.shape == y_test.shape
