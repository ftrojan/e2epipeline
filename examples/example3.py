"""
Example3 is demo of E2EPipeline with transformers nonstandard from the sklearn perspective.
"""

import logging
import numpy as np
from sklearn.base import BaseEstimator

from e2epipeline import E2EPipeline


class TransformerX(BaseEstimator):

    def fit(self, X, y=None):
        self.num = len(X)
        return self

    def transform(self, X, copy=None):
        return X + self.num

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


class TransformerY(BaseEstimator):

    def fit(self, y):
        self.num = y[0]
        return self

    def transform(self, y, copy=None):
        return y + (self.num * 10)

    def fit_transform(self, y, copy=None):
        return self.fit(y).transform(y)


class TransformerXY(BaseEstimator):

    def fit(self, X, y):
        self.min_x = X.min()
        self.min_y = y.min()
        return self

    def transform(self, X, y):
        return X + self.min_x, y + self.min_y

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logging.debug("start")
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
logging.debug(new_x)
logging.debug(new_y)
logging.debug("completed")
