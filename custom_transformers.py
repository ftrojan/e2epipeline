"""Custom transformers to be used in a hierarchical classification pipeline"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, Tuple


ROOT = "<ROOT>"


def extract_graph_from_paths(paths: pd.Series) -> nx.DiGraph:
    """
    Extracts networkx.DiGraph representation of the hierarchy captured in the paths
    :param paths:
    :return: DiGraph
    """
    separator = '/'
    upath = paths.unique()
    nxgraph = nx.DiGraph()
    for path in upath:
        child = path
        idx_last = child.rfind(separator)
        while idx_last >= 0:
            if idx_last > 0:
                parent = path[:idx_last]
            else:
                parent = ROOT
            nxgraph.add_edge(parent, child)
            idx_last = parent.rfind(separator)
            child = parent
    return nxgraph


def pruning_df(tg: pd.Series, hierarchy: nx.DiGraph, size_threshold: int) -> pd.DataFrame:
    """
    Calculates pruning DataFrame
    :param tg:
    :param hierarchy:
    :param size_threshold: integer
    :return:
    """
    # initial dataframe
    node_stats = pd.DataFrame({
        'size_node': [sum(tg == x) for x in hierarchy.nodes()],
        'size_total': np.nan,
        'is_target': False,
        'target': None,
    }, index=hierarchy.nodes())
    # calculate total size of each node
    while node_stats['size_total'].isnull().sum() > 0:
        for row in node_stats.itertuples():
            children = list(hierarchy.successors(row.Index))
            child_sizes = node_stats.loc[children, 'size_total']
            node_stats['size_total'].at[row.Index] = row.size_node + np.sum(child_sizes.values)
    # decide if is_target
    for row in node_stats.itertuples():
        node_stats['is_target'].at[row.Index] = np.all(row.size_total >= size_threshold)
    # assign target to each node top down
    while node_stats['target'].isnull().sum() > 0:
        for row in node_stats.itertuples():
            if row.is_target and row.target is None:
                node_stats.at[row.Index, 'target'] = row.Index
            if row.target is not None:
                children = hierarchy.successors(row.Index)
                nodes_to_assign = [node for node in children if node_stats.at[node, 'target'] is None]
                for node_to_assign in nodes_to_assign:
                    node_stats.at[node_to_assign, 'target'] = row.target
    return node_stats


def recode_warn(column: pd.Series, recode_lookup: pd.Series) -> tuple:
    """Recode values in series using lookup with logging.warn if not found."""
    def lookup_warn(lookup, val):
        try:
            lkp = lookup.at[val]
        except KeyError:
            lkp = None
            counter = warnlist.get(val, 0)
            warnlist[val] = counter + 1
        return lkp
    warnlist = {}
    res = pd.Series([lookup_warn(recode_lookup, value) for value in column])
    if len(warnlist.keys()) > 0:
        msg = f"Unknown values recoded to None. See PruneHierarchicalTarget.warnlist property for details."
        logging.warning(msg)
    return res, warnlist


class PruneHierarchicalTarget(BaseEstimator, TransformerMixin):
    """
    Pruning of hierarchical target by size_threshold.

    Target must be coded as hierarchical path from left to right delimited with slash (/) with leading slash.
    """

    def __init__(
            self,
            size_threshold: int,
    ):
        self.size_threshold = size_threshold
        self.hierarchy = None
        self.prunedf = None
        self.warnlist = {}

    def fit(self, x, y):
        """
        Constructs DiGraph tree representing the hierarchy and calculates
        the lookup table for recoding original target into pruned target.

        :param x: predictors, not used
        :param y: series with hierarchical paths delimited with slash (/)
        :return: fitted object with the following attributes:
            hierarchy: nx.DiGraph the tree with nodes labeled as hierarchical paths
            prunedf: pd.DataFrame for recoding with index of hierarchical paths and the following columns:
                size_node: size of that node (number of observations found in x)
                size_total: total size of that node, size of node + size of all its direct and indirect descendants
                is_target: True iif size_total >= size_threshold
                target: the recoded path, the path of the closest parent with is_target == True
        """
        logging.debug('PruneHierarchicalTarget.fit started')
        self.hierarchy = extract_graph_from_paths(y)
        logging.debug('hierarchy extracted')
        self.prunedf = pruning_df(y, self.hierarchy, self.size_threshold)
        logging.debug('PruneHierarchicalTarget.fit ended')
        return self

    def transform(self, x, y=None):
        """
        Recodes the hierarchical target into pruned hierarchical target using the fitted recoding lookup.

        If y contains unknown paths, they are recoded to None strings.

        :param x: predictors, not used
        :param y: series with hierarchical paths delimited with slash (/)
        :return: series with x recoded into pruned paths
        """
        logging.debug('PruneHierarchicalTarget.transform started')
        if y is None:
            res = (x, )
        else:
            leafs_pruned, self.warnlist = recode_warn(y, self.prunedf['target'])
            y_transformed = pd.Series(leafs_pruned)
            y_transformed.index = y.index
            res = x, y_transformed
        logging.debug('PruneHierarchicalTarget.transform ended')
        return res

    def fit_transform(self, x, y):
        """Calls fit and transform on the given X, y data, resulting X and y are returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data to be transformed. Instances belonging ot infrequent classes will be removed.
        y : array-like, shape (n_samples,)
            Target variable used to count classes and also check which instances are to be removed.
            This target is also transformed.

        Returns
        -------
        X : {array-like, sparse matrix}
            Transformed X.
        y : array-like
            Transformed y.
        """
        return self.fit(x, y).transform(x, y)


class InfrequentClassFilter(BaseEstimator):
    """
    Transformer for removing instances of classes with number of occurrences < threshold.
    It transform both X and y.

    Parameters
    ----------
    size_threshold : int
        Size threshold used for removing instances.
    """

    def __init__(self, size_threshold: int):
        self.size_threshold = size_threshold
        self.classes_to_keep_ = None

    def fit(self, X, y: Union[pd.Series, np.ndarray]) -> 'InfrequentClassFilter':
        """Computes the class counts used later for filtering instances.

        Parameters
        ----------
        X
            Ignored.
        y : array-like, shape (n_samples,)
            Target vector used for counting the occurrences of the classes.

        Returns
        -------
        self : object
        """
        unique, counts = np.unique((y.values if isinstance(y, pd.Series) else y), return_counts=True)
        self.classes_to_keep_ = unique[counts >= self.size_threshold]
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series, np.ndarray], y=None) \
            -> Tuple[pd.DataFrame, pd.Series]:
        """Performs removal of instances belonging to classes whose size < threshold.
        Return modified X and y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data to be transformed. Instances belonging ot infrequent classes will be removed.
        y : array-like, shape (n_samples,)
            Target variable used to check which instances are to be removed. This target is also transformed.

        Returns
        -------
        X : {array-like, sparse matrix}
            Transformed X.
        y : array-like
            Transformed y.
        """
        if y is None:
            return (X, )
        else:
            msg = f"Transformer {type(self).__name__} was not fitted yet. Call 'fit' with " \
                "appropriate arguments before using this method."
            assert self.classes_to_keep_ is not None, msg
            if X is not None and X.shape[0] != y.shape[0]:
                raise ValueError("X and y have different number of instances in %(name)s's transform method." %
                                 {'name': type(self).__name__})
            # boolean_index = y.isin(self.classes_to_keep).values
            boolean_index = np.isin((y.values if isinstance(y, pd.Series) else y), self.classes_to_keep_)
            if X is not None:
                # return X.loc[boolean_index], y.loc[boolean_index]
                return X[boolean_index], y[boolean_index]
            else:
                return None, y[boolean_index]

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) \
            -> Tuple[pd.DataFrame, pd.Series]:
        """Calls fit and transform on the given X, y data, resulting X and y are returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data to be transformed. Instances belonging ot infrequent classes will be removed.
        y : array-like, shape (n_samples,)
            Target variable used to count classes and also check which instances are to be removed.
            This target is also transformed.

        Returns
        -------
        X : {array-like, sparse matrix}
            Transformed X.
        y : array-like
            Transformed y.
        """
        return self.fit(X, y).transform(X, y)


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
