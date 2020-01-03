"""
E2EPipeline is a generalisation of sklearn Pipeline to allow for more flexible mapping of input and output parameters.

Sequentially apply a list of transforms and a final estimator. Intermediate steps of the pipeline must be ‘transforms’,
that is, they must implement fit and transform methods. The final estimator only needs to implement fit.
"""

import numpy as np
import pandas as pd
import scipy.sparse
import logging
import inspect
from typing import Optional, Union, Callable, Any, List


def object_info(obj) -> str:
    """Provide brief textual representation of an object of selected types. Useful for debugging."""
    if isinstance(obj, tuple):
        res = f"tuple{len(obj)} = ({', '.join([object_info(xi) for xi in obj])})"
    elif isinstance(obj, dict):
        res = f"dict{len(obj)} = {{{', '.join([str(k) + ': ' + object_info(v) for k, v in obj.items()])}}}"
    elif isinstance(obj, list):
        res = f"list({len(obj)})"
    elif isinstance(obj, pd.DataFrame):
        res = f"pd.DataFrame{obj.shape}"
    elif isinstance(obj, pd.Series):
        res = f"pd.Series({len(obj)})"
    elif isinstance(obj, np.ndarray):
        res = f"np.array{obj.shape}"
    elif isinstance(obj, scipy.sparse.csr.csr_matrix):
        res = f"scipy.sparse{obj.shape}"
    else:
        res = str(type(obj))
    return res


def state_from_scratch(args: tuple, kwargs: dict) -> dict:
    """Create pipeline state from (args, kwargs)."""
    state = {i: v for i, v in enumerate(args)}
    state.update(kwargs)
    return state


def state_reader(state: dict, consuming_callable: Callable) -> tuple:
    """
    Calculate read definition state vector elements given a consuming signature
    :param state:
    :param consuming_callable:
    :return: tuple of args, kwargs to be used for the subsequent call
    """
    signature = inspect.signature(consuming_callable)
    args = []
    kwargs = {}
    for i, (parameter_name, parameter) in enumerate(signature.parameters.items()):
        if parameter_name in state.keys():
            kwargs[parameter_name] = state[parameter_name]
        elif i in state.keys() and is_compatible(parameter, type(state[i])):
            args.append(state[i])
    return args, kwargs


def state_writer(state: dict, output: Any) -> dict:
    """
    Update state with the given output
    :param state: state to be updated
    :param output:
    :return: new state
    """
    result = state
    if isinstance(output, tuple):
        if len(output) == 2 and isinstance(output[0], tuple) and isinstance(output[1], dict):  # args, kwargs
            result = state_from_scratch(output[0], output[1])
        else:
            for i, v in enumerate(output):
                result[i] = v
    elif isinstance(output, dict):
        for k, v in output.items():
            result[k] = v
    else:
        result[0] = output
    return result


def infer_parameter_type(parameter: inspect.Parameter):
    """Infer callable parameter type using also the default value."""
    parameter_type = parameter.annotation
    if parameter_type == inspect.Parameter.empty:
        if parameter.default is not None and not parameter.default == inspect.Parameter.empty:
            parameter_type = type(parameter.default)
    return parameter_type


def is_compatible(parameter: inspect.Parameter, data_type):
    """Check compatibility of a parameter and a data type."""
    parameter_type = infer_parameter_type(parameter)
    if parameter_type == inspect.Parameter.empty:
        compatible = True
    elif parameter_type == data_type:
        compatible = True
    else:
        # https://stackoverflow.com/questions/45957615/check-a-variable-against-union-type-at-runtime-in-python-3-6
        if hasattr(parameter_type, '__origin__') and parameter_type.__origin__ is Union:
            compatible = data_type in parameter_type.__args__
        else:
            compatible = False
    return compatible


def remap_input(mapping: dict) -> Callable:
    """Pick inputs from state."""
    def remap_state(state: dict) -> dict:
        output = {mapping.get(k, k): v for k, v in state.items()}
        return output
    return remap_state


def rename(mapping: dict) -> Callable:
    """Pick inputs from state by name."""
    return remap_input(mapping)


def reposition(mapping: dict) -> Callable:
    """Pick inputs from state by position."""
    return remap_input(mapping)


def remap_output(slots: Union[List[str], List[int], str, int]) -> Callable:
    """Remap output to state."""
    def name_tuple(output: tuple) -> dict:
        if isinstance(output, tuple):
            result = {n: v for n, v in zip(slots, output)}
        elif isinstance(slots, str) or isinstance(slots, int):
            result = {slots: output}
        else:
            result = output
        return result
    return name_tuple


def name(names: Union[List[str], str]) -> Callable:
    """Name output and store in the state."""
    return remap_output(names)


def position(positions: Union[List[int], int]) -> Callable:
    """Place output in the state to a predefined positions."""
    return remap_output(positions)


class Step:

    def __init__(
            self,
            name: str,
            transformer: object,
            preprocessing: Optional[Callable] = None,
            postprocessing: Optional[Callable] = None
    ):
        assert hasattr(transformer, 'fit')
        if hasattr(transformer, 'predict'):
            self.transform_method = transformer.predict
        elif hasattr(transformer, 'transform'):
            self.transform_method = transformer.transform
        else:
            raise AttributeError(f"The transformer does not implement predict nor transform method.")
        self.name = name
        self.transformer = transformer
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing


class E2EPipeline:

    def __init__(self, steps):
        self.steps = []
        for i, step_defn in enumerate(steps):
            if isinstance(step_defn, tuple):
                assert isinstance(step_defn[0], str)
                step = Step(name=step_defn[0], transformer=step_defn[1])
            elif isinstance(step_defn, Step):
                step = step_defn
            else:  # step is sklearn estimator or transformer
                step = Step(name=str(type(step_defn)), transformer=step_defn)
            self.steps.append(step)
        self.named_steps = {step.name: step.transformer for step in self.steps}
        self.state = {}

    def fit(self, *args, **kwargs):
        logging.debug(f"started, args={object_info(args)}, kwargs={object_info(kwargs)}")
        self.state = state_from_scratch(args, kwargs)
        for i, step in enumerate(self.steps):
            logging.debug(f"step {i} ({step.name}): {object_info(self.state)}")
            if step.preprocessing is None:
                state = self.state
            else:
                state = step.preprocessing(self.state)
            fit_args, fit_kwargs = state_reader(state, step.transformer.fit)
            logging.debug(f"step {i} ({step.name}): fit_args={object_info(fit_args)} fit_kwargs={object_info(fit_kwargs)}")
            step.transformer.fit(*fit_args, **fit_kwargs)
            prt_args, prt_kwargs = state_reader(state, step.transform_method)
            output = step.transform_method(*prt_args, **prt_kwargs)
            if step.postprocessing is not None:
                output = step.postprocessing(output)
            self.state = state_writer(self.state, output)
            logging.debug(f"step {i} ({step.name}): {object_info(self.state)}")
        logging.debug("finished")
        return self

    def predict(self, *args, **kwargs) -> Any:
        logging.debug("started")
        self.state = state_from_scratch(args, kwargs)
        output = None
        for i, step in enumerate(self.steps):
            logging.debug(f"step {i} ({step.name}): {object_info(self.state)}")
            if step.preprocessing is None:
                state = self.state
            else:
                state = step.preprocessing(self.state)
            prt_args, prt_kwargs = state_reader(state, step.transform_method)
            output = step.transform_method(*prt_args, **prt_kwargs)
            if step.postprocessing is not None:
                output = step.postprocessing(output)
            self.state = state_writer(self.state, output)
            logging.debug(f"step {i} ({step.name}): {object_info(self.state)}")
        logging.debug("finished")
        return output
