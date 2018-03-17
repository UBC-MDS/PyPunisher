#!/usr/bin/env python

"""
Checks
======
"""
import numpy as np


def model_check(model):
    """Check that `model` contains `fit`, `predict` and `score` methods.

    Args:
        model : sklearn model
            Any sklearn model

    """
    for attr in ('fit', 'predict', 'score'):
        if not hasattr(model, attr):
            raise AttributeError(
                "`model` does not contain a {} method".format(attr)
            )


def array_check(self):
    """Check that the arrays in `self` are
    truly ndarrays.

    Args:
        self : class object

    """
    for i in ("_X_train", '_y_train', '_X_val', '_y_val'):
        if not isinstance(getattr(self, i), np.ndarray):
            raise TypeError("{} must be a ndarray.".format(i[1:]))


def input_checks(locals_):
    """Check that
    (a) the only one of the the two inputs are non-None
    (b) that the remaining element is numeric and is strictly
    greater than zero.

    Args:
        locals_ : dict
            Yield of `locals()` within a function.

    """
    # Sort so that the order of the parameter name
    # are in a reliable (alphabetical) order.
    locals_.pop('self')
    param_a, param_b = sorted(k for k, p in locals_.items())
    locals_non_none = {k: v for k, v in locals_.items()
                       if v is not None}

    if len(locals_non_none) != 1:
        raise TypeError(
            "At least one of `{a}` and `{b}` must be None.".format(
                a=param_a, b=param_b
            )
        )

    # Unpack the single key and value pair.
    name, obj = tuple(locals_non_none.items())[0]
    if obj is None and not isinstance(obj, (int, float)):
        raise TypeError(
            "`{}` must be of type int or float.".format(name)
        )
    elif not obj > 0:
        raise ValueError(
            "`{}` must be greater than zero.".format(name)
        )
