"""

    Utils
    ~~~~~

"""
import numpy as np
from tqdm import trange


def get_n_features(matrix, min_=2):
    """Get the number of features in a matrix.

    Args:
        matrix : ndarray
            A numeric numpy array. Features are
            assumed to be the columns.
        min_ : int
            The smallest number of features that
            `matrix` is permitted to have.

    Returns
        n_features : int
            The number of features in `matrix`.

    Raises:
        * if `matrix.shape[1] < min_`

    """
    _, n_features = matrix.shape
    if n_features < min_:
        raise IndexError(
            "less than {} features present.".format(min_)
        )
    return n_features


def enforce_use_of_all_cpus(model):
    """For sklearn models which have an `n_jobs` attribute,
    set to -1. This will force all cores on the machine to be
    used.

    Args:
        model : sklearn model
            A trainable sklearn model

    Returns:
        model : sklearn model
            Model 'as is' with `n_jobs` set to one if it
            exists

    """
    if hasattr(model, 'n_jobs'):
        setattr(model, 'n_jobs', -1)
    return model


def worse_case_bar(n, verbose):
    """Generate a progress bar for the worst case of
    a forward or backward selection.

    Args:
        n : int
            Number of iterations
        verbose : bool
            If true, the functionality of this function
            collapses to `range()`.

    Returns:
        `trange` object.

    """
    return trange(n, desc='Worst Case', disable=not verbose)


def array_check(self):
    """Check that the arrays in `self` are
    truly ndarrays.

    Args:
        self : class object

    """
    for i in ("_X_train", '_y_train', '_X_val', '_y_val'):
        if not isinstance(getattr(self, i), np.ndarray):
            raise TypeError("{} must be a ndarray.".format(i[1:]))


def model_check(model):
    """Check that `model` contains `fit`, `predict` and `score` methods.

    Args:
        model : sklearn model
            Any sklearn model

    """
    for attr in ('fit', 'predict', 'score'):
        if not hasattr(model, attr):
            raise AttributeError(
                "`model` does not contain an {} method".format(attr)
            )


def parse_features_param(param, total, param_name):
    if isinstance(param, int) and not 0 < param < total:
        raise ValueError(
            "If an int, `{}` must be on (0, {}).".format(param_name, total)
        )
    if isinstance(param, float) and not 0 < param < 1:
        raise ValueError(
            "If a float, `{}` must be on (0, 1).".format(param_name)
        )

    if isinstance(param, float):
        return int(param * len(total))
    else:
        return param


def input_checks(locals_):
    param_a, param_b = [p for k, p in locals_.items() if k != 'self']
    locals_non_non = {k: v for k, v in locals_.items()
                      if v is not None and k != 'self'}

    if len(locals_non_non) != 1:
        raise TypeError(
            "At least one of `{a}` and `{b}` must be None.".format(
                a=param_a, b=param_b
            )
        )

    # Unpack the single key and value pair
    name, obj = tuple(locals_non_non.items())[0]
    if obj is None and not isinstance(obj, (int, float)):
        raise TypeError(
            "`{}` must be of type int or float.".format(name)
        )
    elif not obj > 0:
        raise ValueError(
            "`{}` must be greater than zero.".format(name)
        )
