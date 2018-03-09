"""

    Utils
    ~~~~~

"""
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
            If true, this function collapses to `range()`.

    Returns:
        `trange` object.

    """
    return trange(n, desc='Worst Case', disable=not verbose)


def parse_n_features(param, total):
    """Parse either the `n_features` (backward)
    or `max_features` (forward) parameter. Namely
    (a) if `param` is an int, ensure it lies on (0, `total`),
    (a) if `param` is a float, ensure it lies on (0, 1).

    Args:
        param : int
            One of `n_features`, `max_features`.
        total : int
            The total features in the data

    Returns:
        int
            * if `n_features`, number of features to select.
            * if `max_features`, the largest number of features to select.

    """
    if isinstance(param, int) and not 0 < param < total:
        raise ValueError(
            "If an int, `n_features` must be on (0, {}).".format(
                total
            )
        )
    if isinstance(param, float) and not 0 < param < 1:
        raise ValueError(
            "If a float, `n_features` must be on (0, 1)."
        )

    if isinstance(param, float):
        return int(param * total)
    else:
        return param
