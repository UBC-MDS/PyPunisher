"""

    Utils
    ~~~~~

"""

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


def parse_n_features(n_features, total):
    """Parse either the `n_features` for forward
    and backward selection. Namely
    (a) if `param` is an int, ensure it lies on (0, `total`),
    (a) if `param` is a float, ensure it lies on (0, 1).

    Args:
        n_features : int
            An `n_features` parameter passed to forward or backward selection.
        total : int
            The total features in the data

    Returns:
        int
            * number of features to select.
            If `n_features` and it lies on (0, `total`),
            it will be returned 'as is'.

    """
    if isinstance(n_features, int) and not 0 < n_features < total:
        raise ValueError(
            "If an int, `n_features` must be on (0, {}).".format(
                total
            )
        )
    if isinstance(n_features, float) and not 0 < n_features < 1:
        raise ValueError(
            "If a float, `n_features` must be on (0, 1)."
        )

    if isinstance(n_features, float):
        return int(n_features * total)
    else:
        return n_features
