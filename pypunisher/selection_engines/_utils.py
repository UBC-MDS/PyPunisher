"""

    Utils
    ~~~~~

"""
from tqdm import trange


def get_n_features(matrix, min_=2):
    _, n_features = matrix.shape
    if n_features < min_:
        raise IndexError(
            "less than {} features present.".format(min_)
        )
    return n_features


def enforce_use_of_all_cpus(model):
    if hasattr(model, 'n_jobs'):
        setattr(model, 'n_jobs', -1)
    return model


def worse_case_bar(n, verbose):
    return trange(n, desc='Worst Case', disable=not verbose)
