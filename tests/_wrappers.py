"""

    Wrapper Functions for Testing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from tests._defaults import DEFAULT_SELECTION_PARAMS
from pypunisher.selection_engines.selection import Selection


def forward(**kwargs):
    fsel = Selection(**DEFAULT_SELECTION_PARAMS)
    return fsel.forward(**kwargs)


def backward(**kwargs):
    bsel = Selection(**DEFAULT_SELECTION_PARAMS)
    return bsel.backward(**kwargs)
