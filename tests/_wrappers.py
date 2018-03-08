"""

    Wrapper Functions for Testing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from tests._defaults import DEFAULT_SELECTION_PARAMS
from pypunisher.selection_engines.forward import ForwardSelection
from pypunisher.selection_engines.backward import BackwardSelection


def forward(**kwargs):
    fsel = ForwardSelection(**DEFAULT_SELECTION_PARAMS)
    return fsel.forward(**kwargs)


def backward(**kwargs):
    bsel = BackwardSelection(**DEFAULT_SELECTION_PARAMS)
    return bsel.backward(**kwargs)
