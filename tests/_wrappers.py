"""

    Wrapper Functions for Testing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from copy import deepcopy
from tests._defaults import DEFAULT_SELECTION_PARAMS
from pypunisher.selection_engines.selection import Selection


def _sel(**kwargs):
    dsp = deepcopy(DEFAULT_SELECTION_PARAMS)
    for k, v in kwargs.items():
        if k in dsp:
            dsp[k] = v
    sel = Selection(**dsp)
    # Extra loop, but avoids popping values during iteration above
    new_kwargs = {k: v for k, v in kwargs.items() if k not in dsp}
    return sel, new_kwargs


def forward(**kwargs):
    sel, new_kwargs = _sel(**kwargs)
    return sel.forward(**new_kwargs)


def backward(**kwargs):
    sel, new_kwargs = _sel(**kwargs)
    return sel.backward(**new_kwargs)
