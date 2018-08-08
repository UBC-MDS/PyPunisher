#!/usr/bin/env python

"""
Wrapper Functions for Testing
=============================
"""
from copy import deepcopy
from tests._defaults import DEFAULT_SELECTION_PARAMS
from pypunisher.selection_engines.selection import Selection

def _sel(**kwargs):
    func_kwargs = dict()
    dsp = deepcopy(DEFAULT_SELECTION_PARAMS)
    for k, v in kwargs.items():
        if k in dsp:
            dsp[k] = v
        else:
            func_kwargs[k] = v
    sel = Selection(**dsp)
    return sel, func_kwargs

def forward(**kwargs):
    sel, func_kwargs = _sel(**kwargs)
    return sel.forward(**func_kwargs)


def backward(**kwargs):
    sel, func_kwargs = _sel(**kwargs)
    return sel.backward(**func_kwargs)
