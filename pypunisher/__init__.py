#!/usr/bin/env python

"""
PyPunisher
==========
PyPunisher is a package for feature and model selection in Python.
Specifically, PyPunisher implements tools for forward and backward model
selection. In order to measure model quality during the selection procedures,
PyPunisher provides two metrics: the Akaike and Bayesian Information
Criterion, both of which punish complex models.
"""

from pypunisher.metrics.criterion import aic, bic
from pypunisher.selection_engines.selection import Selection
from pypunisher._example_data import X_train, X_val, y_train, y_val, true_best_feature

__version_info__ = (3, 0, 0)
__version__ = '.'.join(map(str, __version_info__))
