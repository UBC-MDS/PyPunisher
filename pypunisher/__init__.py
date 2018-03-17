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

__version_info__ = (4, 0, 0)
__version__ = '.'.join(map(str, __version_info__))
