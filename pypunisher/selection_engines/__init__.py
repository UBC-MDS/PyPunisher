#!/usr/bin/env python

"""
Selection Engines
=================
Provides two types of feature selection:
1. Forward selection
2. Backward selection
"""

import warnings

# See: https://github.com/scipy/scipy/issues/5998.
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
