import warnings

# See: https://github.com/scipy/scipy/issues/5998.
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
