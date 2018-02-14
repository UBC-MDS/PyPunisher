"""

    Feature Selection Algorithms
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""


class Selection(object):
    """Feature Selection for Sklearn Models.

    Args:
        model (sklearn model): any sklearn model with `.fit()`, `.predict()` and `.score()` methods.
        X_train (ndarray): a 2D numpy array of (observations, features).
        y_train (ndarray): a 1D array of target classes for X_train.
        X_val (ndarray): a 2D numpy array of (observations, features).
        y_val (ndarray): a 1D array of target classes for X_validate.
        verbose (bool): if True, print additional information as selection occurs.
                        Defaults to True.
    """

    def __init__(self, model, X_train, y_train, X_val, y_val, verbose=True):
        self._model = model
        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val
        self._verbose = verbose

    def _selection_engine(self, *args, **kwargs):
        """A backend to power the `.forward()` and `backward()` selection
        procedures (i.e., the parts that are common to the two).

        Args:
            TBD

        Returns:
            TBD
        """
        raise NotImplementedError

    def forward(self, epsilon=10e-4, epsilon_history=5,
                max_features=None, criterion=None):
        """Perform forward selection on a Sklearn model.

        Args:
            epsilon (float): smallest average change in the model's score to be considered meaningful
                             over the past `epsilon_history` iterations of the algorithm.
            epsilon_history (int): number of past values to consider when comparing `epsilon`.
            max_features (int): the max. number of features to allow.
            criterion (str or None): model selection criterion.
                                    * 'aic': use Akaike Information Criterion.
                                    * 'bic': use Bayesian Information Criterion.
                                    * None: Use the default score built into the model
                                      object (i.e., call `.score()`).

        Returns:
            S (ndarray): column indices of `X_train` (and `X_val`) that denote the chosen features.

        """
        self._selection_engine()

    def backward(self, epsilon=10e-4, epsilon_history=5,
                 max_features=None, criterion=None):
        """Perform backward selection on a Sklearn model.

        Args:
            epsilon (float): smallest average change in the model's score to be considered meaningful
                             over the past `epsilon_history` iterations of the algorithm.
            epsilon_history (int): number of past values to consider when comparing `epsilon`.
            max_features (int): the max. number of features to allow.
            criterion (str or None): model selection criterion.
                                    * 'aic': use Akaike Information Criterion.
                                    * 'bic': use Bayesian Information Criterion.
                                    * None: Use the default score built into the model
                                      object (i.e., call `.score()`).

        Returns:
            S (ndarray): column indices of `X_train` (and `X_val`) that denote the chosen features.

        """
        self._selection_engine()
