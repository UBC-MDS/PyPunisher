"""

    Forward Selection Algorithm
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from pypunisher.selection_engines._utils import (get_n_features,
                                                 enforce_use_of_all_cpus,
                                                 worse_case_bar, array_check,
                                                 model_check,
                                                 parse_features_param,
                                                 input_checks)


class ForwardSelection(object):
    """Forward Selection Algorithm.

    Args:
        model : sklearn model
            any sklearn model with `.fit()`, `.predict()` and
            `.score()` methods.
        X_train ndarray:
            a 2D numpy array of (observations, features).
        y_train : ndarray:
            a 1D array of target classes for X_train.
        X_val : ndarray
            a 2D numpy array of (observations, features).
        y_val : ndarray
            a 1D array of target classes for X_validate.
        criterion : str or None
            model selection criterion.
            * 'aic': use Akaike Information Criterion.
            * 'bic': use Bayesian Information Criterion.
            * None: Use the default score built into the model
              object (i.e., call `.score()`).
        verbose : bool
            if True, print additional information as selection occurs.
            Defaults to True.
    """

    def __init__(self, model, X_train, y_train,
                 X_val, y_val, criterion=None, verbose=True):
        model_check(model)
        self._model = enforce_use_of_all_cpus(model)

        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val
        array_check(self)

        self._verbose = verbose
        self._criterion = criterion  # ToDO: Not Implemented Yet.
        self._n_features = get_n_features(X_train)

    def _fit_and_score_forward(self, S, include):
        features = S + [include]
        self._model.fit(self._X_train[:, features], self._y_train)
        return self._model.score(self._X_val[:, features], self._y_val)

    def _forward_break_criteria(self, S, j_score_dict, max_features):
        # a. Check if the algorithm should halt b/c of features themselves
        if not len(j_score_dict) or len(S) == self._n_features:
            return True
        # b. Break if the number of features in S > max_features.
        elif isinstance(max_features, int) and max_features > len(S):
            return True
        else:
            return False

    def forward(self, min_change=0.5, max_features=None):
        """Perform forward selection on a Sklearn model.

        Args:
            min_change : int or float, optional
                The smallest change to be considered significant.
                Note: `max_features` must be None in order for `min_change` to operate.
            max_features : int
                the max. number of features to allow.
                Note: `min_change` must be None in order for `max_features` to operate.
                Floats will be regarded as proportions of the total
                that must lie on (0, 1)

        Returns:
            S : list
              The column indices of `X_train` (and `X_val`) that denote the chosen features.
        """
        input_checks(locals())
        S = list()
        best_score = None
        itera = list(range(self._n_features))

        if max_features:
            n_features = parse_features_param(
                max_features, total=len(itera), param_name="max_features"
            )

        worse_case = worse_case_bar(self._n_features, verbose=self._verbose)
        for _ in worse_case:
            worse_case.set_postfix(n_features=len(S), score=best_score)
            # 1. Find best feature, j, to add.
            j_score_dict = dict()
            for j in itera:
                j_model_score = self._fit_and_score_forward(S, include=j)
                if best_score is None or (j_model_score > best_score):
                    j_score_dict[j] = j_model_score

            # 2. Save the best j to S, if possible.
            if len(j_score_dict):
                best_j = max(j_score_dict, key=j_score_dict.get)
                best_j_score = j_score_dict[best_j]
                # Update S, the best score and score history ---
                change = best_j_score - best_score if best_score else 0
                best_score = best_j_score  # update the score to beat
                S.append(best_j)  # add feature
                itera.remove(best_j)  # no longer search over this feature.

            if self._forward_break_criteria(S, j_score_dict=j_score_dict,
                                            max_features=max_features):
                break

        return S
