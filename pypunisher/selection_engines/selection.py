"""

    Forward and Backward Selection Algorithms
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from pypunisher.selection_engines._checks import (model_check,
                                                  array_check,
                                                  input_checks)
from pypunisher.selection_engines._utils import (get_n_features,
                                                 enforce_use_of_all_cpus,
                                                 worse_case_bar,
                                                 parse_features_param)


class Selection(object):
    """Unified Forward and Backward Selection Class.

    Args:
        model : sklearn model
            any sklearn model with `.fit()`, `.predict()` and
            `.score()` methods.
        X_train : ndarray
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

    Exposes:
        * .forward()
        * .backward()

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
        self._criterion = criterion
        self._n_features = get_n_features(X_train)

    def _fit_and_score(self, S, feature, algorithm):
        if algorithm == 'forward':
            features = S + [feature]
        else:
            features = [f for f in S if f != feature] if feature else S
        self._model.fit(self._X_train[:, features], self._y_train)
        # ToDo: Add AIC and BIC.
        score = self._model.score(self._X_val[:, features], self._y_val)
        return score

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
                that must lie on (0, 1).

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
                j_model_score = self._fit_and_score(S, feature=j, algorithm='forward')
                if best_score is None or (j_model_score > best_score):
                    j_score_dict[j] = j_model_score

            # 2. Save the best j to S, if possible.
            if len(j_score_dict):
                best_j = max(j_score_dict, key=j_score_dict.get)
                best_j_score = j_score_dict[best_j]
                # Update S, the best score and score history ---
                best_score = best_j_score  # update the score to beat
                S.append(best_j)  # add feature
                itera.remove(best_j)  # no longer search over this feature.

            if self._forward_break_criteria(S, j_score_dict=j_score_dict,
                                            max_features=max_features):
                break

        return S

    def backward(self, n_features=0.5, min_change=None):
        """Run Backward Selection Algorithm.

        Args:
            n_features : int or float
                The number of features to select.
                Floats will be regarded as proportions of the total
                that must lie on (0, 1).
                `min_change` must be None for `n_features` to operate.
            min_change : int or float, optional
                The smallest change to be considered significant.
                `n_features` must be None for `min_change` to operate.

        Returns:
            S : list
              The column indices of `X_train` (and `X_val`) that denote the chosen features.

        Raises:
            if `n_features` and `min_change` are both non-None.

        """
        input_checks(locals())
        S = list(range(self._n_features))  # start with all features

        if n_features:
            n_features = parse_features_param(
                n_features, total=len(S), param_name="n_features"
            )

        last_score = self._fit_and_score(S, feature=None, algorithm='backward')
        worse_case = worse_case_bar(self._n_features, verbose=self._verbose)
        for _ in worse_case:
            worse_case.set_postfix(n_features=len(S), score=last_score)

            # 1. Hunt for the least predictive feature.
            best = None
            for j in S:
                score = self._fit_and_score(S, feature=j, algorithm='backward')
                if score >= last_score and (best is None or score > best[1]):
                    best = (j, score)

            if not best:
                break  # Relent. Removing any `j` yielded a lower score.
            else:
                to_drop, best_new_score = best

            # 2a. Halting Based Blindly Based on `n_features`.
            if isinstance(n_features, int):
                S.remove(to_drop)  # blindly drop.
                last_score = best_new_score
                if len(S) == n_features:
                    break
                else:
                    continue  # i.e., ignore criteria below.

            # 2b. Halt if the change is not longer significant.
            if (best_new_score - last_score) < min_change:
                break
            else:
                S.remove(to_drop)
                last_score = best_new_score

            if len(S) == 1:  # continuing is futile.
                break

        return S
