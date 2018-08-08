#!/usr/bin/env python

"""
Forward and Backward Selection Algorithms
=========================================
"""

from pypunisher.metrics.criterion import aic, bic
from pypunisher._checks import model_check, array_check, input_checks
from pypunisher.selection_engines._utils import (get_n_features,
                                                 enforce_use_of_all_cpus,
                                                 parse_n_features)


class Selection(object):
    """Forward and Backward Selection Algorithms.

    Args:
        model (sklearn model)
            any sklearn model with `.fit()`, `.predict()` and
            `.score()` methods.
        X_train (2d ndarray)
            a 2D numpy array of (observations, features).
        y_train (1d ndarray)
            a 1D array of target classes for X_train.
        X_val (2d ndarray)
            a 2D numpy array of (observations, features).
        y_val (1d ndarray)
            a 1D array of target classes for X_validate.
        criterion (str or None)
            model selection criterion.

            * 'aic': use Akaike Information Criterion.

            * 'bic': use Bayesian Information Criterion.

            * None: Use the model's default (i.e., call ``.score()``).

        verbose (bool)
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

        if criterion not in (None, 'aic', 'bic'):
            raise ValueError("`criterion` must be one of: None, 'aic', 'bic'.")

        self._criterion = criterion
        self._verbose = verbose
        self._total_number_of_features = get_n_features(X_train)

    def _fit_and_score(self, S, feature, algorithm):
        """Fit and score the model

        Args:
            S : list
                The list of features as found in `forward`
                and `backward()`
            feature : int
                The feature to add or drop.
            algorithm : str
                One of: 'forward', 'backward'.

        Returns : float
            The score of the model.

        """
        if algorithm == 'forward':
            features = S + [feature]
        else:
            features = [f for f in S if f != feature] if feature else S
        self._model.fit(self._X_train[:, features], self._y_train)

        X_val, y_val = self._X_val[:, features], self._y_val
        if self._criterion == 'aic':
            # Note: We want to do selection against the validation
            # data, hence `X_train=X_val` and `y_train=y_val`.
            score = aic(self._model, X_train=X_val, y_train=y_val)
        elif self._criterion == 'bic':
            score = bic(self._model, X_train=X_val, y_train=y_val)
        else:
            score = self._model.score(X_val, y_val)
        return score

    @staticmethod
    def _do_not_skip(kwargs):
        """Check for skipping override by looking
        for `_do_not_skip` in keyword arguments
        If it is present, the loops in the algorithms
        will be run to exhaustion.

        Args:
            kwargs : dict
                Keyword Args

        Returns:
            Bool
                If `_do_not_skip` is not present
                      or `_do_not_skip` is present and is True.
                Otherwise, the value of `do_not_skip`
                      is returned.

        """
        return kwargs.get('_do_not_skip', True)

    def _forward_break_criteria(self, S, min_change, best_j_score,
                                j_score_dict, n_features):
        """Check if `forward()` should break

        Args:
            S : list
                The list of features as found in `forward`
                and `backward()`
            min_change : int or float, optional
                The smallest change to be considered significant.
            best_j_score : float
                The best score for a given iteration as evolved
                inside `forward()`.
            j_score_dict : dict
                A dictionary of scores in step 1. of `forward()`.
            n_features : int
                The `n_features` object as developed inide `forward()`.

        Returns:
            bool
                Whether or not `forward()` should halt.

        """
        # a. Check if the algorithm should halt b/c of features themselves
        if not len(j_score_dict) or len(S) == self._total_number_of_features:
            return True
        # b. Break if the change was too small
        if isinstance(min_change, (int, float)) and best_j_score < min_change:
            return True
        # c. Break if the number of features in S > n_features.
        elif isinstance(n_features, int) and n_features > len(S):
            return True
        else:
            return False

    def forward(self, n_features=0.5, min_change=None, **kwargs):
        """Perform Forward Selection on a Sklearn model.

        Args:
            n_features (int)
                the max. number of features to allow.
                Note: ``min_change`` must be None in order for ``n_features`` to operate.
                Floats will be regarded as proportions of the total
                that must lie on (0, 1).
            min_change (int or float)
                The smallest change to be considered significant.
                Note: `n_features` must be None in order for ``min_change`` to operate.
            kwargs (Keyword Args)
                 * `_do_not_skip`:
                    Explore loop exhaustion.
                    **For internal use only**. Not intended for outside use.

        Returns:
            S (list)
              The column indices of ``X_train`` (and ``X_val``) that denote the chosen features.

        Raises:
            if ``n_features`` and ``min_change`` are both non-None.

        """
        input_checks(locals())
        S = list()
        best_score = None
        itera = list(range(self._total_number_of_features))
        do_not_skip = self._do_not_skip(kwargs)

        if n_features and do_not_skip:
            n_features = parse_n_features(n_features, total=len(itera))

        for i in range(self._total_number_of_features):
            if self._verbose:
                print("Iteration: {}".format(i))

            if not do_not_skip:
                continue

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

            if self._forward_break_criteria(S, min_change=min_change,
                                            best_j_score=best_j_score,
                                            j_score_dict=j_score_dict,
                                            n_features=n_features):
                break

        return S

    def backward(self, n_features=0.5, min_change=None, **kwargs):
        """Perform Backward Selection on a Sklearn model.

        Args:
            n_features (int or float)
                The number of features to select.
                Floats will be regarded as proportions of the total
                that must lie on (0, 1).
                ``min_change`` must be None for ``n_features`` to operate.
            min_change (int or float)
                The smallest change to be considered significant.
                `n_features` must be None for ``min_change`` to operate.
            kwargs (Keyword Args)
                * `_do_not_skip` (bool):
                    Explore loop exhaustion.
                    **For internal use only**. Not intended for outside use.
                * `_last_score_punt` (bool):
                    Relax `defeated_last_iter_score` decision boundary.
                    **For internal use only**. Not intended for outside use.

        Returns:
            S (list)
              The column indices of `X_train` (and `X_val`) that denote the chosen features.

        Raises:
            if ``n_features`` and ``min_change`` are both non-None.

        """
        input_checks(locals())
        S = list(range(self._total_number_of_features))  # start with all features
        do_not_skip = self._do_not_skip(kwargs)
        last_score_punt = kwargs.get('_last_score_punt', False)

        if n_features and do_not_skip:
            n_features = parse_n_features(n_features, total=len(S))

        last_iter_score = self._fit_and_score(S, feature=None, algorithm='backward')

        for i in range(self._total_number_of_features):
            if self._verbose:
                print("Iteration: {}".format(i))

            if not do_not_skip:
                continue

            # 1. Hunt for the least predictive feature.
            best = {'feature': None, 'score': None, 'defeated_last_iter_score': True}
            for j in S:
                score = self._fit_and_score(S, feature=j, algorithm='backward')
                if best['score'] is None or score > best['score']:
                    best = {'feature': j, 'score': score,
                            'defeated_last_iter_score': score > last_iter_score}
            to_drop, best_new_score = best['feature'], best['score']

            # 2a. Halting Blindly Based on `n_features`.
            if isinstance(n_features, int):
                S.remove(to_drop)  # blindly drop.
                last_iter_score = best_new_score
                if not len(S) == n_features:
                    continue  # i.e., ignore criteria below.
                else:
                    break
            # 2b. Halt if the change is not longer considered significant.
            else:
                if best['defeated_last_iter_score'] or last_score_punt:
                    if (best_new_score - last_iter_score) < min_change:
                        break  # there was a change, but it was not large enough.
                    else:
                        S.remove(to_drop)
                        last_iter_score = best_new_score
                else:
                    break

        return S