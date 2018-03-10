# PyPunisher

[![Build Status](https://travis-ci.org/UBC-MDS/PyPunisher.svg?branch=master)](https://travis-ci.org/UBC-MDS/PyPunisher)

PyPunisher is a package for feature and model selection in Python. Specifically, this package will implement tools for 
forward and backward model selection (see [here](https://en.wikipedia.org/wiki/Stepwise_regression)). 
In order to measure model quality during the selection procedures, we will also be implement
the Akaike and Bayesian Information Criterion (see below), both of which *punish* complex models -- hence this package's
name.

We recognize that these tools already exist in Python. However, as discussed below, we have some minor
misgivings about how one of these techniques has been implemented, and believe it is possible to make
some improvements in `PyPunisher`.

## Installation

```bash
pip3 install git+git://github.com/UBC-MDS/PyPunisher@master
```

Requires Python 3.6+.

## Functions included:

We will be implementing two stepwise feature selection techniques:

- `forward_selection()`: a feature selection method in which you start with a null model and iteratively add useful features 
- `backward_selection()`: a feature selection method in which you start with a full model and iteratively remove the least useful feature at each step

We will also be implementing metrics that evaluate model performance: 

- `aic()`: computes the [Akaike information criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion)
- `bic()`: computes the [Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion) 

These two criteria will be used to measure the *relative* quality of models within `forward_selection()` and `backward_selection()`. In general, having more parameters in your model increases prediction accuracy but is highly susceptible to overfitting. AIC and BIC add a penalty for the number of features in a model. The penalty term is larger in BIC than in AIC. The lower the AIC and BIC score, the better the model.  


## How does this package fit into the existing Python ecosystem?

In the Python ecosystem, forward selection has been implemented in scikit-learn in the 
[f_regression](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html) function.
As stated in the documentation, *this function uses a linear model for testing the individual effect of each of many regressors*.
Similarly, backward selection is also implemented in scikit-learn in the `RFE()` class.
`RFE()` uses an external estimator that assigns weights to features and it prunes the number of features by
recursively considering smaller and smaller sets of features until the desired number of features to select is eventually 
reached (see: [RFE](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)).

One characteristic of the `RFE()` class that we dislike is its requirement that the user
specify the number of features to select (see the `n_features_to_select` parameter). This strikes us
as a rather crude solution because it is almost never obvious what a sensible value would be.
An alternative approach is to stop removing features when even the least predictive feature produces a
non-trivial decrease in model performance. We hope to allow users to define a "non-trivial decrease" in our
`backward_selection()` function via a parameter.


## How to run unit tests

From root directory, run all test files in terminal:

```
python -m pytest
```

You also have the option to run individual test files by referencing its path. For example: 

```
python -m pytest tests/test_forward_selection.py
```

## Coverage

```bash

$ py.test --cov=pypunisher tests/

platform darwin -- Python 3.6.3, pytest-3.4.1, py-1.5.2, pluggy-0.6.0
...
plugins: cov-2.5.1
collected 13 items                                                                                                                                                           

...

---------- coverage: platform darwin, python 3.6.3-final-0 -----------
Name                                        Stmts   Miss  Cover
---------------------------------------------------------------
pypunisher/__init__.py                          4      0   100%
pypunisher/_checks.py                          19      1    95%
pypunisher/metrics/__init__.py                  1      0   100%
pypunisher/metrics/criterion.py                28      3    89%
pypunisher/selection_engines/__init__.py        2      0   100%
pypunisher/selection_engines/_utils.py         20      2    90%
pypunisher/selection_engines/selection.py      90     11    88%
---------------------------------------------------------------
TOTAL                                         164     17    90%
```

## Contributors: 

Avinash, Tariq, Jill.
