# PyPunisher

[![Build Status](https://travis-ci.org/UBC-MDS/PyPunisher.svg?branch=master)](https://travis-ci.org/UBC-MDS/PyPunisher)
[![Coverage Status](https://coveralls.io/repos/github/UBC-MDS/PyPunisher/badge.svg?branch=coveralls)](https://coveralls.io/github/UBC-MDS/PyPunisher?branch=coveralls)

PyPunisher is a package for feature and model selection in Python. Specifically, this package will implement tools for 
forward and backward model selection (see [here](https://en.wikipedia.org/wiki/Stepwise_regression)). 
In order to measure model quality during the selection procedures, we have also implemented
the Akaike and Bayesian Information Criterion (see below), both of which *punish* complex models -- hence this package's
name.

We recognize that these tools already exist in Python. However, as discussed below, we have some minor
misgivings about how one of these techniques has been implemented, and have made some small some improvements
in `PyPunisher`.

## Installation

```bash
pip3 install git+git://github.com/UBC-MDS/PyPunisher@master
```

Requires Python 3.6+.

## Documentation

The documentation for PyPunisher can be viewed [here](https://ubc-mds.github.io/PyPunisher/index.html).

## Functions included:

We have implemented two stepwise feature selection techniques:

- `forward_selection()`: a feature selection method in which you start with a null model and iteratively add useful features 
- `backward_selection()`: a feature selection method in which you start with a full model and iteratively remove the least useful feature at each step

We have also implemented metrics that evaluate model performance: 

- `aic()`: computes the [Akaike information criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion)
- `bic()`: computes the [Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion) 

These two criteria have been used to measure the *relative* quality of models within `forward_selection()` and `backward_selection()`.
In general, having more parameters in your model increases prediction accuracy but is highly susceptible to overfitting.
AIC and BIC add a penalty for the number of features in a model. The penalty term is larger in BIC than in AIC.
The lower the AIC and BIC score, the better the model.  


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
An alternative approach is to stop removing features the change is trivially small.
We allow users to define a "non-trivial decrease" in our `backward_selection()` function via a parameter `min_change`.


## Examples


### Imports and Create Instance

```python
from pypunisher import Selection, aic, bic
from sklearn.linear_model import LinearRegression
from pypunisher.example_data import X_train, X_val, y_train, y_val, true_best_feature
```

> Note: This example dataset has only one predictive feature. 
> We'll see which one below.

```python
model = LinearRegression()
sel = Selection(model, X_train=X_train, X_val=X_val,
                y_train=y_train, y_val=y_val, verbose=False)
```

### Forward Selection

> Run the forward selection algorithm, i.e., start no features
> and build up a set of length `n_features` that are predictive.

> Let's pretend we know there is a single predictive feature...
> we just down know *which* one!

```python
print(sel.forward(n_features=1))
# [10]
```
> Here we can see that feature 10 was selected. Indeed, it is 
> the best feature!

```python
print(true_best_feature)
# 10
```

### Backward Selection

> We can also go in the opposite direction, i.e., start with all
> of the features and remove the least predictive ones until we
> are left with a set of length `n_features`.

```python
print(sel.backward(n_features=1))
# [10]
```

> Great! Backward selection also obtained the correct answer.

### AIC, BIC


> We can also compute the selection metrics, AIC and BIC.

```python
# Fit the model
_ = model.fit(X_train, y_train)
```


```python
print("AIC", aic(model, X_train=X_train, y_train=y_train))
# AIC 3099.9842544743283
print("BIC", bic(model, X_train=X_train, y_train=y_train))
# BIC 3176.1498936378043
```

> Notes:
>
> * sadly, Sklearn (`LinearRegression`) does not keep a pointer to the data
that was used to train it. So, we have to pass that to `aic()` and `bic()` as well.
> 
> * We could pass 'aic' or 'bic' to the `criterion` parameter of `Selection()`, above,
> if we wanted (by default, it will use sklearns default scoring criterion, 'r-squared')


## How to run unit tests

From root directory, run all test files in terminal:

```
python -m pytest
```

You also have the option to run individual test files by referencing its path. For example: 

```
python -m pytest tests/test_forward_selection.py
```

## Branch Coverage


```bash
mbp:PyPunisher tariq$ py.test --cov=pypunisher tests/ --cov-branch


---------- coverage: platform darwin, python 3.6.3-final-0 -----------
Name                                        Stmts   Miss Branch BrPart  Cover
-----------------------------------------------------------------------------
pypunisher/__init__.py                          5      0      0      0   100%
pypunisher/_checks.py                          21      0     18      0   100%
pypunisher/metrics/__init__.py                  2      0      0      0   100%
pypunisher/metrics/criterion.py                29      0     10      0   100%
pypunisher/selection_engines/__init__.py        3      0      0      0   100%
pypunisher/selection_engines/_utils.py         17      0      8      0   100%
pypunisher/selection_engines/selection.py      99      0     52      0   100%
-----------------------------------------------------------------------------
TOTAL                                         176      0     88      0   100%

```

## Contributors: 

- Avinash, [@avinashkz](https://github.com/avinashkz)
- Tariq, [@TariqAHassan](https://github.com/TariqAHassan/)
- Jill, [@topspinj](https://github.com/topspinj/)

Instructions and guidelines on how to contribute can be found [here](CONTRIBUTING.md).