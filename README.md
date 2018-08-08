## PyPunisher <img src="docs/logo/pypunisher_logo.png" align="right"/>

[![Build Status](https://travis-ci.org/UBC-MDS/PyPunisher.svg?branch=master)](https://travis-ci.org/UBC-MDS/PyPunisher)
[![Coverage Status](https://coveralls.io/repos/github/UBC-MDS/PyPunisher/badge.svg?branch=coveralls)](https://coveralls.io/github/UBC-MDS/PyPunisher?branch=coveralls)

**PyPunisher** is a Python implementation of forward and backward feature selection. Feature selection, or [stepwise regression](https://www.wikiwand.com/en/Stepwise_regression), is a key step in the data science pipeline that reduces model complexity by selecting the most relevant features from the original dataset. In order to measure model quality during the selection procedures, we have also implemented the Akaike and Bayesian Information Criterion (see below), both of which *punish* complex models -- hence this package's
name.

This package implements two stepwise feature selection techniques: 

- `forward_selection()`: starts with a **null model** and iteratively **adds** useful features 
- `backward_elimination()`: starts with a **full model** and iteratively **removes** the least useful feature at each step

We have also implemented metrics that evaluate model performance: 

- `aic()`: computes the [Akaike information criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion)
- `bic()`: computes the [Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion) 

These two criteria have been used to measure the relative quality of models within `forward_selection()` and `backward_elimination()`. In general, having more parameters in your model increases prediction accuracy but is highly susceptible to overfitting. AIC and BIC add a penalty for the number of features in a model. This penalty term is larger in BIC than in AIC. A lower AIC or BIC score indicates a better fit for the data, relative to competing models.  


## Installation

```bash
pip3 install git+git://github.com/UBC-MDS/PyPunisher@master
```

Requires Python 3.6+.

## Documentation

The documentation for PyPunisher can be viewed [here](https://ubc-mds.github.io/PyPunisher/index.html).


## Example Usage

### Imports and Create Instance

```python
from pypunisher import Selection, aic, bic
from sklearn.linear_model import LinearRegression
from pypunisher.example_data import X_train, X_val, y_train, y_val, true_best_feature
```

**Note:** PyPunisher comes with `example_data` for testing and tutorial purposes. This example dataset has only one predictive feature. We'll see which one below.

```python
model = LinearRegression()
sel = Selection(model, X_train=X_train, X_val=X_val,
                y_train=y_train, y_val=y_val, verbose=False)
```

### Forward Selection

Run the forward selection algorithm. Start with no features and build up a set of length `n_features`.

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

We can also go in the opposite direction, and start with all of the features and remove the least predictive ones until we are left with a set of length `n_features`.

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

## Contributions

Instructions and guidelines on how to contribute can be found [here](CONTRIBUTING.md).