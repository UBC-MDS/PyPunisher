## PyPunisher <img src="docs/logo/pypunisher_logo.png" align="right"/>

[![Build Status](https://travis-ci.org/UBC-MDS/PyPunisher.svg?branch=master)](https://travis-ci.org/UBC-MDS/PyPunisher)
[![Coverage Status](https://coveralls.io/repos/github/UBC-MDS/PyPunisher/badge.svg?branch=coveralls)](https://coveralls.io/github/UBC-MDS/PyPunisher?branch=coveralls)

**PyPunisher** is a Python implementation of forward and backward feature selection. Feature selection, or [stepwise regression](https://www.wikiwand.com/en/Stepwise_regression), is a key step in the data science pipeline that reduces model complexity by selecting the most relevant features from the original dataset. This package implements two stepwise feature selection methods: 

- `forward_selection()`: starts with a **null model** and iteratively **adds** useful features 
- `backward_elimination()`: starts with a **full model** and iteratively **removes** the least useful feature at each step

These methods are greedy search algorithms that yield a nested subset of features. The size of the final feature subset depends on what you define as your "stopping criterion". The stopping criterion can be either a threshold that you define, or a pre-defined number of features to include in your model. For example, if you set your stopping criterion to be a threshold (`min_change`), then the feature selection process will stop when the AIC or BIC score no longer improves by that thresholded interval. Alternatively, if you want a specific number of features in your model, then the process will stop once it reaches `n_features`.

In order to measure model quality during the selection procedures, we have also implemented the Akaike and Bayesian Information Criterion, both of which *punish* complex models:

- `aic()`: computes the [Akaike Information Criterion (AIC)](https://en.wikipedia.org/wiki/Akaike_information_criterion)
- `bic()`: computes the [Bayesian Information Criterion (BIC)](https://en.wikipedia.org/wiki/Bayesian_information_criterion)

In general, having more parameters in your model increases prediction accuracy but is highly susceptible to overfitting. AIC and BIC add a penalty for the number of features in a model. This penalty term is larger in BIC than in AIC. A lower AIC or BIC score indicates a better fit for the data, relative to competing models.  


## Installation

```bash
pip3 install git+git://github.com/UBC-MDS/PyPunisher@master
```

Requires Python 3.6+.

## Documentation

The documentation for PyPunisher can be viewed [here](https://ubc-mds.github.io/PyPunisher/index.html).


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