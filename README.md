# PyPunisher

PyPunisher is a package for feature and model selection in Python. Specifically, this package will implement tools for 
forward and backward model selection (see [here](https://en.wikipedia.org/wiki/Stepwise_regression)). 
In order to measure model quality during the selection procedures, we will also be implement
the Akaike and Bayesian Information Criterion (see below), both of which *punish* complex models -- hence this package's
name.

We recognize that these tools already exist in Python. However, as discussed below, we have some minor
misgivings about how this has been done, and believe it is possible to make some improvements in `PyPunisher`.

## Contributors: 

Avinash, Tariq, Jill


## ToDos:

Jill

* function description
    
Tariq

* summary paragraph
    
Avinash 

* where your packages fit into the Python and R ecosystems


## Functions included:

We will be implementing two stepwise feature selection techniques:

- `forward_selection()`: a feature selection method in which you start with a null model and iteratively add useful features 
- `backward_selection()`: a feature selection method in which you start with a full model and iteratively remove the least useful feature at each step

We will also be implementing metrics that evaluate model performance: 

- `aic()`: computes the Akaike information criterion [Akaike information criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion)
- `bic()`: computes the [Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion) 

## How does this Packages Fit into the Existing R and Python Ecosystems?

In the Python ecosystem, forward selection has been implemented in scikit-learn in the 
[f_regression](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html) function.
As stated in the documentation, *this function uses a linear model for testing the individual effect of each of many regressors*.
Similarly, backward selection is also implemented in scikit-learn in the `RFE()` class.
`RFE()` uses an external estimator that assigns weights to features and it prunes the number of features by
recursively considering smaller and smaller sets of features until the desired number of features to select is eventually 
reached (see: [RFE](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)).

In the R ecosystem, forward and backward selection are implemented in both the [olsrr](https://cran.r-project.org/web/packages/olsrr/)
and [MASS](https://cran.r-project.org/web/packages/MASS/MASS.pdf) packages. The former provides
[`ols_step_forward()`](https://www.rdocumentation.org/packages/olsrr/versions/0.4.0/topics/ols_step_forward) and 
[]`ols_step_backward()`](https://www.rdocumentation.org/packages/olsrr/versions/0.4.0/topics/ols_step_backward) for
forward and backward stepwise selection, respectively. Both of these are p-value-based methods of feature selection.
The latter, MASS, contains [`StepAIC()`](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/stepAIC.html),
which is complete with three modes: forward, backward or both. The selection procedure is based on the 
Akaike Information Criterion (AIC). 
