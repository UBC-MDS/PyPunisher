# PyPunisher

The PyPunisher package will implement techniques for feature and model selection.
Namely, it will contain tools for forward and backward selection, as well as tools for computing
AIC and BIC (see below). 


## Contributors: 

Avinash
Tariq
Jill

## Functions included:

- Forward selection 
    * `forward_selection()`
- Backward selection
    * `backwards_selection()`
- Metrics: 
    - '`aic()`': Computes the Akaike information criterion [AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion)
    - '`bic()`: Computes the [Bayesian_information_criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion) 

## ToDos:

Jill

    * function description
    
Tariq

    * summary paragraph
    
Avinash 

    * where your packages fit into the Python and R ecosystems


**Due**: Sunday Feb 11, 2018.


## How the packages fit into the existing R and Python ecosystems.

In Python ecosystem, forward selection has been implemented in scikit learn by the
[f_regression](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html) function. The function uses Linear model for testing the individual effect of each of many regressors. It has been implemented as a scoring function to be used in feature seletion procedure. The backward selection has also been implemented in scikit learn by the [RFE](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) function. RFE uses an external estimator that assigns weights to features and it prunes the number of features by recursively considering smaller and smaller sets of features until the desired number of features to select is eventually reached. Whereas, in R ecosystem, forward and backward selection are implemented by [olsrr package](https://cran.r-project.org/web/packages/olsrr/)
and in [MASS package](https://cran.r-project.org/web/packages/MASS/MASS.pdf) by function
[StepAIC](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/stepAIC.html). StepAIC performs stepwise selection (forward, backward, both) by exact AIC.
