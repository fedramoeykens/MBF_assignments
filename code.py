# -*- coding: utf-8 -*-
"""

@author: zvdvelde
"""

########################## IMPORT MODULES #################################
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import pandas as pd

### Solution exercises Class 3 - Functions & Regressions ###

# 12

def global_min_variance_portfolio(returns):
    """
    Calculate the weights of the Global Minimum Variance Portfolio (GMVP)
    
    Parameters:
        returns (numpy.ndarray): Array of asset returns with shape (n_periods, n_assets)
        
    Returns:
        numpy.ndarray: Column vector of portfolio weights (n_assets x 1)
    """
    # Calculate the covariance matrix
    covariance_matrix = np.cov(returns.T)
    
    # Number of assets
    n_assets = returns.shape[1]
    
    # Vector of ones
    ones_array = np.ones((n_assets, 1))
    
    # Compute GMVP weights
    inv_cov = np.linalg.inv(covariance_matrix)
    gmvp_weights = (inv_cov @ ones_array) / (ones_array.T @ inv_cov @ ones_array)
    
    return gmvp_weights

# 13

data_boston = pd.read_csv("C:/Users/Julie/Documents/manama banking and finance/data science for finance and insurance/Boston.csv")
data_boston.drop(["Unnamed: 0", "black", "lstat"], axis = 1, inplace = True)

# Fit models
model1 = smf.ols('medv ~ ptratio + nox + crim', data = data_boston).fit()
model2 = smf.ols('medv ~ ptratio + nox + crim + age', data = data_boston).fit()
model3 = smf.ols('medv ~ ptratio + nox + crim + age + dis', data = data_boston).fit()
model4 = smf.ols('medv ~ ptratio + nox + crim + age + crim:ptratio', data = data_boston).fit()
model5 = smf.ols('medv ~ ptratio + nox + crim + age + I(age ** 2) + dis', data = data_boston).fit()

# Combine results
results_table = summary_col(
    [model1, model2, model3, model4, model5],
    stars=True,
    float_format="%0.4f",
    model_names=["(1)", "(2)", "(3)", "(4)", "(5)"],
    regressor_order=["Intercept", "ptratio", "nox", "crim", "age",
                     "I(age ** 2)", "dis", "crim:ptratio"],
    info_dict={"N": lambda x: f"{int(x.nobs)}",
               "Adj. R2": lambda x: f"{x.rsquared_adj:.3f}"},
)

print(results_table)

'''
Based on the fact that the coefficient for 'ptratio' is negative and highly
significant in all set-ups, it seems that lowering the pupil-teacher ratio will
have a positive effect on median house prices.
'''

# 14

'''
Model D: The coefficient on the interaction term crim:ptratio (beta_5) measures
how the marginal effect of one regressor on medv depends on the level of the
other. Specifically, the effect of a one-unit increase in ptratio on medv is
given by beta_1 + beta_5 * crim, so beta_5 captures the change in the effect of
ptratio on median house values (in $1,000s) for each one-unit increase in the
per capita crime rate. Symmetrically, the effect of crim is beta_3 + beta_5 *
ptratio. Because the estimated coefficient is not statistically significant, we
cannot reject the null hypothesis that beta_5 = 0. The data therefore provide no
evidence that the effect of the pupil-teacher ratio on house prices varies with
the crime rate, although this does not establish that no such interaction
exists.
'''

'''
Model E: The coefficient on age squared (beta_5) captures non-linearity in the
relationship between age (the proportion of owner-occupied units built prior to
1940) and medv. Because age and age^2 cannot change independently, beta_5
cannot be interpreted in isolation. Instead, the marginal effect of age on
median house values is beta_4 + 2 * beta_5 * age, so beta_5 governs how this
effect changes as age increases. A positive beta_5 implies a convex
relationship, in which the effect of age becomes more positive (or less
negative) at higher values of age; a negative beta_5 implies a concave
relationship. A statistically significant beta_5 would therefore indicate that
the relationship between age and house prices is quadratic rather than linear.
'''