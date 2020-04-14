""" This file contains the MC estimations that I perform as part of my thesis.
As a varyfying excercise I will, first, regress changes in utility on the
average task choice. If the identification strategy of my thesis is correct,
then this should result in an unbiased estimator.
Second, I need to regress observable wages on average task choices while
controling for changes in the amenity term. This resembles the "real world"
application of my model.
"""

# import needed packaged and functions
import statsmodels.api as sms
import pandas as pd
import numpy as np
import sys
import os

from DGP.draw_data import draw_simulation_data


# Make this function callable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

##### DEBUGGING
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + "\\DGP")

#####


# Draw simulatioon data
# Define optional arguments of DGP.
kwargs = {
          # (1) Arguments for price changing process.
          "pi_fun": "pi_fixed",
          "const": [0.05, 0.1]
          }

sim_data = draw_simulation_data(
    T=2,
    N=100,
    J=2,
    penalty="quad",
    p_weight=35,
    p_locus=(0.3, 0.7),
    p_exponent=1.5,
    **kwargs
)

idx = pd.IndexSlice
mean_lmb = np.array(
    sim_data.loc[idx[0, :], "lambda"].values +
    sim_data.loc[idx[1, :], "lambda"].values
    )/2

diff_lmb = np.array(
    sim_data.loc[idx[1, :], "lambda"].values -
    sim_data.loc[idx[0, :], "lambda"].values
    )

util_change = np.array(
    sim_data.loc[idx[1, :], "utility"].values -
    sim_data.loc[idx[0, :], "utility"].values
    )

wage_change = np.array(
    sim_data.loc[idx[1, :], "wage"].values -
    sim_data.loc[idx[0, :], "wage"].values
    )

#### u ~ mean_lmb
pi_1, tilde_pi = sms.OLS(
    endog=util_change.astype(float),
    exog=sms.add_constant(mean_lmb.astype(float))
).fit().params.round(4)
print("In a model that only uses mean task choice as predictor of utility",
      " changes, estimated rel. wage change is {1}".format(pi_1, tilde_pi))

#### w ~ mean_lmb
pi_1, tilde_pi = sms.OLS(
    endog=wage_change.astype(float),
    exog=sms.add_constant(mean_lmb.astype(float))
).fit().params.round(4)

print("In a model that only uses mean task choice as predictor of wage",
      " changes, estimated rel. wage change is {1}".format(pi_1, tilde_pi))

#### w ~ mean_lmb + diff_lmb
exog = pd.DataFrame(data={
    "C": np.ones(100),      # Adjust to N
    "mean_lmb": mean_lmb.astype(float),
    "diff_lmb": diff_lmb.astype(float)
    })
pi_1, tilde_pi, v = sms.OLS(
    endog=wage_change.astype(float),
    exog=exog[["C", "mean_lmb", "diff_lmb"]]
).fit().params.round(4)

print("In a model that uses differences in lmb in addition to mean task ",
      "choice as predictor of wage changes, estimated rel. wage change is",
      " {1}. The amentity coefficient is {2}.".format(pi_1, tilde_pi, v)
      )

#### Correcting wages for changes in amenities
corrected_wage = wage_change.astype(float) - v * diff_lmb.astype(float)

pi_1, tilde_pi = sms.OLS(
    endog=corrected_wage.astype(float),
    exog=sms.add_constant(mean_lmb.astype(float))
).fit().params.round(4)
print("Correcting wage change by the estimated amentiy change and using only",
      " mean task choice as predictor of wage",
      " changes, estimated rel. wage change is {1}".format(pi_1, tilde_pi))


#### including an interacting term of mean_lmb and diff_lmb
#### w ~ mean_lmb + diff_lmb + mean_lmb*diff_lmb
exog["inter"] = (mean_lmb * diff_lmb).astype(float)
pi_1, tilde_pi, inter = sms.OLS(
    endog=wage_change.astype(float),
    exog=exog[["C", "mean_lmb", "inter"]]
).fit().params.round(4)
print(tilde_pi, inter)
