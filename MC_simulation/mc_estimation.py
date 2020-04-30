""" This file contains the MC estimations that I perform as part of my thesis.
As a varyfying excercise I will, first, regress changes in utility on the
average task choice. If the identification strategy of my thesis is correct,
then this should result in an unbiased estimator.
Second, I need to regress observable wages on average task choices while
controling for changes in the amenity term. This resembles the "real world"
application of my model.
"""

# import needed packaged and functions
import matplotlib.pyplot as plt
import statsmodels.api as sms
import pandas as pd
import numpy as np
import sys
import os

from DGP.draw_data import draw_simulation_data

# Make this function callable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

##### DEBUGGING #####
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + "\\DGP")
#####################


# Define optional arguments of DGP.
kwargs = {
          # (1) Arguments for price changing process.
          "pi_fun": "pi_fixed",
          "const": [0.0, 0.1]
          }
# set number of MC iterations
np.random.seed(100)
M = 1000
N = 100

# initialize result arrays
utility_on_mean_lmb = np.empty(M)
adj_wage_on_mean_lmb = np.empty(M)
wage_on_mean_lmb = np.empty(M)
wage_on_mean_and_diff_lmb = np.empty(M)

# Start MC loop
for m in range(M):
    # Draw simulatioon data
    sim_data = draw_simulation_data(
        T=3,
        N=N,
        J=2,
        penalty="quad",
        p_weight=15,
        p_locus=(0.3, 0.7),
        p_exponent=2,
        **kwargs
    )

    idx = pd.IndexSlice

    # Calculate mean task choice in base period
    mean_lmb_base = np.array(
        sim_data.loc[idx[0, :], "lambda"].values +
        sim_data.loc[idx[1, :], "lambda"].values
        )/2

    # Calculate chage in task choice in base period
    diff_lmb_base = np.array(
        sim_data.loc[idx[1, :], "lambda"].values -
        sim_data.loc[idx[0, :], "lambda"].values
        )

    # utility change in base period
    util_change_base = np.array(
        sim_data.loc[idx[1, :], "utility"].values -
        sim_data.loc[idx[0, :], "utility"].values
        )

    # wage change in base period
    wage_change_base = np.array(
        sim_data.loc[idx[1, :], "wage"].values -
        sim_data.loc[idx[0, :], "wage"].values
        )

    # Estimate changes in amenities and skill accumulation in base period
    exog_base = pd.DataFrame(data={
        "const": np.ones(N),
        "mean_lmb_base": mean_lmb_base.astype(float),
        "diff_lmb_base": diff_lmb_base.astype(float),
        "lmb_inter_base": mean_lmb_base.astype(float) *
        diff_lmb_base.astype(float),
        "lmb_0": sim_data.loc[idx[0, :], "lambda"].values.astype(float),
        "lmb_0^2": sim_data.loc[idx[0, :], "lambda"].values.astype(float)**2,
        "lmb_1": sim_data.loc[idx[1, :], "lambda"].values.astype(float),
        "lmb_1^2": sim_data.loc[idx[1, :], "lambda"].values.astype(float)**2,
        "mean_lmb_inter_lmb_0": mean_lmb_base.astype(float) *
        sim_data.loc[idx[0, :], "lambda"].values.astype(float),
        })

    OLS_base_period_result = sms.OLS(
        endog=wage_change_base.astype(float),
        exog=exog_base[[
            # "const",
            # "lmb_0",
            "lmb_1",
            # "lmb_0^2",
            "lmb_1^2",
            # "mean_lmb_base",
            # "diff_lmb_base",
            # "lmb_inter_base"
            ]]
        ).fit()

    # OLS_base_period_result.summary()
    # print(OLS_base_period_result.summary().as_latex())

    # Use estimation result to adjust wages changes for skill and
    # amenity changes in subsequent periods

    # Calculate mean task choice in estimation period
    mean_lmb_est = np.array(
        sim_data.loc[idx[2, :], "lambda"].values +
        sim_data.loc[idx[1, :], "lambda"].values
        )/2

    # Calculate chage in task choice in estimation period
    diff_lmb_est = np.array(
        sim_data.loc[idx[2, :], "lambda"].values -
        sim_data.loc[idx[1, :], "lambda"].values
        )

    # wage change in estimation period
    wage_change_est = np.array(
        sim_data.loc[idx[2, :], "wage"].values -
        sim_data.loc[idx[1, :], "wage"].values
        )

    # wage change in estimation period
    util_change_est = np.array(
        sim_data.loc[idx[2, :], "utility"].values -
        sim_data.loc[idx[1, :], "utility"].values
        )

    # Write exogeneous variables into df
    exog_est = pd.DataFrame(data={
        "const": np.ones(N),
        "mean_lmb_est": mean_lmb_est.astype(float),
        "diff_lmb_est": diff_lmb_est.astype(float),
        "lmb_inter_est": mean_lmb_est.astype(float)*diff_lmb_est.astype(float),
        "lmb_1": sim_data.loc[idx[1, :], "lambda"].values.astype(float),
        "lmb_1^2": sim_data.loc[idx[1, :], "lambda"].values.astype(float)**2,
        "lmb_2": sim_data.loc[idx[2, :], "lambda"].values.astype(float),
        "lmb_2^2": sim_data.loc[idx[2, :], "lambda"].values.astype(float)**2,
        "mean_lmb_inter_lmb_1": mean_lmb_est.astype(float) *
        sim_data.loc[idx[1, :], "lambda"].values.astype(float),
        })

    # Adjust wage change for predicted changes in skills and amentities
    wage_change_est_adjusted = wage_change_est - \
        OLS_base_period_result.predict(
            exog_est[[
                # "mean_lmb_est",
                # "const",
                # "lmb_1",
                "lmb_2",
                # "lmb_1^2",
                "lmb_2^2",
                # "diff_lmb_est",
                # "lmb_inter_est"
                ]]
            )

    # Regress adjusted wage changes on mean_lmb
    pi_tilde = sms.OLS(
        endog=wage_change_est_adjusted.astype(float),
        exog=mean_lmb_est.astype(float)
        ).fit().params.round(6)
    adj_wage_on_mean_lmb[m] = pi_tilde

    # Regress non-adjusted wages on mean_lmb
    # pi_tilde = sms.OLS(
    #     endog=wage_change_est.astype(float),
    #     exog=mean_lmb_est.astype(float)
    #     ).fit().params.round(6)
    # wage_on_mean_lmb[m] = pi_tilde

###############
#### Plots ####
###############

# Plot estimator for mean_lmb from adjusted w ~ mean_lmb
plt.rc('axes', axisbelow=True)
plt.xticks(rotation=45)
plt.xlabel("estimated rel. price change")
plt.ylabel("frequency")
plt.rc('grid', linestyle="-", color='black')
plt.grid(True)
plt.hist(adj_wage_on_mean_lmb, bins=1000, range=(-0.05, 0.05))
plt.figtext(
    x=0,
    y=-0.25,
    s='Distribution of the estimated change in relative skill prices with ' +
      'realized wage as regressand \n' +
      'and mean lambda as regressor' +
      'in a Monte Carlo Simulation with M = 500 repititions \nand N = 300 ' +
      'simulated individuals.' +
      'True relative price change in the underlying DGP: 0.05.'
)
plt.savefig("FIG/MC_rslt_wage_on_mean_lmb.png", bbox_inches="tight")

# Plot estimator for mean_lmb from non-adjusted w ~ mean_lmb
plt.rc('axes', axisbelow=True)
plt.xticks(rotation=45)
plt.xlabel("estimated rel. price change")
plt.ylabel("frequency")
plt.rc('grid', linestyle="-", color='black')
plt.grid(True)
plt.hist(wage_on_mean_lmb, bins=100, range=(0.175, 0.225))
plt.figtext(
    x=0,
    y=-0.25,
    s='Distribution of the estimated change in relative skill prices with ' +
      'realized wage as regressand \n' +
      'and mean lambda as regressor' +
      'in a Monte Carlo Simulation with M = 500 repititions \nand N = 300 ' +
      'simulated individuals.' +
      'True relative price change in the underlying DGP: 0.05.'
)





#
# # Plot estimator for mean_lmb from w ~ mean_lmb + diff_lmb
# plt.rc('axes', axisbelow=True)
# plt.xticks(rotation=45)
# plt.xlabel("estimated rel. price change")
# plt.ylabel("frequency")
# plt.rc('grid', linestyle="-", color='black')
# plt.grid(True)
# plt.hist(wage_on_mean_and_diff_lmb, bins=30, range=(0.0495, 0.0515))
# plt.figtext(
#     x=0,
#     y=-0.25,
#     s='Distribution of the estimated change in relative skill prices with ' +
#       'realized wage as regressand \n' +
#       'and mean lambda as well as changes in lmb as regressors ' +
#       'in a Monte Carlo Simulation with M = 500 repititions \nand N = 500 ' +
#       'simulated individuals. ' +
#       'True relative price change in the underlying DGP: 0.05.'
# )
# plt.savefig("FIG/MC_rslt_wage_on_mean_and_diff_lmb.png", bbox_inches="tight")
#
#
# # Plot estimator for mean_lmb from u ~ mean_lmb
# plt.rc('axes', axisbelow=True)
# plt.xticks(rotation=45)
# plt.xlabel("estimated rel. price change")
# plt.ylabel("frequency")
# plt.rc('grid', linestyle="-", color='black')
# plt.grid(True)
# plt.hist(utility_on_mean_lmb, bins=30, range=(0.0495, 0.051))
# plt.figtext(
#     x=0,
#     y=-0.25,
#     s='Distribution of the estimated change in relative skill prices with ' +
#       'utility as regressand \n' +
#       'and mean lambda as regressor ' +
#       'in a Monte Carlo Simulation with M = 500 repititions \nand N = 500 ' +
#       'simulated individuals. ' +
#       'True relative price change in the underlying DGP: 0.05.'
# )
# plt.savefig("FIG/MC_rslt_utility_on_mean_lmb.png", bbox_inches="tight")
