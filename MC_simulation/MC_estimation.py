""" Call the simulation files and return the estimation resuls
"""
# import packages
import pandas as pd
import numpy as np
import statsmodels.regression.linear_model as sms
import json
import os

# import functions from other modules
from mc_optimal_wage_choice import mc_optimal_wage_choice
from mc_prices import draw_skill_prices

# allows to print DataFrames as whole, not truncated
pd.set_option('display.max_rows', None, 'display.max_columns', None)

# Define indexer and parameters
seed = 600      # set seed for Data Generating Process
N = 100          # number of observations
M = 10          # number of MC interations
idx = pd.IndexSlice     # define Indexslice

# Define optional arguments of DGP.
kwargs = {
          # (1) Arguments for price changing process.
          "pi_fun": "pi_fixed",
          "const": [0.05, 0.1]
          }

# find current working directory.
path = os.path.dirname(__file__)

# calculate true price changes
pi = draw_skill_prices(T=2, J=2, seed=seed, **kwargs)
Dpi_1, Dpi_2 = (pi[:, 1] - pi[:, 0])

# Define Loci and penalty terms that should be calculated
loci = [(0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]
penalty_power = [1.25, 1.5, 2, 2.5, 3]

# Define empty DataFrames that will carry the results of this program
rslt = pd.DataFrame(index=penalty_power, columns=loci)
rslt_difference = pd.DataFrame(index=penalty_power, columns=loci)
count_corner_sol = pd.DataFrame(index=penalty_power, columns=loci)
task_adjustments = pd.DataFrame(index=penalty_power, columns=loci)
# corr_lmb_bar_d_lmb = pd.DataFrame(index=penalty_power, columns=loci)
# corr_d_w_d_lmb = pd.DataFrame(index=penalty_power, columns=loci)
# corr_d_w_lmb_bar = pd.DataFrame(index=penalty_power, columns=loci)

# calculate true parameters for estimators
baseline = (Dpi_1, (Dpi_2-Dpi_1))

# empty dict for Results
rslt_dict = {}

# Run MC Simulation m times.
for m in range(0, M):
    # Simulate data and estimate for each combination of locus and penalty term
    for l in range(0, len(loci)):
        for p in range(0, len(penalty_power)):
            # Simulate data
            mc_data = mc_optimal_wage_choice(
                n=100,
                T=2, J=2,
                penalty='quad',
                p_weight=10,
                p_locus=loci[l],
                p_exponent=penalty_power[p],
                seed=seed,
                **kwargs
                )

            # count the number of cornersolutions
            corner_0 = sum(mc_data.loc[idx[:, "lambda"], 0] == 0) \
                + sum(mc_data.loc[idx[:, "lambda"], 0] == 1)

            corner_1 = sum(mc_data.loc[idx[:, "lambda"], 1] == 0) \
                + sum(mc_data.loc[idx[:, "lambda"], 1] == 1)

            count_corner_sol.iloc[p, l] = (corner_0, corner_1)

            # estimate price changes
            # 1. calculate approximated lambda
            lmb_bar = np.array((mc_data.loc[idx[:, "lambda"], 0]
                               + mc_data.loc[idx[:, "lambda"], 1])/2
                               ).astype(float)

            # 2. get true wage difference
            wage_change = (mc_data.loc[idx[:, "wage"], 1]
                           - mc_data.loc[idx[:, "wage"], 0])

            # run estimation
            beta0, beta1 = sms.OLS(
                   endog=np.array(wage_change).astype(float),
                   exog=sms.add_constant(lmb_bar),
                   ).fit().params.round(4)

            # calculate error in estimated coefficients
            rslt.iloc[p, l] = (beta0, beta1)
            rslt_difference.iloc[p, l] = np.round((beta0-baseline[0],
                                                   beta1-baseline[1]), 4)

            # calculate correlation between lmb_bar and D_lmb
            D_lmb = np.array(mc_data.loc[idx[:, "lambda"], 1]
                             - mc_data.loc[idx[:, "lambda"], 0]).astype(float)

            # As a quality check, take a look at correlation coefficients.
            # corr_lmb_bar_d_lmb.iloc[p, l] = np.corrcoef(lmb_bar, D_lmb).round(4)
            # corr_d_w_d_lmb.iloc[p, l] = np.corrcoef(np.array(wage_change
            #                                                 ).astype(float),
            #                                        D_lmb
            #                                        ).round(4)

            # corr_d_w_lmb_bar.iloc[p, l] = np.corrcoef(np.array(wage_change
            #                                                   ).astype(float),
            #                                          lmb_bar).round(4)

            # store task adjustments
            task_adjustments.iloc[p, l] = np.mean(D_lmb).round(4)
    d = {m: rslt_difference.to_json()}
    rslt_dict.update(d)

# Write results to json file.
# rslt_difference.to_json(
#                         path_or_buf=path+"OUT\\rslt_difference.json",
#                         orient="index",
#                         )
with open(path+"\\OUT\\rslt_dict.json", "w") as json_file:
    json.dump(rslt_dict, json_file)
