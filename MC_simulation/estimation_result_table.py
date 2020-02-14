""" Illustrates MC simulation data in two separate tables. Both present mean
differences between estimated price change and true price change over all MC
simulations as well as according standard deviations.
(1) Presents cominations of baseline decisions b and penalty exponent p for a
fixed penalty weight tau = 30.
(2) Presents combinations of baseline decisions b and pealty weights tau for a
fixed penalty exponent of p = 2.
"""

# import packages
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import os

# import other modules
from read_estimation_rslt import read_estimation_rslt

# Define what Datafile to run on
file = "rslt_difference_dict_pw"
# file = "rslt_dict"

df, df_mean_sd = read_estimation_rslt(M=10, file=file)

# Write data into dataframe.
# Initialize dataframe.
baselines = df_mean_sd.index.get_level_values(level=0).unique()
penalty = df_mean_sd.index.get_level_values(level=1).unique()

df = pd.DataFrame(columns=baselines, index=penalty)
for b in baselines:
    for p in penalty:
        mean = np.round(
                        tuple(df_mean_sd.loc[pd.IndexSlice[str(b), str(p)],
                                             "Mean"
                                             ]
                              ),
                        4
                        )
        sd = np.round(tuple(df_mean_sd.loc[pd.IndexSlice[str(b), str(p)],
                                           "STD"
                                           ]
                            ),
                      4)
        df.loc[p, b] = np.concatenate([mean, sd], axis=0)

# Style and format for each choice of parameters.
if file[-2:] == "pp":
    df.index.name = "Penalty Powers"
    df.columns.name = "Baseline Interval"

if file[-2:] == "pw":
    df.index.name = "Penalty Weights"
    df.columns.name = "Baseline Interval"

print(df.to_latex())
