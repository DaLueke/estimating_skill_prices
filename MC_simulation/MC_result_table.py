""" Illustrates MC simulation data in two separate tables. Both present mean
differences between estimated price change and true price change over all MC
simulations as well as according standard deviations.
(1) Presents cominations of baseline decisions b and penalty exponent p for a
fixed penalty weight tau.
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
df_mc_est, df_mean_sd = read_estimation_rslt(M=100)

# Write data into
baselines = df_mean_sd.index.get_level_values(level=0).unique()
powers = df_mean_sd.index.get_level_values(level=1).unique()

df = pd.DataFrame(columns=baselines, index=powers)
for b in baselines:
    for p in powers:
        # df.loc[p, b] = np.round(tuple(df_mean_sd.loc[pd.IndexSlice[str(b), str(p)], "Mean"]), 4)

        mean = np.round(tuple(df_mean_sd.loc[pd.IndexSlice[str(b), str(p)], "Mean"]), 4)
        sd = np.round(tuple(df_mean_sd.loc[pd.IndexSlice[str(b), str(p)], "STD"]), 4)
        df.loc[p, b] = np.concatenate([mean, sd], axis=0)


df.index.name = "Penalty Powers"
df.columns.name = "Baseline Interval"

print(df.to_latex())
