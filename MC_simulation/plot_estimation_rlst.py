""" Visualizes the simulated data in histograms.
Split in 2 parts: Part (1) visualizes data on the difference between estimated
and true price changes. Part (2) - not yet implemented - will do the same for
the estimated price changes (not in relation to true changes).
"""

# import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# import other modules
from read_estimation_rslt import read_estimation_rslt
df, df_mean_sd = read_estimation_rslt(M=100)

# Get indices from simluated data.
cols = df.index.get_level_values(level=0).unique()
rows = df.index.get_level_values(level=1).unique()

# Define indexslices.
idx = pd.IndexSlice

plt.style.use('seaborn-whitegrid')
# plt.hist(estimations)

# (1) plots for difference between estimates and true values.
fig, ax = plt.subplots(ncols=len(cols),
                       nrows=len(rows),
                       figsize=(20, 15),
                       sharex=True,
                       sharey=True,
                       dpi=300
                       )

# set fixed number of bins.
nbins = np.arange(-0.3, 0.3, 0.001)

# Draw subplots
for n_c, c in enumerate(cols):
    for n_r, r in enumerate(rows):
        hist = list(zip(*df.loc[idx[str(c), str(r)], :]))[1]
        ax[n_r, n_c].hist(hist, color="blue", bins=nbins)
        ax[n_r, n_c].axvline(x=0.0, color="red")

        # name rows of plot-grid.
        if n_c == 0:
            ax[n_r, n_c].set_ylabel(str(rows[n_r]))

        # name columns of plot-grid
        if n_r == 0:
            ax[n_r, n_c].xaxis.set_label_position('top')
            ax[n_r, n_c].set_xlabel(str(cols[n_c]))

# Set layout and print plot to file.
fig.tight_layout()
path = os.getcwd()      # debugging
fig.savefig(fname=path + "\\FIG\\estimation_differences.png")

"""
# (2) plots for estimated values
fig, ax = plt.subplots(ncols=len(cols),
                       nrows=len(rows),
                       figsize=(20, 15),
                       sharex=True,
                       sharey=True,
                       dpi=300
                       )

# set fixed number of bins.
nbins = np.arange(-1.0, 1.0, 0.01)

# Draw subplots
for n_c, c in enumerate(cols):
    for n_r, r in enumerate(rows):
        hist = list(zip(*df_mean_sd.loc[idx[str(c), str(r)], :]))[1]
        ax[n_r, n_c].hist(hist, color="blue", bins=nbins)

        # name rows of plot-grid.
        if n_c == 0:
            ax[n_r, n_c].set_ylabel(str(rows[n_r]))

        # name columns of plot-grid
        if n_r == 0:
            ax[n_r, n_c].xaxis.set_label_position('top')
            ax[n_r, n_c].set_xlabel(str(cols[n_c]))

# Set layout and print plot to file.
fig.savefig(fname=path + "\\FIG\\estimations.png")
"""
