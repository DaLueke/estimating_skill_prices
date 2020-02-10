""" Vizualize the results from my MC Simulations.
Illustrate the distribution of estimated price changes."""

# import packages
import json
import os
import pandas as pd
import numpy as np


# import functions from other modules


# set parameters
M = 10

# find current working directory.
path = os.path.dirname(__file__)


# import data from MC Simulation
with open(path + "\\OUT\\rslt_dict.json") as data_file:
    data = json.load(data_file)

# Loop through all combinations of parameters of the model and receive
# estimated results.
cols = pd.read_json(data["0"], convert_axes=False).columns
rows = pd.read_json(data["0"], convert_axes=False).index

# create MultiIndex, first level is b, second level is p
idx = pd.IndexSlice
index = pd.MultiIndex(levels=[list(cols), list(rows)],
                      codes=[np.repeat(range(len(cols)),
                             len(rows)), list(range(len(rows)))*len(cols)
                             ]
                      )

# Initialize empty dataFrame with hierarchical multiindex for all data.
df = pd.DataFrame(index=index, columns=(range(M)))

# Initialize equivalent df for mean and sd for all combinations of b and p.
df_mean_sd = pd.DataFrame(index=index, columns=[["Mean", "STD"]])

# read simulated data into dataframe+
for c in cols:
    for r in rows:
        for m in range(0, M):
            v = pd.read_json(data[str(m)],
                             convert_axes=False
                             ).loc[str(r), str(c)]
            df.loc[idx[str(c), str(r)], m] = v

        # Caldulate mean and STD over all M iterations.
        mean = tuple(map(np.mean, zip(*df.loc[idx[str(c), str(r)], :])))
        std = tuple(map(np.std, zip(*df.loc[idx[str(c), str(r)], :])))

        # Store mean and STD in corresponding df.
        df_mean_sd.loc[idx[str(c), str(r)], "Mean"] = [[mean]]
        df_mean_sd.loc[idx[str(c), str(r)], "STD"] = [[std]]
