""" Wrapper: Build a Dataframe with optimal lambda, wage and utility for each
individual and each year.
Store this is a Multiindex DataFrame: First level is the individual, second
level is lambda, wage, utility. Columns are years.
"""
# import packages
import pandas as pd
import numpy as np
import statsmodels.regression.linear_model as sms
import matplotlib.pyplot as plt

from mc_optimal_wage_choice import mc_optimal_wage_choice
# import functions from other modules

# Simulate data
seed = 600
mc_data = mc_optimal_wage_choice(
    n=100,
    T=2, J=2,
    penalty='quad',
    p_weight=30,
    p_locus=[0.4, 0.6],
    p_exponent=2,
    seed=seed
    )

plt.scatter(mc_data.loc[idx[:, "lambda"], 0], mc_data.loc[idx[:, "lambda"], 1])
plt.plot([0,1], [0,1], 'r')
# count the number of cornersolutions
corner_0 = sum(mc_data.loc[idx[:, "lambda"], 0] == 0) \
           + sum(mc_data.loc[idx[:, "lambda"], 0] == 1)

corner_1 = sum(mc_data.loc[idx[:, "lambda"], 1] == 0) \
           + sum(mc_data.loc[idx[:, "lambda"], 1] == 1)
# estimate price changes frmo t=1 perspective
# calculate approximated lmb_bar (for now: for t=0 und t=1)
idx = pd.IndexSlice
lmb_bar = (mc_data.loc[idx[:, "lambda"], 0] + mc_data.loc[idx[:, "lambda"], 1])/2

(mc_data.loc[idx[:, "lambda"], 1] - mc_data.loc[idx[:, "lambda"], 0])


# get true wage difference
wage_change = mc_data.loc[idx[:, "wage"], 1] - mc_data.loc[idx[:, "wage"], 0]

# estimate wage change function with the constraint plugged in:
# Dw = Dpi_2 + lmb_bar*(Dpi_1 - Dpi_2)
df_exog = pd.DataFrame(data={
    "lmb_bar": np.array(lmb_bar).astype(float),
    "lmb_bar^2": np.array(lmb_bar).astype(float)**2,
    "lmb_bar^3": np.array(lmb_bar).astype(float)**3,
    "lmb_prev": np.array(mc_data.loc[idx[:, "lambda"], 0]).astype(float),
    "lmb_post": np.array(mc_data.loc[idx[:, "lambda"], 1]).astype(float),
    "D_lmb": np.array(mc_data.loc[idx[:, "lambda"], 1] - mc_data.loc[idx[:, "lambda"], 0]).astype(float),
    "D_lmb^2": np.array(mc_data.loc[idx[:, "lambda"], 1] - mc_data.loc[idx[:, "lambda"], 0]).astype(float)**2,
    "D_lmb^3": np.array(mc_data.loc[idx[:, "lambda"], 1] - mc_data.loc[idx[:, "lambda"], 0]).astype(float)**3,
    })

model_1 = sms.OLS(endog=np.array(wage_change).astype(float),
                  exog=sms.add_constant(df_exog[["lmb_bar"]]),
                  ).fit()
model_1.summary()

model_2 = sms.OLS(
       endog=np.array(wage_change).astype(float),
       exog=sms.add_constant(df_exog[["lmb_bar", "D_lmb", "D_lmb^2", "D_lmb^3"]]),
       #missing="raise",
       #hasconst=Truel-
       ).fit()
model_2.summary()


##### For comparison: see true values:
from mc_prices import draw_skill_prices
pi = draw_skill_prices(T=2, J=2, seed=seed)
# true prices
pi
# true price changes
pi[:, 1] - pi[:, 0]

# true skills
from mc_skills import draw_skills
draw_skills(J=2, seed=seed, n=100)

###### compare true wage change and approx wage change
# get approximated wage difference with true prices
wage_change_approx = lmb_bar * (pi[0, 1] - pi[0, 0]) + (1-lmb_bar) * (pi[1, 1] - pi[1, 0])
plt.scatter(wage_change, wage_change_approx)
plt.plot([-0.2,0.2], [-0.2,0.2], 'k-', color = 'r')
plt.scatter(lmb_bar, wage_change)

### ilustrate my estimation results:
# calculate the predicted wage change for predicted price changes and
# interpolated lambda
pred_wage_change = est_l1 * np.array(lmb_bar).astype(float) + est_l2 * (1-np.array(lmb_bar).astype(float))
df_plot_data = pd.DataFrame(data={
    "lmb": np.array(lmb_bar).astype(float),
    "pred_wage_change": pred_wage_change,
    "true_wage_change": np.array(wage_change)
    })


plt.scatter(df_plot_data["lmb"], df_plot_data["true_wage_change"])

### i see that wage changes are high for low lambda_bars and vice versa.
## that makes sense, because people who have low lmb put a lot into (2) already.
## So they profit most from increase in lmb.

# next. look at wage changes and initial lambdas
plt.scatter(x=np.array(mc_data.loc[idx[:, "lambda"], 0]), y=df_plot_data["true_wage_change"])
## people that put a lot in 2 already profit most!

#### see how lambdas changed
plt.scatter(x=np.array(mc_data.loc[idx[:, "lambda"], 0]), y=np.array(mc_data.loc[idx[:, "lambda"], 1]))

#### plot changes in lmb against inital lambda
plt.scatter(np.array(mc_data.loc[idx[:, "lambda"], 0]), np.array(mc_data.loc[idx[:, "lambda"], 1])-np.array(mc_data.loc[idx[:, "lambda"], 0]))

####### check potential wages and optimally chosen lmb (t=0)
pot_wage_1, pot_wage_2 = np.empty(n),  np.empty(n)
for i in range(n):
    pot_wage_1[i] = (sim_skills[i, :] + sim_prices[:, 0])[0]
    pot_wage_2[i] = (sim_skills[i, :] + sim_prices[:, 0])[1]

# scatter potential wages and lmb
plt.scatter(np.array(mc_data.loc[idx[:, "lambda"], 0]), pot_wage_1)

plt.scatter(np.array(mc_data.loc[idx[:, "lambda"], 0]), pot_wage_2)

##### these look just like they should. lmb increases in potential wage in (1)
##### and decreases in potential wage in (2)


####### check potential wages and optimally chosen lmb (t=1)
pot_wage_1, pot_wage_2 = np.empty(n),  np.empty(n)
for i in range(n):
    pot_wage_1[i] = (sim_skills[i, :] + sim_prices[:, 1])[0]
    pot_wage_2[i] = (sim_skills[i, :] + sim_prices[:, 1])[1]

# scatter potential wages and lmb
plt.scatter(np.array(mc_data.loc[idx[:, "lambda"], 1]), pot_wage_1)

plt.scatter(np.array(mc_data.loc[idx[:, "lambda"], 1]), pot_wage_2)

##### Again, these look just like they should. lmb increases in potential wage in (1)
##### and decreases in potential wage in (2)!

#### Lets have a look at the distribution of error terms wrt explanatory
#### variable lmb_bar
# Get prediction errors
pred_error = np.array(wage_change) - model_1.predict(sms.add_constant(df_exog["lmb_bar"]))
plt.scatter(df_exog["lmb_bar"], pred_error)

### same results, no difference for abs vs ()^2 of sum of errors.
sum((np.array(wage_change)[0] - (lmb_bar[0]*0 + (1-lmb_bar[0])*0))**2)

#### Estimator inconsistent because there is endogenous regressor?
#### Show that my error term and the regressor are correlated.
np.mean(df_exog["lmb_bar"] * model_1.resid)
# mean 0!

np.corrcoef(df_exog["lmb_bar"], model_1.resid)
# correlation = 0 !

# plot the error term against the regressor
plt.scatter(df_exog["lmb_bar"], model_1.resid)



### for testing: consider data where lambda in t=0 and t=1 is equal.
### (no task adjustments, lambda is not endogenous)
from mc_skills import draw_skills

sim_skills = draw_skills(n=100, J=2, seed=seed)
skills_1, skills_2 = np.empty([100]), np.empty([100])
for i in range(0, 100):
    skills_1[i] = sim_skills[i, 0]
    skills_2[i] = sim_skills[i, 1]

df_exog["lmb_prev"]
wage_0 = np.multiply(np.array(df_exog["lmb_prev"]), (skills_1 + pi[0, 0])) + \
    np.multiply(np.array(1-df_exog["lmb_prev"]), (skills_2 + pi[1, 0]))

wage_1 = np.multiply(np.array(df_exog["lmb_prev"]), (skills_1 + pi[0, 1])) + \
    np.multiply(np.array(1-df_exog["lmb_prev"]), (skills_2 + pi[1, 1]))

wage_diff = wage_1 - wage_0

plt.plot(df_exog["lmb_prev"], wage_diff)
sms.OLS(endog=wage_diff, exog=sms.add_constant(df_exog["lmb_prev"])).fit().summary()
### And we see, that here we can estimate the changes in pi very accurately.
