#### plot optimal lambda, which is piecewise given by the p-q-formula, as it is
#### defined to be in (0,1). This script plots the optimal lambdas for an array
#### of values for theta - which is equal to (pi_1 + s_1) - (pi_2 + s_2).
import numpy as np
import matplotlib.pyplot as plt



# define lambda 1 and 2 according to p-q-formula
def lam_1(theta):
    return -1*(2-theta)/(2*theta) + np.sqrt(((2-theta)/(2*theta))**2+(1/theta))


def lam_2(theta):
    return -1*(2-theta)/(2*theta) - np.sqrt(((2-theta)/(2*theta))**2+(1/theta))


# plot optimal lambda 1 and 2 for a range of theta values
theta = list(range(-20, 21))
l1, l2 = np.empty([len(theta), 1]), np.empty([len(theta), 1])
for i in range(0, len(theta)):
    if theta[i] != 0:
        l1[i] = lam_1(theta[i])
        l2[i] = lam_2(theta[i])
    else:
        l1[i], l2[i] = 0.5, 0.5

#print(theta, "\n", l1, "\n", l2)
plt.plot(theta, l1, color='b')
plt.plot(theta, l2, color='g')
plt.axhline(y=0.0, color='r', linestyle='-')
plt.axhline(y=1.0, color='r', linestyle='-')
plt.show()



#### in this section: Try to approximate the change in wages using a taylor
#### series (polynomial degree 3).

# define optimally chosen lambda using above insight.


############################################################################
############################################################################
### Test the identification i derived for Dw
# for now: constant skills -> all Ds1, Ds2 = 0

import pandas as pd
import numpy as np
import statsmodels.regression.linear_model as sms
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt

# import functions from other modules
from mc_optimal_wage_choice import mc_optimal_wage_choice

# Simulate data
seed=600
mc_data = mc_optimal_wage_choice(n=100, T=2, J=2, seed=seed)

# estimate price changes frmo t=1 perspective
# calculate approximated lmb_bar (for now: for t=0 und t=1)
idx = pd.IndexSlice
lmb_bar = (mc_data.loc[idx[:, "lambda"], 0] + mc_data.loc[idx[:, "lambda"], 1])/2

# get true wage difference
wage_change = mc_data.loc[idx[:, "wage"], 1] - mc_data.loc[idx[:, "wage"], 0]

# get Prices
from mc_prices import draw_skill_prices
pi = draw_skill_prices(T=2, J=2, seed=seed)

# get skills
from mc_skills import draw_skills
skills = draw_skills(J=2, seed=seed, n=100)

# calculate first summand: Ds2 + Dpi2
summand1 = pi[1,1]-pi[1,0]

# calculate second summand: 1/2(lmb1 + lmb0)*(Ds1 + Dpi1 - Ds2 - Dpi2)
lmb0 = mc_data.loc[idx[:, "lambda"], 0]
lmb1 = mc_data.loc[idx[:, "lambda"], 1]
summand2 = 0.5*(lmb0+lmb1)*((skills[:, 1] - skills[:, 0]) + (pi[0, 1] - pi[0, 0]) - (skills[:, 1] - skills[:, 0]) - (pi[1, 1] - pi[1, 0]))

# calculate third summand: (1/2)(lmb1 - lmb0)
summand3 = np.array(0.5*(lmb1-lmb0))*(skills[:,0] + pi[0,0] - skills[:,1] - pi[1,0] + skills[:,0] + pi[0,1] - skills[:,1] - pi[1,1])
calculated_wage_change = summand1 + np.array(summand2) + np.array(summand3)

calculated_wage_change - np.array(wage_change)

plt.scatter(calculated_wage_change, wage_change)
plt.plot([-0.2, 0.2],[-0.2, 0.2], color="r")

### analyse why the approximation error is resulting in r^2 =1
# is approximation error (summand3) linear in lmb_bar?
plt.scatter(lmb_bar, summand3)



df = pd.DataFrame(data={"const": np.ones(100).astype(float), "lmb_bar": np.asarray(lmb_bar).astype(float), "endog": np.asarray(summand3).astype(float)})
sms.OLS(endog=df[["endog"]], exog=df[["const", "lmb_bar"]]).fit().summary()

### error term has functional form of -0.5*(Dw1-Dw2) + lmb_bar(Dw1-Dw2)
# But: how can error term be a function of lmb_bar, ergo my regressor?

calculated_wage_change
wage_change


##### take the approximated wage change ignoring the error from dropping
##### the 3rd summand above:
wage_change_approx = lmb_bar * (pi[0, 1] - pi[0, 0]) + (1-lmb_bar) * (pi[1, 1] - pi[1, 0])

# plot true wage change against incorrectly approximated change, the
# approximation error and the corrected approximated change
plt.plot(lmb_bar, wage_change_approx, color='r')
plt.plot(lmb_bar, wage_change, color='blue', linewidth=3)
plt.plot(lmb_bar, summand3, color='y')
plt.plot(lmb_bar, summand3+wage_change_approx, color='g', linewidth=2)
#plt.plot(lmb_bar, np.zeros(100), color='black', alpha=0.5)




#####################################################
###########################################
###########################################
# substract approx error from actual "true" wage changes
y = (wage_change - summand3).astype(float)

# regress this difference on lmb_bar
sms.OLS(endog=y, exog=sms.add_constant(np.array(lmb_bar)).astype(float)).fit().summary()


x=[0.5, 0.5]
np.random.uniform(low=x[0], high=x[1], size=10)
