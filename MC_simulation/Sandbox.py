import pandas as pd
import numpy as np
import sys
import os

##### DEBUGGING #####
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + "\\DGP")
#####################

from DGP.draw_data import draw_simulation_data

kwargs = {
          # (1) Arguments for price changing process.
          "pi_fun": "pi_fixed",
          "const": [0.0, 0.05]
          }



np.random.seed(1)
sim_data = draw_simulation_data(
    T=3,
    N=100,
    J=2,
    penalty="quad",
    p_weight=35,
    p_locus=(0.5, 0.5),
    p_exponent=2,
    **kwargs
)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

print(sim_data)

np.random.seed(1)
from mc_optimal_wage_choice import mc_optimal_wage_choice
mc_data, skills, prices = mc_optimal_wage_choice(
    N=100,
    T=3, J=2,
    penalty='quad',
    p_weight=35,
    p_locus=(0.5, 0.5),
    p_exponent=2,
    **kwargs
    )
mc_data.loc[pd.IndexSlice[:, 'lambda'], 0].values





(skills[1][3, :] + prices[:, 2])[1]












test_0 = [np.array([1., 2.])]

test_1 = test_0[0].copy()
test_1[1] = test_1[1]+ test_1[1]*0.1
test_0 = np.append(test_0, [test_1], axis=0)



import pandas as pd
import numpy as np
import sys
import os

# import other modules
from DGP.mc_prices import draw_skill_prices
from DGP.mc_skills import draw_initial_skills
from DGP.mc_skills import draw_acculumated_skills
sim_skills = draw_initial_skills(N=10, J=2)
skills = draw_acculumated_skills(
            skills=sim_skills[0],
            lmb=np.array(range(10))
            )
sim_skills = np.append(sim_skills, skills, axis=0)






import numpy as np
N=100

# For now, assume that skills are positively correlated
skills_var = np.array([[0, 0.], [0., 1]])



a = [np.array([1.0,2.0,3.0])]
b = [np.array([7.0,8.0,9.0])]
a = np.append(a, b, axis=0)
a[0]
c = a[1].copy()
c[1] = c[1] + 0.1
c
a = np.append(a, [c], axis=0)
a
# I standardize skills to have mean 5
skills_mean = np.array([0, 0])

skills = [np.round(np.random.multivariate_normal(
    mean=skills_mean,
    cov=skills_var,
    size=(N)
    ), 6)]
4*(skills[0][0,:] + 20.23)[0]

skills2 = skills[0][:, 1]  + (0.5 - 0.5) * 0.1

skills2 = np.append(skills, [np.random.multivariate_normal(
    mean=np.array([10, 10]),
    cov=skills_var,
    size=(N)
    )],
    axis=0)

a=1


(skills2[1][4, :] + [0, 1])[0]


np.array([0.4, 0.3, 0.7, 0.56]) - 0.5











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
from mc_skills import draw_skills
from mc_prices import draw_skill_prices


draw_skill_prices(J=2, T=2, seed=600, pi_fun="pi_fixed")

# Simulate data
seed=600
mc_data = mc_optimal_wage_choice(
    n=100,
    T=2, J=2,
    penalty='quad',
    p_weight=30,
    p_locus=[0.4, 0.6],
    p_exponent=3,
    seed=seed
    )

# calculate w2t - w1t to see if its always positive. Only then,
# the necessary condition for maximum, which involves the square root of this
# term, is defined.
sim_skills = draw_skills(n=100, J=2, seed=seed)
sim_prices = draw_skill_prices(T=2, J=2, seed=seed)

# count number of individuals where wage task 1 - wage task 0 <0
sum((sim_skills[:, :] + sim_prices[:, 0])[:,0] - (sim_skills[:, :] + sim_prices[:, 0])[:,1] <0)


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
# Checking the derivation of actual wage change.

# import modules and functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mc_optimal_wage_choice import mc_optimal_wage_choice
from mc_skills import draw_skills
from mc_prices import draw_skill_prices

# First: Draw realized wages as well as workers' skills and task prices.


# Define indexer and parameters
np.random.seed(600)     # set seed for Data Generating Process
N = 1000                 # number of observations
M = 50                  # number of MC interations
idx = pd.IndexSlice     # define Indexslice

# Define optional arguments of DGP.
kwargs = {
          # (1) Arguments for price changing process.
          "pi_fun": "pi_fixed",
          "const": [0.05, 0.1]
          }

# Draw optimal task choices and resulting wages
df, skills, prices = mc_optimal_wage_choice(
                            n=N,
                            T=2,
                            J=2,
                            penalty="quad",
                            p_weight=25,
                            p_locus=[0.2,0.8],
                            p_exponent=2,
                            **kwargs
                            )
df[0:100]
# Calculate wage changes in simlated data
sim_wage_change = (df.loc[idx[:, "wage"],1] - df.loc[idx[:, "wage"],0]).array

# Calculate wage changes analytically
# Draw price-changes and skills
s_tilde = skills[:,1] - skills[:,0]

tilde_pi = prices[1,:] - prices[0,:]

# Calculate \Delta \lambda as well as \bar{lambda}
D_lambda =  (df.loc[idx[:, "lambda"],1] - df.loc[idx[:, "lambda"],0]).array
Bar_lambda = ((df.loc[idx[:, "lambda"],1] + df.loc[idx[:, "lambda"],0]) /2).array

calc_wage_change = (prices[0,1] - prices[0,0]) \
                 + Bar_lambda*(tilde_pi[1]-tilde_pi[0]) \
                 + D_lambda * ((tilde_pi[1]+tilde_pi[0])/2 + s_tilde)

# Scatter the absolute difference between the wage changes
plt.scatter(sim_wage_change, calc_wage_change)
plt.plot([0.04,0.12], [0.04,0.12], color="r")
plt.xlabel("Analytically derived wage changes")
plt.ylabel("Simulated wage changes")
plt.savefig(fname="C:\\Users\\danie\\Desktop\\wage_change.png")

sim_wage_change - calc_wage_change
