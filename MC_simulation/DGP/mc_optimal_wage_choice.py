

def mc_optimal_wage_choice(
        N, T, J, penalty,
        p_weight=20,
        p_locus=[0.5, 0.5],
        p_exponent=2,
        **kwargs
):
    '''
    This script draws the simulated skills and skill prices and
    returns the optimally chosen worktime shares, as well as resulting wages
    and utility. This is done for all time periods.
    Results are returned as multiindex-dataframe.

    All calculations are performed in terms of lambda_2, the share spent on
    the second task.

    Arguments:
        seed            needed for skill and price simulations
        n               number of individuals to simulate
        T               time periods to simulate
        J               number of tasks to choose from
        penalty         functional form of the penalty term, currently supports
                        options 'quad' for a quadratic form and 'log' for a
                        logistic term.
        p_weight        (optional) gives the weight of the quadratic penalty
                        term. Defalt is 20.
        p_locus         (optional) gives borders of randomly (uniform) drawn
                        baseline decision of worktime shares. (i.e. the
                        minimum of the penalty term.)
        p_exponent      (optional) gives the exponent of the penalty term.

    Returns:
        Hierarchical dataframe with individuals as first level index and
        parameters on second level. One column per period.

    Assumptions:
        Functional form of wage and utility follows the arguments in my Thesis.
        Particularly: Penalty term is sum of logs of worktime shares.

    Thoughts:
    Skills are the source of differences in incomes in this model, as all
    individuals face the same prices.
    Plus: If skills have a greater impact on wages, then the impact of changes
    in prices on lmb will decrease - as skills are timeinvariant for now.
    '''
    # import packages
    from scipy.optimize import minimize
    import pandas as pd
    import numpy as np
    import sys
    import os

    # import other modules
    from DGP.mc_skills import draw_skills
    from DGP.mc_prices import draw_skill_prices

    # Make this function callable
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

    # Draw simulated skills and prices
    sim_skills = draw_skills(N=N, J=J, T=T)
    sim_prices = draw_skill_prices(T=T, J=J, **kwargs)

    # Define individual wage as a function of worktime shares including a
    # standard normal dist. errorterm
    def wage(lmb, i, t):
        return (1-lmb)*(sim_skills[t, i, :] + sim_prices[:, t])[0] \
            + (lmb)*(sim_skills[t, i, :] + sim_prices[:, t])[1]

    # Define utility as a function of wage and worktime shares
    # Note: defined negative to use minimizing function
    # TODO: Does the current version actually work for log_utility?
    #       Does the locus-arg result in error?
    def utility_log(lmb, i, t):
        return -1*(wage(lmb, i, t) + np.log(lmb) + np.log(1-lmb))

    # Define an alternative utility function that is quadratic in lmb
    def utility_quad(lmb, i, t, locus):
        return -1*(wage(lmb, i, t) - p_weight*(abs(locus[i]-lmb))**p_exponent)

    # What penalty term should be used is provided by argument "penalty"
    utility_function = eval('utility_' + penalty)

    # In case of exponential penalty: find randomized values for the center of
    # the penalty term.
    locus = np.random.uniform(low=p_locus[0], high=p_locus[1], size=N)

    # Find optimal worktime shares lmb, according wages and untility
    lmb_opt = np.empty([T, N])
    util_opt = np.empty([T, N])
    wage_opt = np.empty([T, N])
    for i in range(N):
        for t in range(T):
            opt = minimize(
                fun=utility_function,
                x0=0.5,
                args=(i, t, locus),
                method='SLSQP',
                bounds=[(0.0, 1.0)],
                options={"maxiter": 15}
                )
            lmb_opt[t, i] = np.round(opt["x"], 6)
            util_opt[t, i] = (-1)*opt["fun"]
            wage_opt[t, i] = wage(lmb_opt[t, i], i, t)

    # Initialize DataFrame with simulated data
    # define multiindex
    index = pd.MultiIndex(
                     levels=[list(range(N)), ["lambda", "wage", "utility"]],
                     codes=[list(np.repeat(range(N), 3)), [0, 1, 2]*N]
                     )

    # write simulated data into multiindexed dataframe
    mc_data = pd.DataFrame(index=index, columns=list(range(T)))
    idx = pd.IndexSlice

    for t in range(T):
        mc_data.loc[idx[:, "lambda"], t] = lmb_opt[t, :]
        mc_data.loc[idx[:, "wage"], t] = wage_opt[t, :]
        mc_data.loc[idx[:, "utility"], t] = util_opt[t, :]

    return mc_data, sim_skills, sim_prices
