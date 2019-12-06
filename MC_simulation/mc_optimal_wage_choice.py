def mc_optimal_wage_choice(n, T, J=2, seed=555):
    '''
    This script draws the simulated skills and skill prices around
    returns the optimally chosen worktime shares, as well as resulting wages
    and utility. This is done for all time periods.
    Results are returned as multiindex-dataframe.

    Arguments:
        seed            needed for skill and price simulations
        n               number of individuals to simulate
        T               time periods to simulate

    Returns:
        Hierachical dataframe with individuals as first level index and
        parameters on second level. One column per period.

    Assumptions:
        Functional form of wage and utility follows the arguments in my Thesis.
        Particularly: Penalty term is sum of logs of worktime shares.

    Thoughts:
    Currently, all wages will be very similar as prices are the same for all i
    and vary only in [0,1]. In order to get a more expectable variance in
    wages, skills have to have higher variance! (Skills are the source of
    differences in incomes in this model!)
    Plus: If skills have a greater impact on wages, then the impact of changes
    in prices on lmb will decrease - as skills are timeinvariant for now.
    '''
    # import packages
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    # import other modules
    from mc_skills import draw_skills
    from mc_prices import draw_skill_prices

    # Draw simulated skills and prices
    sim_skills = draw_skills(n=n, J=J, seed=seed)
    sim_prices = draw_skill_prices(T=T, J=J, seed=seed)

    # Define individual wage as a function of worktime shares
    def wage(lmb, i, t):
        return lmb*(sim_skills[i, :] + sim_prices[:, t])[0] \
            + (1-lmb)*(sim_skills[i, :] + sim_prices[:, t])[1]

    # Define utility as a function of wage and worktime shares
    # Note: defined negative to use minimizing function
    def utility(lmb, i, t):
        return -1*(wage(lmb, i, t) + np.log(lmb) + np.log(1-lmb))

    # Find optimal worktime shares lmb, according wages and untility
    lmb_opt = np.empty([T, n])
    util_opt = np.empty([T, n])
    wage_opt = np.empty([T, n])
    for i in range(n):
        for t in range(T):
            opt = minimize(
                fun=utility,
                x0=0.5,
                args=(i, t),
                method='SLSQP',
                bounds=[(0.01, 0.99)],
                options={"maxiter": 15}
                )
            lmb_opt[t, i] = opt["x"]
            util_opt[t, i] = (-1)*opt["fun"]
            wage_opt[t, i] = wage(lmb_opt[t, i], i, t)

    # Initialize DataFrame with simulated data
    # define multiindex
    index = pd.MultiIndex(
                     levels=[list(range(n)), ["lambda", "wage", "utility"]],
                     codes=[list(np.repeat(range(n), 3)), [0, 1, 2]*n]
                     )

    # write simulated data into multiindexed dataframe
    mc_data = pd.DataFrame(index=index, columns=list(range(T)))
    idx = pd.IndexSlice

    for t in range(T):
        mc_data.loc[idx[:, "lambda"], t] = lmb_opt[t]
        mc_data.loc[idx[:, "wage"], t] = wage_opt[t]
        mc_data.loc[idx[:, "utility"], t] = util_opt[t]

    return mc_data
