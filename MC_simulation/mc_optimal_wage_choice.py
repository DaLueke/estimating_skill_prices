def mc_optimal_wage_choice():
    '''
    This script draws the simulated skills and skill prices around
    returns the optimally chosen worktime shares, as well as resulting wages
    and utility.

    Input:

    Output:

    Assumptions:

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
    from scipy.optimize import minimize

    # import other modules
    from mc_skills import draw_skills
    from mc_prices import draw_skill_prices

    # Set seed.
    seed = 555
    np.random.seed(seed)

    # Set parameters
    n = 20

    # Draw simulated skills and prices
    sim_skills = draw_skills(n=n, seed=seed)
    sim_prices = draw_skill_prices()

    # Define individual wage as a function of worktime shares
    def wage(lmb, i):
        return lmb*(sim_skills[i, :] + sim_prices[:, i])[0] \
            + (1-lmb)*(sim_skills[i, :] + sim_prices[:, i])[1]

    # Define utility as a function of wage and worktime shares
    # Note: defined negative to use minimizing function
    def utility(lmb, i):
        return -1*(wage(lmb, i) + np.log(lmb) + np.log(1-lmb))

    # Find optimal worktime shares lmb, according wages and untility
    lmb_opt, util_opt, wage_opt = np.empty([3, n])
    for i in range(n):
        opt = minimize(
            fun=utility,
            x0=0.5,
            args=i,
            method='SLSQP',
            bounds=[(0, 1)],
            options={"maxiter": 15}
            )
        lmb_opt[i] = opt["x"]
        util_opt[i] = (-1)*opt["fun"]
        wage_opt[i] = wage(lmb_opt[i], n)

    return lmb_opt, util_opt, wage_opt
