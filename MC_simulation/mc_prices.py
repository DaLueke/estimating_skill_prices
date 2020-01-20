def draw_skill_prices(T, J, seed):
    """ Draws initial skill prices and simulates random prices changes.
    Agruments:
        T              Number of periods
        J              Number of tasks
        seed           Seed for random draw of prices

    Returns:
        pi1, pi2       Prices for tasks 1 and 2

    Assumptions:
        (1) Initial Skill Prices
        (2) Chages in Skill Prices are normal distributed
    """
    # import packages
    import numpy as np

    # set seed
    np.random.seed(seed)

    # Set initial task prices
    # Assume task 1 (social) has a lower price than 2 (non-social)
    pi1_0 = 5
    pi2_0 = 6

    # Draw stadard normal distributed changes in log prices
    d_pi_normal = np.around(np.random.normal(size=(J, T-1)), 4)

    # Draw changes in log prices that are uniformly distributed in [-0.2, 0.2]
    d_pi_uniform = np.around(
        np.random.uniform(low=-0.2, high=0.2, size=(J, T-1)),
        4
        )

    # Define price array
    pi = np.empty([J, T])
    pi[:, 0] = pi1_0, pi2_0

    # Calculate prices in each period
    for t in range(1, T):
        pi[:, t] = pi[:, t-1] + d_pi_uniform[:, t-1]

    return pi
