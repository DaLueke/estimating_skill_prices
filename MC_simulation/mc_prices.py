def draw_skill_prices(T=25, J=2, seed=555):
    """ Draws initial skill prices and simulates random prices changes.
    Agruments:
        T:              Number of periods

    Returns:
        pi1, pi2:       Prices for tasks 1 and 2

    Assumptions:
        (1) Initial Skill Prices
        (2) Chages in Skill Prices are normal distributed
    """
    # import packages
    import numpy as np

    # Set seed.
    np.random.seed(seed)

    # Define time horizon, assume we can observe 25 years
    T = T

    # Define number of Tasks
    J = J

    # Set initial task prices
    # Assume task 1 (social) has a lower price than 2 (non-social)
    pi1_0 = 6.5
    pi2_0 = 7

    # Draw price changes
    d_pi = np.around(np.random.normal(size=(J, T-1)), 2)

    # Define price array
    pi = np.empty([J, T])
    pi[:, 0] = pi1_0, pi2_0

    # Calculate prices in each period
    for t in range(1, T):
        pi[:, t] = pi[:, t-1] + d_pi[:, t-1]

    return pi
