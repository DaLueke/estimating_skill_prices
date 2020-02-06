

def draw_skill_prices(
    T, J, seed,
    pi_fun='pi_fixed',
    low=-0.2,
    high=0.2,
    const=[0.05, 0.1]
):
    """ Draws initial skill prices and simulates random prices changes.
    Agruments:
        T              (int) Number of periods
        J              (int) Number of tasks
        seed           (int) Seed for random draw of prices
        pi_fun         (str) defines the process of wage changes. Currently
                       implemented options:
                       - pi_normal: Draws from standard normal distribution.
                       - pi_uniform: Draws uniform distribution. Borders
                         are defined in "low" and "high" arguments.
                       - pi_fixed: Non-random, constant price changes.
                         Changes can be provided in "const" argument.
        low            (int) Lower border of uniform distributed price changes.
        high           (int) Upper border of uniform distributed price changes.
        const          (list) Changes for pi_fixed option.

    Returns:
        pi1, pi2       JxT array of prices for tasks 1 and 2.

    Assumptions:
        (1) Initial Skill Prices
    """
    # import packages
    import numpy as np

    # set seed
    np.random.seed(seed)

    # define functions that return price changes for different specifications
    # (1) Draw stadard normal distributed changes in log prices.
    def pi_normal(J=J, T=T, **kwargs):
        pi_normal = np.around(np.random.normal(size=(J, T-1)), 4)
        return pi_normal

    # (2) Draw changes in log prices that are uniformly distributed over
    # some interval.
    def pi_uniform(J=2, T=T, **kwargs):
        low, high = kwargs['low'], kwargs['high']
        pi_uniform = np.around(
            np.random.uniform(low, high, size=(J, T-1)),
            4
            )
        return pi_uniform

    # (3) Fix changes in log prices.
    def pi_fixed(J=J, T=T, **kwargs):
        const = kwargs['const']
        pi_fixed = np.array([const, ]*T).transpose()
        return pi_fixed

    # Set initial task prices
    # Assume task 1 (social) has a lower price than 2 (non-social)
    pi1_0 = 5
    pi2_0 = 6

    # Define price array
    pi = np.empty([J, T])
    pi[:, 0] = pi1_0, pi2_0

    # Get price changes.
    # Find price changes function of choice:
    price_changes = eval(pi_fun)
    d_pi = price_changes(T=T, J=J, low=low, high=high, const=const)

    # Calculate prices in each period
    for t in range(1, T):
        pi[:, t] = pi[:, t-1] + d_pi[:, t-1]

    return pi
