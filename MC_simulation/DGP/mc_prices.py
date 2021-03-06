

def draw_skill_prices(
    T,
    J,
    pi_fun='pi_fixed',
    low=-0.2,
    high=0.2,
    const=[0.0, 0.05]
):
    """ Draws initial skill prices and simulates random prices changes.
    With the normalization of wages in task 1 to zero, some parts
    of this function are redundent. However, the way this function is currently
    set up allows for a simulation without this normalization, too.

    Arguments:
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
        (1) Initial relative skill price for task 2 is +5%
        (2) No price changes in a base period (t=0 to t=1)
    """
    # import packages
    import numpy as np

    # # set seed
    # np.random.seed(seed)

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
    # Assume task 1 has a lower price than task 2
    pi1_0 = 0
    pi2_0 = 0.1

    # Define price array
    pi = np.empty([J, T])

    # Set intial prices
    pi[:, 0] = pi1_0, pi2_0
    pi[:, 1] = pi1_0, pi2_0

    # Get price changes.
    # Find price changes function of choice:
    price_changes = eval(pi_fun)
    d_pi = price_changes(T=T, J=J, low=low, high=high, const=const)

    # Calculate prices in each period, while there is no price change in a base
    # period (from t=0 to t=1)
    for t in range(2, T):
        pi[:, t] = pi[:, t-1] + d_pi[:, t-1]

    return pi
