

def draw_simulation_data(
    T,
    N,
    J,
    penalty,
    p_weight,
    p_locus,
    p_exponent,
    store_data=False,
    **kwargs
):
    """ This function draws simulation data.

    Arguments:
        T           (int) # of periods
        N           (int) # of observations
        J           (int) # of tasks
        penalty     (str) functional form of the penalty term (see
                    mc_optimal_wage_choice for detail)
        p_weight    (int) weight of the penalty term
        p_locus     (tuple) locus of the penalty term
        p_exponent  (int) exponent of the penalty term
        store_data  (bool) optional, can be used to store simulated data in
                    a pickle file in "OUT\\simulation_data.p". Default: False.

        Returns:    Hierarchical DataFrame with simulated data. First level
                    are periods, second level are individuals.
    """
    # import needed packages and functions
    import pandas as pd
    import numpy as np
    import pickle
    import sys
    import os

    # Make this function callable
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

    from DGP.mc_optimal_wage_choice import mc_optimal_wage_choice

    # simulate data:
    mc_data, skills, prices = mc_optimal_wage_choice(
        N=N,
        T=T, J=2,
        penalty='quad',
        p_weight=35,
        p_locus=(0.3, 0.7),
        p_exponent=2,
        **kwargs
        )

    # Define Multiindex DataFrame
    idx = pd.IndexSlice
    index = pd.MultiIndex.from_product(
        [list(range(T)), list(range(N))],
        names=['year', 'individual']
        )

    # write simulation data into DataFrame
    df = pd.DataFrame(index=index)
    for t in range(T):
        df.loc[idx[t, :], "lambda"] = mc_data.loc[idx[:, 'lambda'], t].values
        df.loc[idx[t, :], "utility"] = mc_data.loc[idx[:, 'utility'], t].values
        df.loc[idx[t, :], "wage"] = mc_data.loc[idx[:, 'wage'], t].values
        df.loc[idx[t, :], "skills_1"] = skills[t][:, 0]
        df.loc[idx[t, :], "skills_2"] = skills[t][:, 1]
        df.loc[idx[t, :], "prices_1"] = np.repeat(prices[0, t], N)
        df.loc[idx[t, :], "prices_2"] = np.repeat(prices[1, t], N)

    # Optional: Store simulated data in pickle file.
    if store_data:
        pickle.dump(df, open('..\\OUT\\simulation_data.p', 'wb'))
    else:
        pass

    return df
