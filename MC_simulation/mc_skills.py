def draw_skills(n, J, seed):
    """ This script draws two correlated normal distributed skills and
    calculates their percentiles.
    Arguments:
        n           Number of individuals to simulate
        J           Number of tasks
        seed        Seed for the draws from skill distributions

    Returns:
        skills      n x J array of skills

    Assumptions:
        (1) Skills' distribution
            (1a) Skills' variances: Skills are positively correlated
            (1b) Skills' means
    """
    # Import packages
    import numpy as np

    # Set parameter
    np.random.seed(seed)

    # For now, assume that skills are positively correlated
    skills_var = np.array([[3, 0.3], [0.3, 3]])

    # I standardize skills to have mean 0
    skills_mean = np.array([5, 5])

    # Draw skills for n observations
    skills = np.random.multivariate_normal(
        mean=skills_mean,
        cov=skills_var,
        size=(n)
        )

    ''' ----- This part converts skills into skill-percentiles
    # Calculate skill percentiles
    # Problem with percentiles: If the skill-percentile of a worker impacts her
    # wage, skill accumulation can only impact wage, for a growth in skills
    # in comparison to all other workers. Skill growth would in this case
    # always be an improvement in comparison to all other workers.
    def calc_percentile(array):
        """ This function calculates the percentile of each value of the
        inputarray.
        Input:
            array - input array with values that will be mapped
                    into percentiles.
        Output:
            percentiles -   array that contains the percentiles for each value
                            of the
            input array.
        """
        # Get the number of columns in the array
        n, J = np.shape(array)
        percentiles = np.empty([n, J])
        for k in range(J):
            for i in range(n):
                percentiles[i, k] = sum(array[:, k] < array[i, k]) / n
        return percentiles
    '''
    return skills
