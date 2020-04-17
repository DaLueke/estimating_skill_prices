

def draw_initial_skills(
    N,
    J,
):
    """ This script draws skill endowments for J tasks and n individuals.
    For now skill endowments are assumed to be time-invariant (absence of skill
    accumulation).

    Arguments:
        n           (int) Number of individuals to simulate
        J           (int) Number of tasks

    Returns:
        skills      n x J array of skills

    Assumptions:
        (1) Skills' distribution
            (1a) Skills' variances: Skills are positively correlated
            (1b) Skills' means
    """
    # Import packages
    import numpy as np

    # # Set parameter
    # np.random.seed(seed)

    # For now, assume that skills are not correlated
    skills_var = np.array([[0, 0.], [0., 1]])

    # I standardize skills to have mean 0
    skills_mean = np.array([0, 0])

    # Draw skills for n observations
    skills = [np.round(np.random.multivariate_normal(
        mean=skills_mean,
        cov=skills_var,
        size=(N)
        ), 6)]

    # In a specification without changes in skills:
    # Repeat same skills array for each of T periods.

    return skills


def draw_acculumated_skills(skills, lmb):
    skills[:, 1] = skills[:, 1] + (lmb - 0.5)
    # skills[:, 1] = skills[:, 1] + 10
    return [skills]
