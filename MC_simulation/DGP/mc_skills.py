

def draw_initial_skills(
    N,
    J,
):
    """ This script draws initial skill endowments for J tasks and N
    individuals.
    For the simulations I normalize skills in task 1 to be zero. Therefore,
    parts of this function are redundant. However, this function
    is set up so that it works without that normalization, too.

    Arguments:
        N           (int) Number of individuals to simulate
        J           (int) Number of tasks

    Returns:
        skills      N x J array of skills

    """
    # Import packages
    import numpy as np

    # For now, assume that skills are not correlated
    skills_var = np.array([[0, 0.], [0., .25]])

    # I standardize skills to have mean 0
    skills_mean = np.array([0, 0])

    # Draw skills for n observations
    skills = [np.round(
        np.random.multivariate_normal(
            mean=skills_mean,
            cov=skills_var,
            size=(N)
            ), 6)
        ]

    return skills


def draw_acculumated_skills(skills, lmb):
    """ This function is used to simulate skill accumulation.

    Assumptions:
        - skill accumulation is modeled as a "learning by doing" process.
            -> worker get better in that task that they spend the majority of
            their working time on.
    """

    skills[:, 1] = skills[:, 1] + (lmb - 0.5)
    return [skills]
