# import packages
import numpy as np

# import other modules
from mc_skills import draw_skills
from mc_wages import draw_skill_prices

# Set seed.
seed = 555
np.random.seed(seed)

sim_skills = draw_skills(n=20, seed=seed)
sim_prices = draw_skill_prices()


print(sim_skills, "\n", sim_prices)
