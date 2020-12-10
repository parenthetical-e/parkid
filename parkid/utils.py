import numpy as np


def R_homeostasis(reward, total_reward, set_point, cost=0.0):
    """Update reward value assuming homeostatic value.
    
    Value loosely based on Keramati and Gutkin, 2014.
    https://elifesciences.org/articles/04811
    """
    if np.isclose(set_point, np.inf):
        return reward
    if np.isclose(set_point, 0.0):
        return reward
    if np.isclose(reward, 0.0):
        return 0.0

    deviance = set_point - total_reward
    h = (deviance + reward - cost) / set_point

    return np.clip(h, a_min=-+0.0, a_max=np.inf)
