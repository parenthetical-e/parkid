import numpy as np


def R_homeostasis(reward, total_reward, set_point):
    """Update reward value assuming homeostatic value.
    
    Value based on Keramati and Gutkin, 2014.
    https://elifesciences.org/articles/04811
    """
    deviance_last = np.abs(set_point - total_reward)
    deviance = np.abs(set_point - (total_reward + reward))
    reward_value = deviance_last - deviance
    return reward_value