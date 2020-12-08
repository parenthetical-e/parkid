import numpy as np


def R_homeostasis(reward, total_reward, set_point):
    """Update reward value assuming homeostatic value.
    
    Value based on Keramati and Gutkin, 2014.
    https://elifesciences.org/articles/04811
    """
    deviance = (set_point - total_reward) / set_point
    return deviance + (reward / set_point)
