#! /usr/bin/env python
import numpy as np
import gym

from copy import deepcopy
from gym import spaces
from gym.utils import seeding
from itertools import cycle

# Gym is annoying these days...
import warnings
warnings.filterwarnings("ignore")

from infomercial.local_gym.bandit import BanditEnv


class BanditUniform121(BanditEnv):
    """A 121 armed bandit."""
    def __init__(self, p_min=0.2, p_max=0.8, p_best=0.8, best=54):
        self.best = [best]
        self.num_arms = 121

        # ---
        self.p_min = p_min
        self.p_max = p_max
        self.p_best = p_best

        # Generate intial p_dist
        # (gets overwritten is seed())
        p_dist = np.random.uniform(self.p_min, self.p_max,
                                   size=self.num_arms).tolist()
        p_dist[self.best[0]] = self.p_best

        # reward
        r_dist = [1] * self.num_arms

        # ---
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Reset p(R) dist with the seed
        self.p_dist = self.np_random.uniform(self.p_min,
                                             self.p_max,
                                             size=self.num_arms).tolist()
        self.p_dist[self.best[0]] = self.p_best

        return [seed]


class BanditChange121(BanditUniform121):
    """Change the best choice in BanditUniform121 to the worst"""
    def __init__(self, p_min=0.2, p_max=0.8, p_best=0.8, org_best=54):
        super().__init__(p_min=p_min,
                         p_max=p_max,
                         p_best=p_best,
                         best=org_best)

        # Make the best the worst
        self.original_best = deepcopy(self.best)
        self.p_dist[self.original_best[0]] = 0.1

        # Now update the best
        self.best = [np.argmax(self.p_dist)]

    def seed(self, seed=None):
        super().seed(seed)
        self.original_best = deepcopy(self.best)
        self.p_dist[self.original_best[0]] = 0.1
        self.best = [np.argmax(self.p_dist)]

        return [seed]
