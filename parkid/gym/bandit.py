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


class BanditEnv(gym.Env):
    """
    n-armed bandit environment  

    Params
    ------
    p_dist : list
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist : list or list or lists
        A list of either rewards (if number) or means and standard deviations (if list) of the payout that bandit has
    """
    def __init__(self, p_dist, r_dist):
        if len(p_dist) != len(r_dist):
            raise ValueError(
                "Probability and Reward distribution must be the same length")

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError(
                    "Standard deviation in rewards must all be greater than 0")

        self.p_dist = p_dist
        self.r_dist = r_dist

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)
        self.seed()

    def step(self, action):
        assert self.action_space.contains(action)
        state = 0
        reward = 0
        self.done = False

        if self.np_random.uniform() < self.p_dist[action]:
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]
            else:
                reward = self.np_random.normal(self.r_dist[action][0],
                                               self.r_dist[action][1])

        return state, reward, self.done, {}

    def reset(self):
        return [0]

    def render(self, mode='human', close=False):
        pass


class BanditUniform4(BanditEnv):
    """A 4 armed bandit."""
    def __init__(self, p_min=0.1, p_max=0.3, p_best=0.6, best=2):
        self.best = [best]
        self.num_arms = 4

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

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Reset p(R) dist with the seed
        self.p_dist = self.np_random.uniform(self.p_min,
                                             self.p_max,
                                             size=self.num_arms).tolist()
        self.p_dist[self.best[0]] = self.p_best

        return [seed]


class BanditChange4(BanditEnv):
    """Change the worst choice to the best BanditUniform4"""
    def __init__(self,
                 p_min=0.1,
                 p_max=0.3,
                 p_best=0.99,
                 p_org=0.6,
                 org_best=2):

        # Init
        self.num_arms = 4
        self.p_min = p_min
        self.p_max = p_max
        self.p_best = p_best
        self.orginal = BanditUniform4(p_min=p_min,
                                      p_max=p_max,
                                      p_best=p_org,
                                      best=org_best)

        # Build p_dist from org
        self.p_dist = deepcopy(self.orginal.p_dist)
        self.best = [np.argmin(self.p_dist)]
        self.p_dist[self.best[0]] = self.p_best

        # Build r_dist
        self.r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=self.p_dist, r_dist=self.r_dist)

    def seed(self, seed=None):
        # Set
        self.np_random, seed = seeding.np_random(seed)

        # Build p_dist from seed
        self.orginal.seed(seed)
        self.p_dist = deepcopy(self.orginal.p_dist)
        self.best = [np.argmin(self.p_dist)]
        self.p_dist[self.best[0]] = self.p_best

        return [seed]


class BanditStaticRegMonster(BanditEnv):
    """The 'static'** 4 armed bandit, based on Sumner [1]

    ** The static refers to Emily's naming scheme. It doesn't make as much sense
       here but we are all burdened by history are we not?
    
    [1] Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 0
        self.best = [best]

        # Generate static/intial p_dist
        p_dist = [0.6, 0.2, 0.3, 0.1]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditDynamicRegMonster(BanditEnv):
    """The 'dynamic'** 4 armed bandit, based on Sumner [1]
    
    ** The dynamic refers to Emily's naming scheme. It doesn't make as much
       sense here but we are all burdened by history are we not?

    [1] Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 3
        self.best = [best]

        # Generate the changed/dynamic p_dist
        p_dist = [0.6, 0.2, 0.3, 0.8]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditBigMonster1(BanditEnv):
    """A 4 armed bandit, a variation on Sumner [1], with arm probabilites:

    >>> p_dist = [0.4, 0.2, 0.3, 0.1]

    [1]: Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 0
        self.best = [best]

        # Generate static/intial p_dist
        p_dist = [0.4, 0.2, 0.3, 0.1]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditBigMonster2(BanditEnv):
    """A 4 armed bandit, a variation on Sumner [1], with arm probabilites:

    >>> p_dist = [0.4, 0.2, 0.3, 0.2]

    [1]: Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 0
        self.best = [best]

        # Generate static/intial p_dist
        p_dist = [0.4, 0.2, 0.3, 0.2]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditBigMonster3(BanditEnv):
    """A 4 armed bandit, a variation on Sumner [1], with arm probabilites:

    >>> p_dist = [0.4, 0.2, 0.3, 0.3]

    [1]: Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 0
        self.best = [best]

        # Generate static/intial p_dist
        p_dist = [0.4, 0.2, 0.3, 0.3]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditBigMonster4(BanditEnv):
    """A 4 armed bandit, a variation on Sumner [1], with arm probabilites:

    >>> p_dist = [0.4, 0.2, 0.3, 0.4]

    [1]: Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 0
        self.best = [best]

        # Generate static/intial p_dist
        p_dist = [0.4, 0.2, 0.3, 0.4]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditBigMonster5(BanditEnv):
    """A 4 armed bandit, a variation on Sumner [1], with arm probabilites:

    >>> p_dist = [0.4, 0.2, 0.3, 0.5]

    [1]: Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 0
        self.best = [best]

        # Generate static/intial p_dist
        p_dist = [0.4, 0.2, 0.3, 0.5]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditBigMonster6(BanditEnv):
    """A 4 armed bandit, a variation on Sumner [1], with arm probabilites:

    >>> p_dist = [0.4, 0.2, 0.3, 0.6]

    [1]: Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 3
        self.best = [best]

        # Generate static/intial p_dist
        p_dist = [0.4, 0.2, 0.3, 0.6]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditBigMonster7(BanditEnv):
    """A 4 armed bandit, a variation on Sumner [1], with arm probabilites:

    >>> p_dist = [0.4, 0.2, 0.3, 0.7]

    [1]: Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 3
        self.best = [best]

        # Generate static/intial p_dist
        p_dist = [0.4, 0.2, 0.3, 0.7]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditBigMonster8(BanditEnv):
    """A 4 armed bandit, a variation on Sumner [1], with arm probabilites:

    >>> p_dist = [0.4, 0.2, 0.3, 0.8]

    [1]: Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 3
        self.best = [best]

        # Generate static/intial p_dist
        p_dist = [0.4, 0.2, 0.3, 0.8]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditBigMonster9(BanditEnv):
    """A 4 armed bandit, a variation on Sumner [1], with arm probabilites:

    >>> p_dist = [0.4, 0.2, 0.3, 0.9]

    [1]: Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 3
        self.best = [best]

        # Generate static/intial p_dist
        p_dist = [0.4, 0.2, 0.3, 0.9]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditBigMonster10(BanditEnv):
    """A 4 armed bandit, a variation on Sumner [1], with arm probabilites:

    >>> p_dist = [0.4, 0.2, 0.3, 1.0]

    [1]: Sumner, E. S. et al. The Exploration Advantage: Children’s instinct to
    explore allows them to find information that adults miss. PsyArxiv h437v,
    11 (2019).
    """
    def __init__(self):
        self.num_arms = 4
        best = 3
        self.best = [best]

        # Generate static/intial p_dist
        p_dist = [0.4, 0.2, 0.3, 1.0]

        # reward (0, 1) values
        r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class BanditUniform121(BanditEnv):
    """A 121 armed bandit."""
    def __init__(self, p_min=0.1, p_max=0.3, p_best=0.6, best=54):
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

        # !
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Reset p(R) dist with the seed
        self.p_dist = self.np_random.uniform(self.p_min,
                                             self.p_max,
                                             size=self.num_arms).tolist()
        self.p_dist[self.best[0]] = self.p_best

        return [seed]


class BanditChange121(BanditEnv):
    """Change the worst choice to the best BanditUniform121"""
    def __init__(self,
                 p_min=0.1,
                 p_max=0.3,
                 p_best=0.99,
                 p_org=0.6,
                 org_best=54):

        # Init
        self.num_arms = 121
        self.p_min = p_min
        self.p_max = p_max
        self.p_best = p_best
        self.orginal = BanditUniform121(p_min=p_min,
                                        p_max=p_max,
                                        p_best=p_org,
                                        best=org_best)

        # Build p_dist from org
        self.p_dist = deepcopy(self.orginal.p_dist)
        self.best = [np.argmin(self.p_dist)]
        self.p_dist[self.best[0]] = self.p_best

        # Build r_dist
        self.r_dist = [1] * self.num_arms

        # !
        BanditEnv.__init__(self, p_dist=self.p_dist, r_dist=self.r_dist)

    def seed(self, seed=None):
        # Set
        self.np_random, seed = seeding.np_random(seed)

        # Build p_dist from seed
        self.orginal.seed(seed)
        self.p_dist = deepcopy(self.orginal.p_dist)
        self.best = [np.argmin(self.p_dist)]
        self.p_dist[self.best[0]] = self.p_best

        return [seed]
