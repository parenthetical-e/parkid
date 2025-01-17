import numpy as np

from scipy.special import softmax
from collections import OrderedDict


def R_update(state, R, critic, lr):
    """Delta update"""
    update = lr * (R - critic(state))
    critic.update(state, update)

    return critic


def E_update(state, E, critic, lr):
    """Bellman update"""
    update = lr * E
    critic.replace(state, update)

    return critic


class CountMemory:
    """A simple state counter."""
    def __init__(self):
        self.memory = dict()

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        # Init?
        if state not in self.memory:
            self.memory[state] = 0

        # Update count in memory
        # and then return it
        self.memory[state] += 1

        return self.memory[state]

    def state_dict(self):
        return self.memory

    def load_state_dict(self, state_dict):
        self.memory = state_dict


class WSLS:
    """Win-stay lose-switch policy control"""
    def __init__(self, actor_E, critic_E, actor_R, critic_R, boredom=0.0):
        self.actor_R = actor_R
        self.critic_R = critic_R
        self.actor_E = actor_E
        self.critic_E = critic_E
        self.boredom = boredom

    def __call__(self, E, R):
        return self.forward(E, R)

    def update(self, action, E, R, lr_R):
        if R is not None:
            self.critic_R = R_update(action, R, self.critic_R, lr_R)
        if E is not None:
            self.critic_E = E_update(action, E, self.critic_E, lr=1)

    def forward(self, E, R):
        if (E - self.boredom) > R:
            critic = self.critic_E
            actor = self.actor_E
            policy = 0
        else:
            critic = self.critic_R
            actor = self.actor_R
            policy = 1

        return actor, critic, policy


class WSLSh(WSLS):
    """Win-stay lose-switch policy control, for homeostatic rewards"""
    def __init__(self, actor_E, critic_E, actor_R, critic_R, boredom=0.0):
        super().__init__(actor_E, critic_E, actor_R, critic_R, boredom=boredom)

    def forward(self, E, R):
        if (E - self.boredom) >= R:
            critic = self.critic_E
            actor = self.actor_E
            policy = 0
        else:
            critic = self.critic_R
            actor = self.actor_R
            policy = 1

        return actor, critic, policy


class Critic:
    """A tabular critic"""
    def __init__(self, num_inputs, default_value):
        self.num_inputs = num_inputs
        self.default_value = default_value

        self.model = OrderedDict()
        for n in range(self.num_inputs):
            self.model[n] = self.default_value

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        return self.model[state]

    def update(self, state, update):
        self.model[state] += update

    def replace(self, state, update):
        self.model[state] = update

    def state_dict(self):
        return self.model

    def load_state_dict(self, state_dict):
        self.model = state_dict


class NoisyCritic:
    def __init__(self,
                 num_inputs,
                 default_value,
                 default_noise_scale=0,
                 seed_value=None):

        self.num_inputs = num_inputs
        self.default_value = default_value
        self.default_noise_scale = default_noise_scale
        self.prng = np.random.RandomState(seed_value)

        # Init
        self.model = OrderedDict()
        self.inital_values = OrderedDict()
        for n in range(self.num_inputs):
            # Def E0. Add noise? None by default.
            delta = 0.0
            if self.default_noise_scale > 0:
                delta = self.prng.normal(0,
                                         scale=default_value *
                                         self.default_noise_scale)

            # Set E0
            self.inital_values[n] = self.default_value + delta
            self.model[n] = self.inital_values[n]

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        return self.model[state]

    def update(self, state, update):
        self.model[state] = update

    def state_dict(self):
        return self.model


class RandomActor(object):
    def __init__(self, num_actions, boredom=0.0, seed_value=None):
        self.prng = np.random.RandomState(seed_value)
        self.boredom = boredom
        self.num_actions = num_actions
        self.actions = list(range(self.num_actions))
        # Undef for softmax. Set to False: API consistency.
        self.tied = False

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        """Values are a dummy var. Pick at random"""

        # Threshold
        values = np.asarray(values) - self.boredom

        # Move the the default policy?
        if np.sum(values < 0) == len(values):
            return None

        # Make a random choice
        action = self.prng.choice(self.actions)
        return action


class DeterministicActor(object):
    def __init__(self, num_actions, tie_break='next', boredom=0.0):
        self.num_actions = num_actions
        self.tie_break = tie_break
        self.boredom = boredom
        self.action_count = 0
        self.tied = False

    def _is_tied(self, values):
        # One element can't be a tie
        if len(values) < 1:
            return False

        # Apply the threshold, rectifying values less than 0
        t_values = [max(0, v - self.boredom) for v in values]

        # Check for any difference, if there's a difference then
        # there can be no tie.
        tied = True  # Assume tie
        v0 = t_values[0]
        for v in t_values[1:]:
            if np.isclose(v0, v):
                continue
            else:
                tied = False

        return tied

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        # Pick the best as the base case, ....
        action = np.argmax(values)

        # then check for ties.
        #
        # Using the first element is argmax's tie breaking strategy
        if self.tie_break == 'first':
            pass
        # Round robin through the options for each new tie.
        elif self.tie_break == 'next':
            self.tied = self._is_tied(values)
            if self.tied:
                self.action_count += 1
                action = self.action_count % self.num_actions
        else:
            raise ValueError("tie_break must be 'first' or 'next'")

        return action


class ThresholdActor(object):
    def __init__(self, num_actions, boredom=0.0, seed_value=None):
        self.prng = np.random.RandomState(seed_value)
        self.boredom = boredom
        self.num_actions = num_actions
        self.actions = list(range(self.num_actions))
        self.action_count = 0
        self.tied = False

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        # Threshold
        values = np.asarray(values) - self.boredom

        # Pick from postive values, while there still are positive values.
        mask = values > 0
        if np.sum(mask) > 0:
            filtered = [a for (a, m) in zip(self.actions, mask) if m]
            action = self.prng.choice(filtered)
        else:
            self.tied = True
            action = None

        return action


class EpsilonActor:
    def __init__(self,
                 num_actions,
                 epsilon=0.1,
                 decay_tau=0.001,
                 seed_value=42):
        self.epsilon = epsilon
        self.decay_tau = decay_tau
        self.num_actions = num_actions
        self.seed_value = seed_value
        self.prng = np.random.RandomState(self.seed_value)

    def __call__(self, values):
        return self.forward(values)

    def decay_epsilon(self):
        self.epsilon -= (self.decay_tau * self.epsilon)

    def forward(self, values):
        # If values are zero, be random.
        if np.isclose(np.sum(values), 0):
            action = self.prng.randint(0, self.num_actions, size=1)[0]

            return action

        # Otherwise, do Ep greedy
        if self.prng.rand() < self.epsilon:
            action = self.prng.randint(0, self.num_actions, size=1)[0]
        else:
            action = np.argmax(values)

        return action


class SoftmaxActor:
    def __init__(self, num_actions, temp=1, seed_value=42):
        self.temp = temp
        self.inv_temp = 1 / self.temp

        self.num_actions = num_actions
        self.seed_value = seed_value
        self.prng = np.random.RandomState(self.seed_value)
        self.actions = list(range(self.num_actions))

    def __call__(self, values):
        return self.forward(values)

    def forward(self, values):
        values = np.asarray(values)
        probs = softmax(values * self.inv_temp)
        action = self.prng.choice(self.actions, p=probs)

        return action
