import os
import fire
import gym
import numpy as np

from copy import deepcopy
from scipy.stats import entropy
from collections import OrderedDict
from noboard.csv import SummaryWriter

from infomercial.memory import DiscreteDistribution
from infomercial.distance import kl
from infomercial.utils import estimate_regret
from infomercial.utils import load_checkpoint
from infomercial.utils import save_checkpoint

from parkid.utils import R_homeostasis
from parkid.models import Critic
from parkid.models import DeterministicActor
from parkid.models import WSLS
from parkid.models import WSLSh
from parkid.gym.bandit import BanditUniform121
from parkid.gym.bandit import BanditChange121


def parkid(num_episodes=1000,
           change=100,
           tie_break='next',
           par_boredom=0.0,
           kid_boredom=0.0,
           share=0.0,
           set_point=None,
           lr_R=.1,
           master_seed=42,
           initial_bins=None,
           log_dir=None):
    """Parents and kids play a game of changing bandits"""

    # ------------------------------------------------------------------------
    # Sanity
    if change > num_episodes:
        raise ValueError("change must be less the num_episodes")
    # log
    log = SummaryWriter(log_dir=log_dir, write_to_disk=True)

    # ------------------------------------------------------------------------
    # Init tasks
    env1 = BanditUniform121()
    env2 = BanditChange121()
    env1.seed(master_seed)
    env2.seed(master_seed)
    env1.reset()
    env2.reset()
    num_actions = env1.action_space.n
    all_actions = list(range(num_actions))

    # Set the homeostatic R default
    if set_point is None:
        set_point = num_episodes

    # Init values
    R_0 = 0
    E_0 = entropy(np.ones(num_actions) / num_actions)
    par_E = E_0
    par_R = R_0
    kid_E = E_0
    kid_R = R_0

    # Init agents and memories
    # PAR
    par_wsls = WSLS(
        actor_E=DeterministicActor(num_actions,
                                   tie_break=tie_break,
                                   boredom=par_boredom),
        critic_E=Critic(num_actions, default_value=E_0),
        actor_R=DeterministicActor(num_actions,
                                   tie_break='first',
                                   boredom=par_boredom),
        critic_R=Critic(num_actions, default_value=R_0),
    )
    par_memories = [
        DiscreteDistribution(initial_bins=initial_bins)
        for _ in range(num_actions)
    ]
    # KID
    kid_wsls = WSLSh(
        actor_E=DeterministicActor(num_actions,
                                   tie_break=tie_break,
                                   boredom=kid_boredom),
        critic_E=Critic(num_actions, default_value=E_0),
        actor_R=DeterministicActor(num_actions,
                                   tie_break='first',
                                   boredom=kid_boredom),
        critic_R=Critic(num_actions, default_value=R_0),
    )
    kid_memories = [
        DiscreteDistribution(initial_bins=initial_bins)
        for _ in range(num_actions)
    ]

    # ------------------------------------------------------------------------
    # !
    total_R = 0.0
    total_E = 0.0
    total_G = 0.0
    for n in range(num_episodes):
        # ---
        # Set env
        if n < change:
            env = env1
        else:
            env = env2
        env.reset()

        # ---
        # PAR move (always first)
        actor, critic, par_policy = par_wsls(par_E, par_R)
        par_action = actor(list(critic.model.values()))

        # Est. regret and save it
        par_G = estimate_regret(all_actions, par_action, critic)

        # Pull a lever.
        par_state, par_R, _, _ = env.step(par_action)
        par_R = R_homeostasis(par_R, total_R, num_episodes)

        # ---
        # KID move
        actor, critic, kid_policy = kid_wsls(kid_E, kid_R)
        kid_action = actor(list(critic.model.values()))

        # Est. regret and save it
        kid_G = estimate_regret(all_actions, kid_action, critic)

        # Pull a lever.
        kid_state, kid_R, _, _ = env.step(kid_action)
        kid_R = R_homeostasis(kid_R, total_R, set_point)

        # ---
        # Share and learn
        par_R = par_R * (1 - share)
        kid_R += par_R * share

        # PAR
        old = deepcopy(par_memories[par_action])
        par_memories[par_action].update((int(par_state), int(par_R)))
        new = deepcopy(par_memories[par_action])
        par_E = kl(new, old, E_0)
        old = deepcopy(par_memories[par_action])
        par_memories[kid_action].update((int(kid_state), int(kid_R)))
        new = deepcopy(par_memories[par_action])
        par_E += kl(new, old, E_0)  # note inplace

        # KID
        old = deepcopy(kid_memories[kid_action])
        kid_memories[kid_action].update((int(kid_state), int(kid_R)))
        new = deepcopy(kid_memories[kid_action])
        kid_E = kl(new, old, E_0)
        old = deepcopy(kid_memories[par_action])
        kid_memories[par_action].update((int(par_state), int(kid_R)))
        new = deepcopy(kid_memories[par_action])
        kid_E += kl(new, old, E_0)  # note inplace

        # Learning, both policies.
        par_wsls.update(par_action, par_E, par_R, lr_R)
        kid_wsls.update(kid_action, kid_E, kid_R, lr_R)

        # ---
        # Log
        log.add_scalar("par_policy", par_policy, n)
        log.add_scalar("par_state", par_state, n)
        log.add_scalar("par_action", par_action, n)
        log.add_scalar("par_regret", par_G, n)
        log.add_scalar("par_score_E", par_E, n)
        log.add_scalar("par_score_R", par_R, n)
        log.add_scalar("par_value_E", par_wsls.critic_E(par_action), n)
        log.add_scalar("par_value_R", par_wsls.critic_R(par_action), n)
        log.add_scalar("kid_policy", kid_policy, n)
        log.add_scalar("kid_state", kid_state, n)
        log.add_scalar("kid_action", kid_action, n)
        log.add_scalar("kid_regret", kid_G, n)
        log.add_scalar("kid_score_E", kid_E, n)
        log.add_scalar("kid_score_R", kid_R, n)
        log.add_scalar("kid_value_E", kid_wsls.critic_E(kid_action), n)
        log.add_scalar("kid_value_R", kid_wsls.critic_R(kid_action), n)
        total_E += par_E + kid_E
        total_R += par_R + kid_R
        total_G += par_G + kid_G
        log.add_scalar("total_G", total_G, n)
        log.add_scalar("total_E", total_E, n)
        log.add_scalar("total_R", total_R, n)
        tie = 0
        if actor.tied:
            tie = 1
        log.add_scalar("kid_ties", tie, n)
    log.close()

    # ------------------------------------------------------------------------
    # Build the final result and save it
    result = dict(best1=env1.best,
                  best2=env2.best,
                  num_episodes=num_episodes,
                  change=change,
                  tie_break=tie_break,
                  par_boredom=par_boredom,
                  kid_boredom=kid_boredom,
                  par_critic_E=par_wsls.critic_E.state_dict(),
                  par_critic_R=par_wsls.critic_R.state_dict(),
                  par_memories=[m.state_dict() for m in par_memories],
                  kid_critic_E=kid_wsls.critic_E.state_dict(),
                  kid_critic_R=kid_wsls.critic_R.state_dict(),
                  kid_memories=[m.state_dict() for m in kid_memories],
                  total_E=total_E,
                  total_R=total_R,
                  total_G=total_G,
                  lr_R=lr_R,
                  master_seed=master_seed)
    save_checkpoint(result, filename=os.path.join(log.log_dir, "result.pkl"))

    return total_R


def par(num_episodes=1000,
        change=500,
        tie_break='next',
        par_boredom=0.0,
        lr_R=.1,
        master_seed=42,
        initial_bins=None,
        log_dir=None):
    """Parents play a game of changing bandits"""

    # ------------------------------------------------------------------------
    # Sanity
    if change > num_episodes:
        raise ValueError("change must be less the num_episodes")
    # log
    log = SummaryWriter(log_dir=log_dir, write_to_disk=True)

    # ------------------------------------------------------------------------
    # Init tasks
    env1 = BanditUniform121()
    env2 = BanditChange121()
    env1.seed(master_seed)
    env2.seed(master_seed)
    env1.reset()
    env2.reset()
    num_actions = env1.action_space.n
    all_actions = list(range(num_actions))

    # Init values
    R_0 = 0
    E_0 = entropy(np.ones(num_actions) / num_actions)
    par_E = E_0
    par_R = R_0

    # Init agents and memories
    # PAR
    par_wsls = WSLS(
        actor_E=DeterministicActor(num_actions,
                                   tie_break=tie_break,
                                   boredom=par_boredom),
        critic_E=Critic(num_actions, default_value=E_0),
        actor_R=DeterministicActor(num_actions,
                                   tie_break='first',
                                   boredom=par_boredom),
        critic_R=Critic(num_actions, default_value=R_0),
    )
    par_memories = [
        DiscreteDistribution(initial_bins=initial_bins)
        for _ in range(num_actions)
    ]

    # ------------------------------------------------------------------------
    # !
    total_R = 0.0
    total_E = 0.0
    total_G = 0.0
    for n in range(num_episodes):
        # ---
        # Set env
        if n < change:
            env = env1
        else:
            env = env2
        env.reset()

        # ---
        # PAR move (always first)
        actor, critic, par_policy = par_wsls(par_E, par_R)
        par_action = actor(list(critic.model.values()))

        # Est. regret and save it
        par_G = estimate_regret(all_actions, par_action, critic)

        # Pull a lever.
        par_state, par_R, _, _ = env.step(par_action)
        par_R = R_homeostasis(par_R, total_R, num_episodes)

        # PAR
        old = deepcopy(par_memories[par_action])
        par_memories[par_action].update((int(par_state), int(par_R)))
        new = deepcopy(par_memories[par_action])
        par_E = kl(new, old, E_0)

        # Learning, both policies.
        par_wsls.update(par_action, par_E, par_R, lr_R)

        # ---
        # Log
        log.add_scalar("par_policy", par_policy, n)
        log.add_scalar("par_state", par_state, n)
        log.add_scalar("par_action", par_action, n)
        log.add_scalar("par_regret", par_G, n)
        log.add_scalar("par_score_E", par_E, n)
        log.add_scalar("par_score_R", par_R, n)
        log.add_scalar("par_value_E", par_wsls.critic_E(par_action), n)
        log.add_scalar("par_value_R", par_wsls.critic_R(par_action), n)
        total_E += par_E
        total_R += par_R
        total_G += par_G
        log.add_scalar("total_G", total_G, n)
        log.add_scalar("total_E", total_E, n)
        log.add_scalar("total_R", total_R, n)
        tie = 0
        if actor.tied:
            tie = 1
        log.add_scalar("kid_ties", tie, n)
    log.close()

    # ------------------------------------------------------------------------
    # Build the final result and save it
    result = dict(best1=env1.best,
                  best2=env2.best,
                  num_episodes=num_episodes,
                  change=change,
                  tie_break=tie_break,
                  par_boredom=par_boredom,
                  par_critic_E=par_wsls.critic_E.state_dict(),
                  par_critic_R=par_wsls.critic_R.state_dict(),
                  par_memories=[m.state_dict() for m in par_memories],
                  total_E=total_E,
                  total_R=total_R,
                  total_G=total_G,
                  lr_R=lr_R,
                  master_seed=master_seed)
    save_checkpoint(result, filename=os.path.join(log.log_dir, "result.pkl"))

    return total_R


if __name__ == "__main__":
    fire.Fire({"parkid": parkid, "par": par})
