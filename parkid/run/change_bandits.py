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
from parkid.utils import rectified_linear as phi

from parkid.models import Critic
from parkid.models import DeterministicActor
from parkid.models import WSLS
from parkid.models import WSLSh
from parkid.models import CountMemory
from parkid.models import SoftmaxActor

from parkid.gym import bandit
from parkid.models import R_update


def random(num_episodes=1000,
           change=100,
           env_name1="BanditUniform4",
           env_name2="BanditChange4",
           lr_R=.1,
           master_seed=42,
           log_dir=None,
           write_to_disk=True,
           output=True):
    """One random agent on changing bandits."""

    # ------------------------------------------------------------------------
    # Sanity
    if change > num_episodes:
        raise ValueError("change must be less the num_episodes")
    # log
    log = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # ------------------------------------------------------------------------
    # Init envs
    Env1 = getattr(bandit, env_name1)
    Env2 = getattr(bandit, env_name2)
    env1 = Env1()
    env2 = Env2()
    env1.seed(master_seed)
    env2.seed(master_seed)
    env1.reset()
    env2.reset()

    prng = np.random.RandomState(master_seed)

    num_actions = env1.action_space.n
    all_actions = list(range(num_actions))

    # Init agent - just a critic
    # This one is never used, added
    # to generate the ideal RL value
    # learning curves for some env.seed()
    # aand for some lr_R
    R_0 = 0.0
    critic_R = Critic(num_actions, default_value=R_0)

    # ------------------------------------------------------------------------
    # !
    total_R = 0.0
    change_R = 0.0
    for n in range(num_episodes):
        # ---
        # Set env
        if n < change:
            env = env1
        else:
            env = env2
        env.reset()

        # ---
        # get best action, and use it
        #
        # (call this action, etc, par_*
        # to make it easy to plot
        par_action = prng.choice(num_actions)
        _, par_R, _, _ = env.step(par_action)

        # Learning, critic
        critic_R = R_update(par_action, par_R, critic_R, lr_R)

        # ---
        # Log
        log.add_scalar("best", env.best[0], n)
        log.add_scalar("p_opt", env.p_dist[env.best[0]], n)
        log.add_scalar("par_action", par_action, n)
        log.add_scalar("par_score_R", par_R, n)
        log.add_scalar("par_value_R", critic_R(par_action), n)

        total_R += par_R
        if n >= change:
            change_R += par_R
        log.add_scalar("total_R", total_R, n)
        log.add_scalar("change_R", change_R, n)

    log.close()

    # ------------------------------------------------------------------------
    # Build the final result and save it
    result = dict(best1=env1.best,
                  best2=env2.best,
                  num_episodes=num_episodes,
                  change=change,
                  par_critic_R=critic_R.state_dict(),
                  total_R=total_R,
                  change_R=change_R,
                  lr_R=lr_R,
                  master_seed=master_seed)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(log.log_dir, "result.pkl"))

    if output:
        return {"total_R": total_R, "change_R": change_R}
    else:
        return None


def oracle(num_episodes=1000,
           change=100,
           env_name1="BanditUniform4",
           env_name2="BanditChange4",
           lr_R=.1,
           master_seed=42,
           log_dir=None,
           write_to_disk=True,
           output=True):
    """The perfect agent on changing bandits."""
    # ------------------------------------------------------------------------
    # Sanity
    if change > num_episodes:
        raise ValueError("change must be less the num_episodes")
    # log
    log = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # ------------------------------------------------------------------------
    # Init envs
    Env1 = getattr(bandit, env_name1)
    Env2 = getattr(bandit, env_name2)
    env1 = Env1()
    env2 = Env2()
    env1.seed(master_seed)
    env2.seed(master_seed)
    env1.reset()
    env2.reset()

    num_actions = env1.action_space.n
    all_actions = list(range(num_actions))

    # Init agent - just a critic
    # This one is never used, added
    # to generate the ideal RL value
    # learning curves for some env.seed()
    # aand for some lr_R
    R_0 = 0.0
    critic_R = Critic(num_actions, default_value=R_0)

    # ------------------------------------------------------------------------
    # !
    total_R = 0.0
    change_R = 0.0
    for n in range(num_episodes):
        # ---
        # Set env
        if n < change:
            env = env1
        else:
            env = env2
        env.reset()

        # ---
        # get best action, and use it
        #
        # (call this action, etc, par_*
        # to make it easy to plot
        par_action = env.best[0]
        _, par_R, _, _ = env.step(par_action)

        # Learning, critic
        critic_R = R_update(par_action, par_R, critic_R, lr_R)

        # ---
        # Log
        log.add_scalar("best", env.best[0], n)
        log.add_scalar("p_opt", env.p_dist[env.best[0]], n)
        log.add_scalar("par_action", par_action, n)
        log.add_scalar("par_score_R", par_R, n)
        log.add_scalar("par_value_R", critic_R(par_action), n)

        total_R += par_R
        if n >= change:
            change_R += par_R
        log.add_scalar("total_R", total_R, n)
        log.add_scalar("change_R", change_R, n)

    log.close()

    # ------------------------------------------------------------------------
    # Build the final result and save it
    result = dict(best1=env1.best,
                  best2=env2.best,
                  num_episodes=num_episodes,
                  change=change,
                  par_critic_R=critic_R.state_dict(),
                  total_R=total_R,
                  change_R=change_R,
                  lr_R=lr_R,
                  master_seed=master_seed)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(log.log_dir, "result.pkl"))

    if output:
        return {"total_R": total_R, "change_R": change_R}
    else:
        return None


def parkid(num_episodes=1000,
           change=100,
           env_name1="BanditUniform4",
           env_name2="BanditChange4",
           tie_break='next',
           par_boredom=0.0,
           kid_boredom=0.0,
           share=0.0,
           kid_scale=1,
           parent_threshold=0.0,
           set_point=None,
           lr_R=.1,
           share_update=False,
           master_seed=42,
           log_dir=None,
           write_to_disk=True,
           output=True):
    """A parent and kid on changing bandits."""

    # ------------------------------------------------------------------------
    # Sanity
    if change > num_episodes:
        raise ValueError("change must be less the num_episodes")
    # log
    log = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # ------------------------------------------------------------------------
    # Init envs
    Env1 = getattr(bandit, env_name1)
    Env2 = getattr(bandit, env_name2)
    env1 = Env1()
    env2 = Env2()
    env1.seed(master_seed)
    env2.seed(master_seed)
    env1.reset()
    env2.reset()

    num_actions = env1.action_space.n
    all_actions = list(range(num_actions))

    # Init values
    R_0 = 0.0
    E_0 = entropy(np.ones(num_actions) / num_actions)
    par_E = E_0
    par_R = R_0
    kid_E = E_0
    kid_R = R_0
    kid_H = R_0
    initial_bins = [0, 1]

    if set_point is None:
        set_point = num_episodes

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
        boredom=par_boredom,
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
        boredom=kid_boredom,
    )
    kid_memories = [
        DiscreteDistribution(initial_bins=initial_bins)
        for _ in range(num_actions)
    ]

    # ------------------------------------------------------------------------
    # !
    total_R = 0.0
    change_R = 0.0
    change_E = 0.0
    total_E = 0.0
    total_G = 0.0
    total_kid_R = 0.0
    total_kid_H = 0.0
    for n in range(num_episodes):
        # ---
        # Set env
        if n < change:
            env = env1
        else:
            env = env2
        env.reset()

        # ---
        # Get shared E from the last round:
        # Gate and rescale?
        #
        # I am not sure gating then scaling, rather than scaling
        # then gating makes the most physical sense, but it is
        # very convient to do. Proceed for now. Revisit this later.
        share_E = kid_scale * phi(kid_E, parent_threshold)

        # PAR move (always first)
        actor, critic, par_policy = par_wsls(par_E + share_E, par_R)
        par_action = actor(list(critic.model.values()))
        par_state, par_R, _, _ = env.step(par_action)

        # KID move
        actor, critic, kid_policy = kid_wsls(kid_E, kid_H)
        kid_action = actor(list(critic.model.values()))
        kid_state, kid_R, _, _ = env.step(kid_action)

        kid_H = R_homeostasis(kid_R + (par_R * share), total_kid_R, set_point)
        par_G = estimate_regret(all_actions, par_action, critic)
        kid_G = estimate_regret(all_actions, kid_action, critic)

        # ---
        # PAR
        old = deepcopy(par_memories[par_action])
        par_memories[par_action].update((int(par_state), int(par_R)))
        new = deepcopy(par_memories[par_action])
        par_E = kl(new, old, E_0)

        # KID
        old = deepcopy(kid_memories[kid_action])
        kid_memories[kid_action].update((int(kid_state), int(kid_R)))
        new = deepcopy(kid_memories[kid_action])
        kid_E = kl(new, old, E_0)

        # Learning, both policies.
        # Direct
        par_wsls.update(par_action, par_E, par_R * (1 - share), lr_R)
        kid_wsls.update(kid_action, kid_E, kid_H, lr_R)
        # Shared
        if share_update:
            par_wsls.update(kid_action, kid_E, None, lr_R)
            kid_wsls.update(par_action, None, par_R, lr_R)

        # ---
        # Log
        log.add_scalar("best", env.best[0], n)
        log.add_scalar("p_opt", env.p_dist[env.best[0]], n)
        log.add_scalar("par_policy", par_policy, n)
        log.add_scalar("par_state", par_state, n)
        log.add_scalar("par_action", par_action, n)
        log.add_scalar("par_regret", par_G, n)
        log.add_scalar("par_score_E", par_E, n)
        log.add_scalar("par_share_E", share_E, n)
        log.add_scalar("par_score_R", par_R, n)
        log.add_scalar("par_value_E", par_wsls.critic_E(par_action), n)
        log.add_scalar("par_value_R", par_wsls.critic_R(par_action), n)
        log.add_scalar("kid_policy", kid_policy, n)
        log.add_scalar("kid_state", kid_state, n)
        log.add_scalar("kid_action", kid_action, n)
        log.add_scalar("kid_regret", kid_G, n)
        log.add_scalar("kid_score_E", kid_E, n)
        log.add_scalar("kid_score_R", kid_R, n)
        log.add_scalar("kid_score_H", kid_H, n)
        log.add_scalar("kid_value_E", kid_wsls.critic_E(kid_action), n)
        log.add_scalar("kid_value_R", kid_wsls.critic_R(kid_action), n)
        total_E += max(par_E, kid_E)
        total_R += max(par_R, kid_R)
        total_kid_R += kid_R
        total_kid_H += kid_H
        total_G += par_G
        if n >= change:
            change_R += max(par_R, kid_R)
            change_E += max(par_E, kid_E)
        log.add_scalar("total_G", total_G, n)
        log.add_scalar("total_E", total_E, n)
        log.add_scalar("total_R", total_R, n)
        log.add_scalar("total_kid_R", total_kid_R, n)
        log.add_scalar("total_kid_H", total_kid_H, n)
        log.add_scalar("change_R", change_R, n)
        log.add_scalar("change_E", change_E, n)
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
                  change_R=change_R,
                  total_G=total_G,
                  lr_R=lr_R,
                  master_seed=master_seed)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(log.log_dir, "result.pkl"))

    if output:
        return {"total_R": total_R, "change_R": change_R}
    else:
        return None


def parpar(
        num_episodes=1000,
        change=100,
        env_name1="BanditUniform4",
        env_name2="BanditChange4",
        tie_break='next',
        par_boredom=0.0,
        share=0.0,  # dummy
        share_update=False,  # dummy
        lr_R=.1,
        master_seed=42,
        log_dir=None,
        write_to_disk=True,
        output=True):
    """Two parents on changing bandits."""

    # ------------------------------------------------------------------------
    # Sanity
    if change > num_episodes:
        raise ValueError("change must be less the num_episodes")
    # log
    log = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # ------------------------------------------------------------------------
    # Init envs
    Env1 = getattr(bandit, env_name1)
    Env2 = getattr(bandit, env_name2)
    env1 = Env1()
    env2 = Env2()
    env1.seed(master_seed)
    env2.seed(master_seed)
    env1.reset()
    env2.reset()

    num_actions = env1.action_space.n
    all_actions = list(range(num_actions))

    # Init values
    R_0 = 0.0
    E_0 = entropy(np.ones(num_actions) / num_actions)
    par_E = E_0
    par_R = R_0
    alt_E = E_0
    alt_R = R_0
    alt_boredom = par_boredom
    initial_bins = [0, 1]

    # Init agents and memories
    # PAR
    par_wsls = WSLS(actor_E=DeterministicActor(num_actions,
                                               tie_break=tie_break,
                                               boredom=par_boredom),
                    critic_E=Critic(num_actions, default_value=E_0),
                    actor_R=DeterministicActor(num_actions,
                                               tie_break='first',
                                               boredom=par_boredom),
                    critic_R=Critic(num_actions, default_value=R_0),
                    boredom=par_boredom)
    par_memories = [
        DiscreteDistribution(initial_bins=initial_bins)
        for _ in range(num_actions)
    ]
    # ALT
    alt_wsls = WSLS(actor_E=DeterministicActor(num_actions,
                                               tie_break=tie_break,
                                               boredom=alt_boredom),
                    critic_E=Critic(num_actions, default_value=E_0),
                    actor_R=DeterministicActor(num_actions,
                                               tie_break='first',
                                               boredom=alt_boredom),
                    critic_R=Critic(num_actions, default_value=R_0),
                    boredom=alt_boredom)
    alt_memories = [
        DiscreteDistribution(initial_bins=initial_bins)
        for _ in range(num_actions)
    ]

    # ------------------------------------------------------------------------
    # !
    total_R = 0.0
    change_R = 0.0
    change_E = 0.0
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

        # ---
        # ALT move
        actor, critic, alt_policy = alt_wsls(alt_E, alt_R)
        alt_action = actor(list(critic.model.values()))

        # Est. regret and save it
        alt_G = estimate_regret(all_actions, alt_action, critic)

        # Pull a lever.
        alt_state, alt_R, _, _ = env.step(alt_action)

        # ---
        # PAR
        old = deepcopy(par_memories[par_action])
        par_memories[par_action].update((int(par_state), int(par_R)))
        new = deepcopy(par_memories[par_action])
        par_E = kl(new, old, E_0)

        # ALT
        old = deepcopy(alt_memories[alt_action])
        alt_memories[alt_action].update((int(alt_state), int(alt_R)))
        new = deepcopy(alt_memories[alt_action])
        alt_E = kl(new, old, E_0)

        # Learning, both policies.
        par_wsls.update(par_action, par_E, par_R, lr_R)
        alt_wsls.update(alt_action, alt_E, alt_R, lr_R)

        # Shared
        # if share_update:
        # pass
        # par_wsls.update(alt_action, alt_E, None, lr_R)
        # alt_wsls.update(par_action, None, par_R, lr_R)

        # ---
        # Log
        log.add_scalar("best", env.best[0], n)
        log.add_scalar("p_opt", env.p_dist[env.best[0]], n)
        log.add_scalar("par_policy", par_policy, n)
        log.add_scalar("par_state", par_state, n)
        log.add_scalar("par_action", par_action, n)
        log.add_scalar("par_regret", par_G, n)
        log.add_scalar("par_score_E", par_E, n)
        log.add_scalar("par_score_R", par_R, n)
        log.add_scalar("par_value_E", par_wsls.critic_E(par_action), n)
        log.add_scalar("par_value_R", par_wsls.critic_R(par_action), n)
        log.add_scalar("alt_policy", alt_policy, n)
        log.add_scalar("alt_state", alt_state, n)
        log.add_scalar("alt_action", alt_action, n)
        log.add_scalar("alt_regret", alt_G, n)
        log.add_scalar("alt_score_E", alt_E, n)
        log.add_scalar("alt_score_R", alt_R, n)
        log.add_scalar("alt_value_E", alt_wsls.critic_E(alt_action), n)
        log.add_scalar("alt_value_R", alt_wsls.critic_R(alt_action), n)
        total_E += max(par_E, alt_E)
        total_R += max(par_R, alt_R)
        total_G += max(par_G, alt_G)
        if n >= change:
            change_R += max(par_R, alt_R)
            change_E += max(par_E, alt_E)
        log.add_scalar("total_G", total_G, n)
        log.add_scalar("total_E", total_E, n)
        log.add_scalar("total_R", total_R, n)
        log.add_scalar("change_R", change_R, n)
        log.add_scalar("change_E", change_E, n)
        tie = 0
        if actor.tied:
            tie = 1

        log.add_scalar("alt_ties", tie, n)
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
                  alt_critic_E=alt_wsls.critic_E.state_dict(),
                  alt_critic_R=alt_wsls.critic_R.state_dict(),
                  alt_memories=[m.state_dict() for m in alt_memories],
                  total_E=total_E,
                  total_R=total_R,
                  change_R=change_R,
                  total_G=total_G,
                  lr_R=lr_R,
                  master_seed=master_seed)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(log.log_dir, "result.pkl"))

    if output:
        return {"total_R": total_R, "change_R": change_R}
    else:
        return None


def ucbucb(num_episodes=1000,
           change=100,
           env_name1="BanditUniform4",
           env_name2="BanditChange4",
           temp=1.0,
           beta=1.0,
           lr_R=.1,
           master_seed=42,
           log_dir=None,
           write_to_disk=True,
           output=True):
    """Two UCB agents on changing bandits."""

    # ------------------------------------------------------------------------
    # Sanity
    if change > num_episodes:
        raise ValueError("change must be less the num_episodes")
    # log
    log = SummaryWriter(log_dir=log_dir, write_to_disk=write_to_disk)

    # ------------------------------------------------------------------------
    # Init envs
    Env1 = getattr(bandit, env_name1)
    Env2 = getattr(bandit, env_name2)
    env1 = Env1()
    env2 = Env2()
    env1.seed(master_seed)
    env2.seed(master_seed)
    env1.reset()
    env2.reset()

    num_actions = env1.action_space.n
    all_actions = list(range(num_actions))

    # Init values
    R_0 = 0.0
    par_R = R_0
    alt_R = R_0

    # Agents and memories
    par_critic = Critic(num_actions, default_value=par_R)
    par_actor = SoftmaxActor(num_actions, temp=temp, seed_value=master_seed)
    par_count = CountMemory()

    alt_critic = Critic(num_actions, default_value=alt_R)
    alt_actor = SoftmaxActor(num_actions, temp=temp, seed_value=master_seed)
    alt_count = CountMemory()

    # ------------------------------------------------------------------------
    # !
    total_R = 0.0
    change_R = 0.0
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
        par_action = par_actor(list(par_critic.model.values()))

        # Est. regret and save it
        par_G = estimate_regret(all_actions, par_action, par_critic)

        # Pull a lever.
        par_state, par_R, _, _ = env.step(par_action)

        # ---
        # ALT move
        alt_action = alt_actor(list(alt_critic.model.values()))

        # Est. regret and save it
        alt_G = estimate_regret(all_actions, alt_action, alt_critic)

        # Pull a lever.
        alt_state, alt_R, _, _ = env.step(alt_action)

        # ---
        # Count bonus:
        # PAR
        par_bonus = ((2 * np.log(n + 1)) / par_count(par_action))**(0.5)

        # ALT
        alt_bonus = ((2 * np.log(n + 1)) / alt_count(alt_action))**(0.5)

        # Learning, both policies.
        par_payout = par_R + (beta * par_bonus)
        par_critic = R_update(par_action, par_payout, par_critic, lr_R)
        alt_payout = alt_R + (beta * alt_bonus)
        alt_critic = R_update(alt_action, alt_payout, alt_critic, lr_R)

        # ---
        # Log
        log.add_scalar("best", env.best[0], n)
        log.add_scalar("par_state", par_state, n)
        log.add_scalar("par_action", par_action, n)
        log.add_scalar("par_regret", par_G, n)
        log.add_scalar("par_bonus", par_bonus, n)
        log.add_scalar("par_score_R", par_R, n)
        log.add_scalar("par_value_R", par_critic(par_action), n)
        log.add_scalar("alt_state", alt_state, n)
        log.add_scalar("alt_action", alt_action, n)
        log.add_scalar("alt_regret", alt_G, n)
        log.add_scalar("alt_bonus", alt_bonus, n)
        log.add_scalar("alt_score_R", alt_R, n)
        log.add_scalar("alt_value_R", alt_critic(alt_action), n)
        total_R += max(par_R, alt_R)
        total_G += max(par_G, alt_G)
        if n >= change:
            change_R += max(par_R, alt_R)
        log.add_scalar("total_G", total_G, n)
        log.add_scalar("total_R", total_R, n)
        log.add_scalar("change_R", change_R, n)

    log.close()

    # ------------------------------------------------------------------------
    # Build the final result and save it
    result = dict(best1=env1.best,
                  best2=env2.best,
                  num_episodes=num_episodes,
                  change=change,
                  beta=beta,
                  temp=temp,
                  lr_R=lr_R,
                  par_critic=par_critic.state_dict(),
                  par_memories=par_count.state_dict(),
                  alt_critic=alt_critic.state_dict(),
                  alt_memories=alt_count.state_dict(),
                  total_R=total_R,
                  change_R=change_R,
                  total_G=total_G,
                  master_seed=master_seed)

    if write_to_disk:
        save_checkpoint(result,
                        filename=os.path.join(log.log_dir, "result.pkl"))

    if output:
        return {"total_R": total_R, "change_R": change_R}
    else:
        return None


if __name__ == "__main__":
    fire.Fire({
        "oracle": oracle,
        "parkid": parkid,
        "parpar": parpar,
        "ucbucb": ucbucb,
        "random": random
    })
