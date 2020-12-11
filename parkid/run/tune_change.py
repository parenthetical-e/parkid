import fire
import os
import csv
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from scipy.stats import loguniform
from functools import partial
from multiprocessing import Pool

# Borrow utils
from infomercial.utils import save_checkpoint
from infomercial.utils import load_checkpoint

# Borrow tune utils
from infomercial.exp.tune_bandit import get_best_trial
from infomercial.exp.tune_bandit import get_sorted_trials
from infomercial.exp.tune_bandit import get_best_result
from infomercial.exp.tune_bandit import get_configs
from infomercial.exp.tune_bandit import get_metrics
from infomercial.exp.tune_bandit import save_csv

# Our target
from parkid.run import change_bandits


def _train(exp_func=None,
           env_name1=None,
           env_name2=None,
           change=None,
           share_update=None,
           metric=None,
           num_episodes=None,
           num_repeats=None,
           master_seed=None,
           config=None):

    # Run
    scores = []
    for n in range(num_repeats):
        seed = None
        if master_seed is not None:
            seed = master_seed + n

        trial = exp_func(
            env_name1=env_name1,
            env_name2=env_name2,
            change=change,
            share_update=share_update,
            num_episodes=num_episodes,
            master_seed=seed,  # override
            **config)
        scores.append(trial[metric])

    # Override metric, with num_repeat average
    trial[metric] = np.median(scores)

    # Save metadata
    trial.update({
        "config": config,
        "env_name1": env_name1,
        "env_name2": env_name2,
        "change": change,
        "num_episodes": num_episodes,
        "num_repeats": num_repeats,
        "metric": metric,
        "master_seed": master_seed,
    })

    return trial


def random(name,
           model_name="parkid",
           env_name1="BanditUniform4",
           env_name2="BanditChange4",
           change=100,
           num_episodes=40,
           share_update=False,
           num_repeats=10,
           num_samples=10,
           num_processes=1,
           metric="change_R",
           verbose=False,
           master_seed=None,
           **config_kwargs):
    """Tune hyperparameters for change_bandits."""

    # -
    # Init:
    # Separate name from path
    path, name = os.path.split(name)

    # Look up the bandit run function were using in this tuning.
    exp_func = getattr(change_bandits, model_name)

    # Build the parallel callback
    trials = []

    # generate sep prngs for each kwargs
    prngs = []
    for i in range(len(config_kwargs)):
        if master_seed is not None:
            prng = np.random.RandomState(master_seed + i)
        else:
            prng = np.random.RandomState()
        prngs.append(prng)

    def append_to_results(result):
        # Keep only params and scores, to save
        # memory for when N is large
        trial = {}
        trial["config"] = result["config"]
        trial[metric] = result[metric]
        trials.append(trial)

    # Setup default params
    params = dict(exp_func=exp_func,
                  env_name1=env_name1,
                  env_name2=env_name2,
                  change=change,
                  share_update=share_update,
                  num_episodes=num_episodes,
                  num_repeats=num_repeats,
                  metric=metric,
                  master_seed=master_seed,
                  config={})

    # -
    # Run!
    # Setup the parallel workers
    workers = []
    pool = Pool(processes=num_processes)
    for n in range(num_samples):

        # Reset param sample for safety
        params["config"] = {}
        params["config"]["write_to_disk"] = False
        # Make a new sample
        for i, (k, par) in enumerate(config_kwargs.items()):
            try:
                mode, low, high = par
                mode = str(mode)

                if mode == "loguniform":
                    params["config"][k] = loguniform(
                        low, high).rvs(random_state=prngs[i])
                    print(f"{k} : {params['config'][k]}")
                elif mode == "uniform":
                    params["config"][k] = prngs[i].uniform(low=low, high=high)
                else:
                    raise ValueError(f"mode {mode} not understood")

            except TypeError:  # number?
                params["config"][k] = float(par)
            except ValueError:  # string?
                params["config"][k] = str(par)

        # A worker gets the new sample
        workers.append(
            pool.apply_async(_train,
                             kwds=deepcopy(params),
                             callback=append_to_results))

    # Get the worker's result (blocks until complete)
    for worker in tqdm(workers):
        worker.get()
    pool.close()
    pool.join()

    # Cleanup - dump write_to_disk arg
    for trial in trials:
        del trial["config"]["write_to_disk"]

    # -
    # Sort and save the configs of all trials
    sorted_configs = {}
    for i, trial in enumerate(get_sorted_trials(trials, metric)):
        sorted_configs[i] = trial["config"]
        sorted_configs[i].update({metric: trial[metric]})
    save_csv(sorted_configs, filename=os.path.join(path, name + "_sorted.csv"))

    return get_best_trial(trials, metric)


# ----------------------------------------------------------------------------
# !
if __name__ == "__main__":
    # Create CL interface
    fire.Fire({"random": random})
