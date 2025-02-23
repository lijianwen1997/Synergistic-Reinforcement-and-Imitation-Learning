"""
Description: This script is adapted from stable_baselin3/common/evaluation.py
             It evaluates the policy and returns mean_reward, episode_rewards, trajectory_good
Link: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py
Last Revision: Mar 2024
"""
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from imitation.data.types import Transitions


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = 1,
    return_episode_rewards: bool = False,
    warn: bool = True,
    env_name: str = "unity_riverine",
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((n_envs,), dtype=bool)

    states_l = []
    actions_l = []
    infos_l = []
    next_states_l = []
    dones_l = []
    trajectory_l = []
    trajectory_good = []
    np.set_printoptions(suppress=True)
    if env_name == 'unity_riverine':
        deterministic = False
    while (episode_counts < episode_count_targets).any():

        actions, states = model.predict(observations, state=states, deterministic=deterministic)
        states_l.append(observations.tolist()[0])

        actions_l.append(actions.tolist()[0])

        observations, rewards, dones, infos = env.step(actions)

        infos_l.append(infos[0])
        next_states_l.append(observations.tolist()[0])
        dones_l.append(dones.tolist()[0])

        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                # check if the trajectories meet the requirements
                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    good_steps = 25
                    if env_name == 'CliffCircular-gym-v0':
                        if current_rewards[i] >= reward_threshold and current_lengths[i] < 27:
                            trajectory_l.append(Transitions(np.array(states_l[:good_steps]),
                                                            np.array(actions_l[:good_steps]),
                                                            np.array(infos_l[:good_steps]),
                                                            np.array(next_states_l[:good_steps]),
                                                            np.array(dones_l[:good_steps])))
                    else:
                        if current_rewards[i] > reward_threshold and current_lengths[i] > 300:
                            trajectory_l.append(
                                Transitions(np.array(states_l[:good_steps]), np.array(actions_l[:good_steps]),
                                            np.array(infos_l[:good_steps]),
                                            np.array(next_states_l[:good_steps]), np.array(dones_l[:good_steps])))

                    states_l = []
                    actions_l = []
                    infos_l = []
                    next_states_l = []
                    dones_l = []
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        for i in range(len(trajectory_l)):
            if episode_rewards[i] >= reward_threshold:
                trajectory_good.append(trajectory_l[i])
    if return_episode_rewards:
        return episode_rewards, episode_lengths, trajectory_good
    return mean_reward, episode_rewards, trajectory_good
