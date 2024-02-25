import gym
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env

import bc
from train_utils import *
from cliff_circular.cliffcircular_gym import CliffCircularGymEnv


if __name__ == "__main__":
    """
    Train IL agent in gym CircularCliff environment.
    """
    il_name = "BC"
    train_il = False
    render_mode = None  # or 'human'
    tmp_path = "/tmp/sb3_log/"
    seed = 1

    # env_name = "LunarLanderContinuous-v2"  # "CartPole-v1" #   #"LunarLander-v2" #
    env_name = 'CliffCircular-gym-v0'
    env = gym.make(env_name, render_mode=render_mode, seed=seed)
    # check_env(env)

    np.set_printoptions(suppress=True)
    train_il_ep = 100
    n_eval_episodes = 50
    rng = np.random.default_rng(seed=seed)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    dataset = 'success'

    # load trajectories
    print("Load good transitions.")
    # d_s = read_csv(dataset, env_name + '-1', seed=0, test_type='merge')
    d_s = read_csv(dataset, env_name + '-3', seed=0, test_type='merge')

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=d_s,
        ent_weight=0.0,
        batch_size=2,  # set to a smaller batch size for CliffCircular environment
        rng=rng,
        verbose=1,
    )

    # evaluate before training
    # reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=n_eval_episodes, render=False,)
    # print(f"Reward of IL before training: "+'{:.4f}'.format(reward))

    # training or loading
    if train_il:
        print("Training a policy using Behavior Cloning")
        bc_trainer.train(n_epochs=train_il_ep, log_interval=100, progress_bar=True)
        bc_trainer.save_policy("weight/" + il_name + "_" + env_name + "_" + str(train_il_ep))
        # evaluate the trained agent
        reward, _ = evaluate_policy(
            bc_trainer.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=n_eval_episodes,
            render=False,
            deterministic=True,
        )
    else:
        print("Load a policy of Behavior Cloning")
        bc_trainer = bc.reconstruct_policy("weight/" + il_name + "_" + env_name + "_" + str(train_il_ep))
        # evaluate the trained agent
        obs = env.reset()
        all_rewards = []
        all_lens = []
        current_rewards = 0
        ep_len = 0
        done = False
        ep = 0
        while ep < n_eval_episodes:
            actions, _ = bc_trainer.predict(obs, deterministic=True)  # we want deterministic policy prediction
            obs, reward, done, infos = env.step(int(actions))
            ep_len += 1
            current_rewards += reward
            if done:
                print(f'{ep=} {current_rewards=}')
                all_lens.append(ep_len)
                all_rewards.append(current_rewards)
                ep += 1
                obs = env.reset()
                current_rewards = 0
                ep_len = 0
        mean_reward = np.mean(all_rewards)
        mean_len = np.mean(all_lens)
        print(f'{mean_reward=} {mean_len=}')
        # reward, _ = evaluate_policy(bc_trainer, env, n_eval_episodes=n_eval_episodes, render=False)
    # print(f"Reward of IL after training using good demonstration: "+'{:.4f}'.format(reward))

    for _ in range(1):
        cnt = 0
        for i in range(d_s.acts.shape[0]):
            if train_il:
                prediction, _ = bc_trainer.policy.predict(d_s.obs[i])
            else:
                prediction, _ = bc_trainer.predict(d_s.obs[i])
            # print(prediction, d_s.acts[i])
            if np.array_equal(int(prediction), int(d_s.acts[i][0])):
                cnt += 1
        print("Success rate: ", cnt / d_s.acts.shape[0])

