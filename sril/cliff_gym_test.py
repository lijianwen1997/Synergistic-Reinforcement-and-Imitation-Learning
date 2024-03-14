import time
import numpy as np

from stable_baselines3.common.logger import configure
import sys

from ppo import PPO
import bc
from bc import *
from evaluation import evaluate_policy

import time

# Modifiable values
import sys
sys.path.append("../cliff_circular")
from cliffcircular_gym import CliffCircularGymEnv
tmp_path = "/tmp/sb3_log/"

np.set_printoptions(suppress=True)

rng = np.random.default_rng(0)
algo = "bc" # "static", "dynamic", "ppo"
seed = 1
n_eval_episodes = 100
def test():
    env_name = 'CliffCircular-gym-v0'
    if algo == "bc":

        rewardss = []
        for seed in range(8,9):
            rewards = []
            for i in range(10,11):
                env = gym.make(env_name, seed=8)
                #model_save_name = "weight/expert/BC_MEDIUM_"+env_name+"_"+str(i*14000)+"_seed_"+str(seed)+"_deg"
                model_save_name = "weight/BC_"+env_name+"_50"
                print(model_save_name)
                model = bc.reconstruct_policy(model_save_name)
                t0 = time.time()
                reward, episode_rewards, traj_good, traj_bad = evaluate_policy(
                    model,  # type: ignore[arg-type]
                    env,
                    n_eval_episodes=n_eval_episodes,
                    render=False,
                    reward_threshold=18,
                    env_name = env_name,
                )
                print( time.time()-t0)
                rewards.append(np.sum(episode_rewards) / n_eval_episodes)
                print(np.sum(episode_rewards) / n_eval_episodes)

                env.close()
                #time.sleep(1)  # wait for unity window to close completely
            print(episode_rewards)
            rewardss.append(rewards)

        print(rewardss)


    else:
        if algo == "ppo":

            model_save_name = "cliff_training_1_140k_static_bc"
            env = gym.make(env_name, seed=8)
            #model = PPO("MlpPolicy", env=env,   n_steps=1024, batch_size=64, n_epochs=10)
            model = PPO.load(model_save_name, env=env)

            reward, episode_rewards, traj_good, traj_bad = evaluate_policy(
                model.policy,  # type: ignore[arg-type]
                env,
                n_eval_episodes=n_eval_episodes,
                render=False,
                reward_threshold=18,
                env_name=env_name,
            )
            print(episode_rewards)
            print(np.sum(episode_rewards) / n_eval_episodes)




if __name__ == '__main__':
    test()
