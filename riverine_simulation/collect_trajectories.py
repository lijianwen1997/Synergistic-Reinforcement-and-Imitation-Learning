import time
import csv
import os
import sys

import numpy as np

from env_utils import make_unity_env
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

import matplotlib.pyplot as plt

sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning")
from ppo import PPO
import bc

# choose which map to test your trained agent
# difficulty_level = 'easy'
difficulty_level = 'medium'
# difficulty_level = 'hard'

'''
Specify unity river environment path and seed
'''
env_seed = 1
# env_path = '/home/edison/Terrain/circular_river_medium/circular_river_medium.x86_64'
# env_path = '/home/edison/Downloads/circular_river_medium_9_14/circular_river_medium/circular_river_medium.x86_64'
# env_path = f'/home/edison/Terrain/circular_river_{difficulty_level}/circular_river_{difficulty_level}.x86_64'
env_path = f'/home/edison/Terrain/circular_river_{difficulty_level}_configurable/circular_river_{difficulty_level}_configurable.x86_64'
# env_path = None

'''
Specify the RL model
'''
# model_path = 'weights/PPO_BC_STATIC_EASY_SEED2.zip'
# model_path = 'medium_ppo_seed_3_100k_static_bc.zip'
# model_path = 'medium_ppo_seed_1_150k_static_bc.zip'
# model_path = 'PPO_MEDIUM_unity_river_105000_seed_1good.zip'
# model_path = 'PPO_MEDIUM_unity_river_120000_seed_1good.zip'
# model_path = 'medium_ppo_seed_1_150k.zip'
model_path = 'weights/BC_medium_retrain_RGB_50_05'
# model_path = 'weights/BC_medium_retrain_RGB_50_04'
# model_path = 'models_old/PPO_MEDIUM_unity_river_130000_seed_1static.zip'
# model_path = 'models_old/PPO_MEDIUM_unity_river_85000_seed_1good.zip'

# model_name = model_path.split('/')[1].split('.')[0]
model_name = model_path.split('.')[0]

# VAE model, should be the same model across all maps
vae_model_name = f'vae-sim-rgb-all.pth'

'''
Specify the map difficulty level
'''
# difficulty_level = 'hard'
# difficulty_level = 'medium'
# difficulty_level = 'easy'

'''
Specify the algorithm
'''
# algorithm = 'ppo'
algorithm = 'bc'
# algorithm = 'ppo_static_bc'
# algorithm = 'ppo_dynamic_bc'

max_episodes = 50
save_traj = False
traj_id = 0
# traj_dir = 'river_splines/medium/traj_' + 'continue/'
traj_dir = f'river_splines/{difficulty_level}/{algorithm}/traj_' + 'random/'
# if os.path.exists(traj_dir):
#     os.rmdir(traj_dir)
os.makedirs(traj_dir, exist_ok=True)
traj_path = traj_dir + 'traj' + str(traj_id) + '.csv'

# stat csv path
model_stats_dir = f'river_splines/{difficulty_level}/{algorithm}/'
os.makedirs(model_stats_dir, exist_ok=True)
stats_csv_path = model_stats_dir + 'stats.csv'
if os.path.exists(stats_csv_path):
    os.remove(stats_csv_path)


save_stats = False
pause_after_ep = False


def update_traj_path():
    global traj_path
    traj_path = traj_dir + 'traj' + str(traj_id) + '.csv'


def write_point(obs_point: np.ndarray):
    with open(traj_path, 'a+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([obs_point[0], obs_point[2], obs_point[1]])


if __name__ == '__main__':
    # env = make_unity_env(env_path, 1, True, env_seed, start_index=1, vae_model_name=vae_model_name)

    unity_env = UnityEnvironment(file_name=env_path, no_graphics=False, seed=env_seed,
                                 additional_args=['-logFile', 'unity.log'])
    env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False, allow_multiple_obs=True,
                            encode_obs=True, wait_frames_num=0, vae_model_name=vae_model_name)

    if 'BC' in model_name:
        model = bc.reconstruct_policy(model_path)
        print('Use BC!')
    else:
        model = PPO.load(model_path)
        print(f'PPO model {model_name} is loaded!')

    ep_mean_rew = []
    ep_reward = []
    obs = env.reset()
    if save_traj:
        write_point(obs[1])

    while traj_id < max_episodes:
        action, _states = model.predict(obs[0])
        # action, _states = model.predict(obs)
        # print(f'{action=}')
        obs, rewards, done, info = env.step(action)

        # if rewards > 5:
        #     exit(0)

        if save_traj:
            write_point(obs[1])

        ep_reward.append(rewards)
        # print(f'{ep_reward=}')

        if done:
            ep_mean_rew.append(sum(ep_reward))
            print(f'Done! Episode id: {traj_id}, len: {len(ep_reward)}, rew: {sum(ep_reward)}, mean rew: {np.mean(ep_mean_rew)}')

            if save_stats:
                with open(stats_csv_path, 'a+', newline='') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([len(ep_reward), sum(ep_reward)])  # [episode length, episode reward]

            if pause_after_ep:
                i = input('Press n to continue, q to quit ...')
                if i == 'n':
                    obs = env.reset()
                    traj_id += 1
                    update_traj_path()
                    ep_reward = []
                    continue
                elif i == 'q':
                    exit(0)
            else:
                obs = env.reset()
                traj_id += 1
                update_traj_path()
                ep_reward = []
