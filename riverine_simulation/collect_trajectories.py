import csv
import os
import numpy as np

from mlagents_envs.environment import UnityEnvironment
from riverine_simulation.unity_gym_env import UnityToGymWrapper

from drl_deg.ppo import PPO


'''
Specify unity river environment path and seed
'''
env_seed = 1
# choose which map to test your trained agent
riverine_map = 'training'  # or 'testing'
env_path = f'unity_riverine_envs/riverine_{riverine_map}_env/riverine_{riverine_map}_env.x86_64'

'''
Specify the RL model to load
'''
model_path = 'policy_models/ppo_bc_dynamic.zip'  # or 'policy_models/ppo_bc_static.zip'
model_name = model_path.split('/')[1].split('.')[0]

# VAE model, should be the same model across all maps
vae_model_name = f'vae-sim-rgb-all.pth'

'''
Specify the algorithm
'''
algorithm = 'ppo_dynamic_bc'  # or 'ppo_static_bc', or 'ppo', 'bc' if there is the model

max_episodes = 50
traj_id = 0
traj_dir = f'{riverine_map}/{algorithm}/traj_' + 'random/'
os.makedirs(traj_dir, exist_ok=True)
traj_path = traj_dir + 'traj' + str(traj_id) + '.csv'

# stat csv path
model_stats_dir = f'{riverine_map}/{algorithm}/'
os.makedirs(model_stats_dir, exist_ok=True)
stats_csv_path = model_stats_dir + 'stats.csv'
if os.path.exists(stats_csv_path):
    os.remove(stats_csv_path)

save_traj = False  # whether to save agent position as it goes
save_stats = False  # whether to save episode length and reward for all episodes
pause_after_ep = False


def update_traj_path():
    global traj_path
    traj_path = traj_dir + 'traj' + str(traj_id) + '.csv'


def write_point(obs_point: np.ndarray):
    """
    Save agent point to csv file
    :param obs_point:
    :return:
    """
    with open(traj_path, 'a+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([obs_point[0], obs_point[2], obs_point[1]])


if __name__ == '__main__':
    unity_env = UnityEnvironment(file_name=env_path, no_graphics=False, seed=env_seed,
                                 additional_args=['-logFile', 'unity.log'])
    env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False, allow_multiple_obs=True,
                            encode_obs=True, wait_frames_num=0, vae_model_name=vae_model_name)

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

        if save_traj:
            write_point(obs[1])

        ep_reward.append(rewards)

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
