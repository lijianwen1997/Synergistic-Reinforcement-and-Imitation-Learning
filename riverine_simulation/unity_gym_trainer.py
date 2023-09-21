import time
import os
import numpy as np

from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3 import PPO

import sys
sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning")
from ppo import PPO
import bc
from train_mirl import SaveOnBestTrainingRewardCallback
from evaluation import evaluate_policy

from env_utils import make_unity_env


# Modifiable values
env_seed = 1
train_rl = True
rl_name = "PPO"
il_name = "BC"
sample_good = False
sample_bad = False
train_il_good = False
train_il_bad = False
tmp_path = "/tmp/sb3_log/"

# vae_model_name = 'vae-sim-rgb-easy.pth'
# vae_model_name = 'vae-sim-rgb-medium.pth'
vae_model_name = 'vae-sim-rgb-all.pth'

tb_log_dir = './ppo_river_tensorboard/'
# tb_log_name = 'easy_ppo_compsition'
# tb_log_name = 'easy_ppo_idle_500'
tb_log_name = 'medium_ppo_small_reward'


# model_save_name = 'circular_easy_ppo_composition'
# model_save_name = 'circular_easy_ppo_idle_500' + '_seed_' + str(env_seed) + '_100k'
model_save_name = 'circular_medium_ppo' + '_seed_' + str(env_seed) + '_small_reward'
# env_name ="LunarLanderContinuous-v2" # 'Pendulum-v1' #"LunarLander-v2" #"CartPole-v1" #
# env_name ="unity-river"

np.set_printoptions(suppress=True)
# train_ppo_ep = 2000000
train_ppo_ep = 150000
train_il_ep = 50
train_il_bad_ep = 1
failure_steps = 5
sample_ep = 5
sample_bad_ep = 100
check_freq = int(train_ppo_ep / 10)
rng = np.random.default_rng(0)
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


def train(seed: int = env_seed, use_callback: bool = True):
    # env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action1d.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    # env_path = '/home/edison/Terrain/circular_river_easy/circular_river_easy.x86_64'
    # env_path = '/home/edison/Terrain/circular_river_easy_500/circular_river_easy_500.x86_64'
    # env_path = '/home/edison/Terrain/circular_river_medium_reward/circular_river_medium_reward.x86_64'
    # env_path = '/home/edison/Terrain/circular_river_medium/circular_river_medium.x86_64'
    env_path = '/home/edison/Downloads/circular_river_medium_9_14/circular_river_medium/circular_river_medium.x86_64'
    # env_path = None

    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=tmp_path,
                                                verbose=1) if use_callback else None

    for seed in [1]:
        for use_bc in [True]:
            env = make_unity_env(env_path, 1, True, seed=seed, start_index=1, vae_model_name=vae_model_name)
            print(f'Unity env is created!')

            if train_rl:
                model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=10, verbose=1,
                        tensorboard_log=tb_log_dir)
            else:
                model = PPO.load(model_save_name + '.zip', env)

            model.use_action_loss = use_bc

            # tb_log_name = 'medium_ppo_seed_' + str(seed) + ('_static_bc' if use_bc else '')
            tb_log_name = 'medium_old_ppo_seed_' + str(seed) + ('_static_bc' if use_bc else '')
            model.learn(total_timesteps=train_ppo_ep, progress_bar=True, tb_log_name=tb_log_name, callback=callback)

            # model_save_name = 'medium_ppo_seed_' + str(seed) + '_150k' + ('_static_bc' if use_bc else '')
            model_save_name = 'medium_old_ppo_seed_' + str(seed) + '_150k' + ('_static_bc' if use_bc else '')
            model.save(model_save_name)
            print(f'Model {model_save_name} has been trained and saved!')

            env.close()
            time.sleep(10)


def predict():
    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder")
    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/utils")
    from evaluation import evaluate_policy
    from vae import VAE
    from dataset import InputChannelConfig
    import torch
    import matplotlib.pyplot as plt
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

    # plt.ion()
    # fig = plt.figure()
    # ax_rgb = fig.add_subplot(111)

    # env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    # env_path = '/home/edison/Terrain/circular_river_easy/circular_river_easy.x86_64'
    # env_path = '/home/edison/Terrain/circular_river_medium/circular_river_medium.x86_64'
    env_path = None

    unity_env = UnityEnvironment(file_name=env_path, no_graphics=False, seed=env_seed,
                                 additional_args=['-logFile', 'unity.log'])
    env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False, encode_obs=True,
                        wait_frames_num=0, vae_model_name=vae_model_name)

    # env = make_unity_env(env_path, 1, True, seed=env_seed, start_index=1, vae_model_name=vae_model_name)
    print(f'Gym environment {env_path} is created!')

    # channel_config = InputChannelConfig.RGB_ONLY
    # latent_dim = 1024
    # hidden_dims = [32, 64, 128, 256, 512, 1024]
    # vae_model_path = '/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder/models/vae-sim-rgb.pth'
    # vae_model = VAE(in_channels=channel_config.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
    # vae_model.eval()
    # vae_model.load_state_dict(torch.load(vae_model_path, map_location=torch.device('cpu')))
    # print(f'VAE model is loaded!')

    # model = PPO.load("terrain_rgb_ppo.zip")
    # model = PPO.load("terrain_rgb_ppo_action4d_bc_static.zip")
    # model = PPO.load("terrain_rgb_ppo_action4d_bc_dynamic_good.zip")
    # model = PPO.load(model_save_name + '.zip')
    model = PPO.load('circular_medium_ppo_seed_1_300k.zip')
    print(f'PPO model {model_save_name} is loaded!')

    obs = env.reset()


    # n_eval_episodes = 50
    #
    # rewards, episode_lens, traj_good, traj_bad = evaluate_policy(
    #     model.policy,  # type: ignore[arg-type]
    #     env,
    #     n_eval_episodes=n_eval_episodes,
    #     render=False,
    #     return_episode_rewards=True
    # )
    # print(rewards)
    # print(np.sum(rewards) / n_eval_episodes)

    step_reward = []
    episode_reward = []
    ep_id = 0
    while ep_id < 50:
        # recon_img = vae_model.decode(torch.Tensor(obs))[0].permute((1, 2, 0)).detach().numpy() * 255
        # ax_rgb.imshow(recon_img.astype(np.uint8))
        #
        # fig.canvas.draw()
        # fig.canvas.flush_events()

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        step_reward.append(rewards)
        # print(f'{info=}')
        if done:
            print(f'Done!')
            print(f'Episode length {len(step_reward)}, episode reward {np.sum(step_reward)}')
            episode_reward.append(np.sum(step_reward))
            obs = env.reset()
            step_reward = []
            ep_id += 1
            # break
        # time.sleep(0.5)
        # env.render()
    print(f'Episode reward mean: {np.mean(episode_reward)}, max: {np.max(episode_reward)}, min: {np.min(episode_reward)}')


def predict_bc():
    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder")
    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/utils")
    from vae import VAE
    from dataset import InputChannelConfig
    import torch
    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure()
    ax_rgb = fig.add_subplot(111)

    env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    # env_path = None
    env = make_unity_env(env_path, 1, True)
    print(f'Gym environment created!')

    channel_config = InputChannelConfig.RGB_ONLY
    latent_dim = 1024
    hidden_dims = [32, 64, 128, 256, 512, 1024]
    vae_model_path = '/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder/models/vae-sim-rgb.pth'
    vae_model = VAE(in_channels=channel_config.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
    vae_model.eval()
    vae_model.load_state_dict(torch.load(vae_model_path, map_location=torch.device('cpu')))
    print(f'VAE model is loaded!')

    bc_policy = bc.reconstruct_policy('/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/weight/BC_RGB_1000')
    print(f'bc policy is loaded!')

    print(f'Start predicting ...')
    obs = env.reset()
    episode_reward = []
    while True:
        recon_img = vae_model.decode(torch.Tensor(obs))[0].permute((1, 2, 0)).detach().numpy() * 255
        ax_rgb.imshow(recon_img.astype(np.uint8))

        fig.canvas.draw()
        fig.canvas.flush_events()

        action, _ = bc_policy.predict(obs)

        obs, reward, done, info = env.step(action)

        episode_reward.append(reward)
        print(f'{action=}, {reward=}, episode_reward {np.sum(episode_reward)}')
        # print(f'{info=}')
        # time.sleep(2)
        if done:
            print(f'Done')
            break
            obs = env.reset()


if __name__ == '__main__':
    train(seed=env_seed, use_callback=False)
    # predict()
    # predict_bc()
