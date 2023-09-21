import csv
import os
import sys
import io
import torch
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel, EngineConfig
from mlagents_envs.side_channel.agent_configuration_channel import AgentConfigurationChannel, AgentConfig
from mlagents_envs.side_channel.segmentation_receiver_channel import SegmentationReceiverChannel
from mlagents_envs.side_channel.rgb_receiver_channel import RGBReceiverChannel
from mlagents_envs.key2action import Key2Action
from env_utils import make_unity_env

sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder")
sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/utils")
from vae import VAE
from dataset import InputChannelConfig

import gym

import numpy as np
# from ppo import PPO
from stable_baselines3 import PPO, TD3
from imitation.util import util

import bc
from train_utils import *


#### Unity env absolute path ####
env_path = None  # require Unity Editor to be running
# env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
# env_path = '/home/edison/Terrain/circular_river_medium/circular_river_medium.x86_64'
# env_path = '/home/edison/Terrain/circular_river_medium_configurable/circular_river_medium_configurable.x86_64'
# env_path = '/home/edison/Terrain/circular_river_easy/circular_river_easy.x86_64'
# env_path = '/home/edison/Terrain/circular_river_hard/circular_river_hard.x86_64'
# env_path = '/home/edison/River/mlagent-ram-seg.x86_64'
# env_path = '/home/edison/River/mlagent-ram-test2.x86_64'
# env_path = '/home/edison/River/mlagent-ram-4D.x86_64'
# env_path = '/home/edison/Research/ml-agents/Visual3DBall.x86_64'
# env_path = '/home/edison/TestAgent/testball.x86_64'
# env_path = '/home/edison/RollerBall/SlidingCube.x86_64'

width, height = 640, 480

channel_env = EnvironmentParametersChannel()
channel_env.set_float_parameter("simulation_mode", 1.0)

channel_eng = EngineConfigurationChannel()
channel_eng.set_configuration_parameters(width=width, height=height, quality_level=1, time_scale=1,
                                         target_frame_rate=None, capture_frame_rate=None)
# channel_eng.set_configuration(EngineConfig.default_config())

channel_agent = AgentConfigurationChannel()
# channel_agent.set_configuration(AgentConfig.default_config())
channel_agent.set_configuration_parameters(max_idle_steps=500)

channel_seg = SegmentationReceiverChannel()
channel_rgb = RGBReceiverChannel()

env_seed = 1
# vae_model_name = 'vae-sim-rgb-medium.pth'
# vae_model_name = 'vae-sim-rgb-easy.pth'
vae_model_name = 'vae-sim-rgb-all.pth'

unity_env = UnityEnvironment(file_name=env_path, no_graphics=False, seed=env_seed,
                             # side_channels=[channel_env, channel_eng, channel_seg, channel_rgb],
                             side_channels=[channel_env, channel_eng, channel_agent],
                             # side_channels=[channel_float],
                             additional_args=['-logFile', 'unity.log'])
env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False, encode_obs=False,
                        wait_frames_num=0, vae_model_name=vae_model_name)
# env = make_unity_env(env_path, 1, True, env_seed, encode_obs=False, vae_model_name=vae_model_name)
obs = env.reset()
# print(f'{env.observation_space=}')
print(f'{env.action_space=}')
print(f'{obs.shape=}')

vae_model = env.vae_model

# plt.ion()
# fig = plt.figure()
# ax_rgb = fig.add_subplot(121)
# ax_mask = fig.add_subplot(122)

last_mask = None
mask_show = None
last_obs = None
is_mask_sync = False
cur_sync_frame_num = 0
min_sync_frame_num = 5

save_fig = False  # whether save figures and csv
record_bad = False  # set to False to manually record good demos
get_bc_act = False  # whether output the action prediction of BC, both multi-discrete and one-hot multi-discrete

# load NN models for VAE encoding and IL
mode = 'sim'  # or 'real' or 'both'
channel_config = InputChannelConfig.RGB_ONLY  # or 'MASK_ONLY' or 'RGB_MASK'

if get_bc_act:
    il_model_path = '/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/weight/BC_RGB_500'
    print(f'{il_model_path=}')
    bc_policy = bc.reconstruct_policy(il_model_path)
    print(f'IL model is loaded!')

# trajectory_path = 'trajectories/easy'
trajectory_path = 'trajectories/medium'
trajectory_good_path = trajectory_path + '/good_new'
trajectory_bad_path = trajectory_path + '/bad'
demo_id = 0
print(f'Current demo: {demo_id}')
demo_name = f'demo{demo_id}'
demo_path = os.path.join(trajectory_bad_path if record_bad else trajectory_good_path, demo_name)
if save_fig:
    os.makedirs(demo_path, exist_ok=True)
csv_path = os.path.join(demo_path, 'traj_random.csv')
img_dir = os.path.join(demo_path, 'images')
mask_dir = os.path.join(demo_path, 'masks')
if save_fig:
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

obs_path = None
mask_path = None
waited_frames = 0
wait_frames_num = 5

k2a = Key2Action()  # start a new thread
if record_bad:
    k2a.listener.stop()  # stop capturing keyboard input if using random actions
# k2a.listener.stop()


def update_demo_path():
    global demo_name, demo_path, csv_path, img_dir, mask_dir
    print(f'Current demo: {demo_id}')
    demo_name = f'demo{demo_id}'
    demo_path = os.path.join(trajectory_bad_path if record_bad else trajectory_good_path, demo_name)
    if save_fig:
        os.makedirs(demo_path, exist_ok=True)

    csv_path = os.path.join(demo_path, 'traj_random.csv')

    img_dir = os.path.join(demo_path, 'images')
    mask_dir = os.path.join(demo_path, 'masks')
    if save_fig:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)


# define one-hot multi-discrete actions as human expert demonstrates
actions = np.ones([9, 4])
actions[1, 3] = 0  # up
actions[2, 3] = 2  # down
actions[3, 2] = 0  # l r
actions[4, 2] = 2  # r r
actions[5, 1] = 0  # f
actions[6, 1] = 2  # b
actions[7, 0] = 0  # l
actions[8, 0] = 2  # r


if __name__ == '__main__':
    i = 1
    episode_steps = 0
    max_episode_steps = 100
    while i < 1000000:
        # get next action either manually or randomly
        action = k2a.get_multi_discrete_action()  # no action if no keyboard input
        # action = k2a.get_discrete_action()
        if record_bad and is_mask_sync:
            action = k2a.get_random_action()
            is_mask_sync = False

        # predict action using nn model
        if vae_model is not None and get_bc_act:
            if channel_config == InputChannelConfig.MASK_ONLY:
                obs = torch.Tensor(mask_show).permute((2, 0, 1))[0].unsqueeze(0).unsqueeze(0)
            elif channel_config == InputChannelConfig.RGB_ONLY:
                obs = torch.Tensor(obs).permute((2, 0, 1)).unsqueeze(0) / 255.0
            # print(f'{obs.shape=}')
            encoding = vae_model.encode(obs)[0][0].to("cuda:0")
            acts = util.safe_to_tensor(actions).to("cuda:0")
            _, log_prob, entropy = bc_policy.evaluate_actions(encoding.unsqueeze(0), acts)
            print(f'{log_prob=}')
            action_il = log_prob.cpu().detach().numpy().argmax()
            action_int, _ = bc_policy.predict(encoding.cpu().detach().numpy())

            is_mask_sync = False
            print(f'IL: {action_il=}, KEY: {action=}, PRED: {action_int=}')

        # step once, if done, get ready to save stuff to the new folders/file
        obs, reward, done, info = env.step(action)


        # save image-mask-action to csv only when action is not all 0
        if save_fig and obs_path is not None and mask_path is not None and any(a != 1 for a in action):
            with open(csv_path, 'a+', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([obs_path, action, reward, 1 if done else 0])
            i += 1
            episode_steps += 1
            print(f'Saved {i}th img-action-reward-done tuple to {csv_path}.')

        if done or episode_steps == max_episode_steps:
            if done:
                print('Done!')
            else:
                print(f'Collected {max_episode_steps} steps!')
            # break
            episode_steps = 0
            env.reset()
            demo_id += 1
            if save_fig:
                update_demo_path()
            is_mask_sync = False
            cur_sync_frame_num = 0
            continue

        # mask = channel_seg.get_segmentation_mask()
        # rgb = channel_rgb.get_rgb()

        # if mask is None:
        #     print(f'Mask is not ready!')
        #     env.reset()
        #     mask_path = None
        #     continue

        # if mask != last_mask:
        #     mask_stream = io.BytesIO(mask)
        #     mask_show = mpimg.imread(mask_stream, format='png')
        #     # if save_fig:
        #     #     mask_path = os.path.join(mask_dir, f'{i}.png')
        #     #     plt.imsave(mask_path, mask_show)  # mask figure is over-written only when it changes
        #     last_mask = mask
        #     is_mask_sync = False
        # else:
        #     cur_sync_frame_num += 1
        #     # print(f'Synced count {cur_sync_frame_num}')
        #     if cur_sync_frame_num >= min_sync_frame_num:
        #         print(f'Mask is fully synced!')
        #         cur_sync_frame_num = 0
        #         is_mask_sync = True

        if save_fig:
            obs_path = os.path.join(img_dir, ('%04d' % i) + '.jpg')
            plt.imsave(obs_path, obs)  # observation figure changes frequently, so over-written every step
            mask_path = os.path.join(mask_dir, ('%04d' % i) + '.png')
            # plt.imsave(mask_path, mask_show)

        # rgb_stream = io.BytesIO(rgb)
        # rgb_show = mpimg.imread(rgb_stream)

        # mask_stream = io.BytesIO(mask)
        # mask_show = mpimg.imread(mask_stream, format='png')

        # ax_rgb.imshow(obs)
        # ax_rgb.imshow(rgb_show)
        # ax_mask.imshow(mask_show)

        # fig.canvas.draw()
        # fig.canvas.flush_events()