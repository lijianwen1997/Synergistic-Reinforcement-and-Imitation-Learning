import csv
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from mlagents_envs.environment import UnityEnvironment
from unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel, EngineConfig
from agent_configuration_channel import AgentConfigurationChannel, AgentConfig
from key2action import Key2Action
from env_utils import make_unity_env

from encoder.vae import VAE
from encoder.dataset import InputChannelConfig

from drl_deg.train_utils import *


env_path = 'unity_riverine_envs/riverine_training_env/riverine_training_env.x86_64'
# env_path = 'unity_riverine_envs/riverine_testing_env/riverine_testing_env_new.x86_64'
# env_path = None  # de-annotate if want to interact with Unity Editor directly

width, height = 1024, 1024  # change as you wish

channel_env = EnvironmentParametersChannel()
channel_env.set_float_parameter("simulation_mode", 1.0)

channel_eng = EngineConfigurationChannel()
channel_eng.set_configuration_parameters(width=width, height=height, quality_level=1, time_scale=1,
                                         target_frame_rate=None, capture_frame_rate=None)

channel_agent = AgentConfigurationChannel()
channel_agent.set_configuration_parameters(max_idle_steps=5000)  # increase idle threshold to allow human input interval

env_seed = 1  # Unity environment seed

vae_model_name = 'vae-sim-rgb-all.pth'  # use RGB image for VAE input

unity_env = UnityEnvironment(file_name=env_path, no_graphics=False, seed=env_seed,
                             side_channels=[channel_env, channel_eng, channel_agent],
                             additional_args=['-logFile', 'unity.log'])
env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False, encode_obs=False, wait_frames_num=0)

obs = env.reset()
print(f'{obs.shape=}')
print(f'{env.action_space=}')

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

save_fig = False  # whether to save figures and csv
record_bad = False  # set to False to manually record good demos, otherwise random actions

trajectory_path = 'trajectories/training'
trajectory_good_path = trajectory_path + '/good'
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


if __name__ == '__main__':
    i = 1  # total steps during demo collection
    episode_steps = 0  # steps in current episode
    max_episode_steps = 500  # step limit of episode
    while i < 1000000:
        # get next action either manually or randomly
        action = k2a.get_multi_discrete_action()  # no action if no keyboard input
        # action = k2a.get_discrete_action()
        if record_bad and is_mask_sync:
            action = k2a.get_random_action()
            is_mask_sync = False

        # step once, if done, get ready to save stuff to the new folders/file
        obs, reward, done, info = env.step(action)

        if not np.all(np.array(action) == 1):
            i += 1
            episode_steps += 1
            print(f'EpStep: {episode_steps}, action: {action}, reward: {reward}, done: {done}')

        # save image-mask-action to csv only when action is not all 0
        if save_fig and obs_path is not None and mask_path is not None and any(a != 1 for a in action):
            with open(csv_path, 'a+', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([obs_path, action, reward, 1 if done else 0])
            print(f'Saved {i}th img-action-reward-done tuple to {csv_path}.')

        if done or episode_steps == max_episode_steps:
            if done:
                print(f'Done, episode len: {episode_steps}!')
            else:
                print(f'Collected {max_episode_steps} steps!')
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