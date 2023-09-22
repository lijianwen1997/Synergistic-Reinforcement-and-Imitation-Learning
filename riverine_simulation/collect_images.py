import os
import matplotlib.pyplot as plt

from mlagents_envs.environment import UnityEnvironment
from unity_gym_env import UnityToGymWrapper
from agent_configuration_channel import AgentConfigurationChannel


if __name__ == '__main__':
    riverine_map = 'training'  # or 'testing'
    env_path = f'unity_riverine_envs/riverine_{riverine_map}_env/riverine_{riverine_map}_env.x86_64'
    assert os.path.exists(env_path), f'{env_path} does not exist!'

    img_save_dir = 'sim_images'
    os.makedirs(img_save_dir, exist_ok=True)
    print(f'Saving images to {img_save_dir} ...')

    max_img_num = 2000
    wait_frame_num = 0

    channel_agent = AgentConfigurationChannel()  # can modify agent configs as you need

    unity_env = UnityEnvironment(file_name=env_path, no_graphics=False, seed=1,
                                 side_channels=[channel_agent],
                                 additional_args=['-logFile', 'unity.log'])
    env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False,
                            encode_obs=False, wait_frames_num=wait_frame_num)
    print(f'Env loaded!')

    i = -1
    while (i := i + 1) < max_img_num:
        env.reset()
        obs, _, _, _ = env.step([1] * 4)  # no action to make view stable
        # print(f'{obs.shape=}')
        obs_path = os.path.join(img_save_dir, ('%04d' % i) + '.jpg')
        plt.imsave(obs_path, obs)
        # time.sleep(0.1)

    print(f'All images saved!')
