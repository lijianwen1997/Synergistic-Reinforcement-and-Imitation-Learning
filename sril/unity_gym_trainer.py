import time

from stable_baselines3.common.logger import configure

import sys
sys.path.append("../riverine_simulation")
from ppo import PPO
from bc import *
from callback import UpdateExpertCallback
from evaluation import evaluate_policy

from env_utils import make_unity_env


# Modifiable values
train_rl = True
tmp_path = "/tmp/sb3_log/"
vae_model_name = 'vae-sim-rgb-all.pth'
tb_log_dir = './ppo_river_tensorboard/'
np.set_printoptions(suppress=True)
train_ppo_ep = 140000
check_freq = int(train_ppo_ep / 20)  # check callback 1o times
rng = np.random.default_rng(0)
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


def train():
    env_path = '../riverine_simulation/unity_riverine_envs/riverine_training_env/riverine_training_env.x86_64'
    seeds = [1, 2, 3, 4]
    use_bc = [1, 1, 1, 1]   # if use bc expert
    sril = [1, 1, 1, 0]      # if dynamically update expert
    for index in range(len(seeds)):
        callback = UpdateExpertCallback(check_freq=check_freq,
                                        log_dir=tmp_path,
                                        verbose=1,
                                        seed=seeds[index],
                                        env_name='unity_riverine',
                                        reward_threshold=5) if use_bc[index] else None
        env = make_unity_env(env_path, 1, False, seed=seeds[index], start_index=1, vae_model_name=vae_model_name)
        print(f'Unity env is created!')
        if sril[index]:
            model_type = "_dynamic_bc"
        elif use_bc[index]:
            model_type = "_static_bc"
        else:
            model_type = "_no_bc"

        tb_log_name = 'riverine_training_' + str(seeds[index]) + model_type
        model_save_name = 'riverine_training_' + str(seeds[index]) + '_140k' + model_type

        if train_rl:
            model = PPO("MlpPolicy",
                        env,
                        n_steps=1024,
                        batch_size=64,
                        n_epochs=10,
                        verbose=1,
                        tensorboard_log=tb_log_dir,
                        env_name='unity_riverine')
        else:
            model = PPO.load(model_save_name + '.zip', env)

        model.use_action_loss = use_bc[index]
        model.learn(total_timesteps=train_ppo_ep, progress_bar=True, tb_log_name=tb_log_name, callback=callback)
        model.save(model_save_name)
        print(f'Model {model_save_name} has been trained and saved!')

        env.close()
        time.sleep(10)  # wait for unity window to close completely


if __name__ == '__main__':
    train()
