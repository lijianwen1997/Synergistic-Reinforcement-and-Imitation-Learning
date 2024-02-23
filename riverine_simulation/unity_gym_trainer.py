import time

from stable_baselines3.common.logger import configure

from drl_deg.ppo import PPO
from drl_deg.bc import *
from drl_deg.callback import UpdateExpertCallback
from drl_deg.evaluation import evaluate_policy

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

vae_model_name = 'vae-sim-rgb-all.pth'

tb_log_dir = './ppo_river_tensorboard/'

np.set_printoptions(suppress=True)
train_ppo_ep = 140000
train_il_ep = 50
train_il_bad_ep = 1
failure_steps = 5
sample_ep = 5
sample_bad_ep = 100
check_freq = int(train_ppo_ep / 10)  # check callback 1o times
rng = np.random.default_rng(0)
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


def train():
    env_path = 'unity_riverine_envs/riverine_training_env/riverine_training_env.x86_64'
    seeds = [1, 2, 3, 4]
    use_bc = [1, 1, 1, 1]   # if use bc expert
    deg = [1, 1, 1, 0]      # if dynamically update expert
    for index in range(len(seeds)):
        callback = UpdateExpertCallback(check_freq=check_freq, log_dir=tmp_path,
                                        verbose=1,seed=seeds[index]) if use_bc[index] else None
        env = make_unity_env(env_path, 1, True, seed=seeds[index], start_index=1, vae_model_name=vae_model_name)
        print(f'Unity env is created!')
        if deg[index]:
            model_type = "_dynamic_bc"
        elif use_bc[index]:
            model_type = "_static_bc"
        else:
            model_type = "_no_bc"

        tb_log_name = 'riverine_training_' + str(seeds[index]) + model_type
        model_save_name = 'riverine_training_' + str(seeds[index]) + '_140k' + model_type

        if train_rl:
            model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=10, verbose=1,
                    tensorboard_log=tb_log_dir)
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
