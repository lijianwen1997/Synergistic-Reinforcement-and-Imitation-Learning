import time

from stable_baselines3.common.logger import configure

from ppo import PPO
from bc import *
from callback import UpdateExpertCallback
from evaluation import evaluate_policy



# Modifiable values
env_seed = 1
train_rl = True
rl_name = "PPO"
il_name = "BC"
sample_good = False
sample_bad = False
train_il_good = False
train_il_bad = False

import sys
sys.path.append("../cliff_circular")
from cliffcircular_gym import CliffCircularGymEnv


tmp_path = "/tmp/sb3_log/"


tb_log_dir = './ppo_cliff_tensorboard/'

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
    env_name = 'CliffCircular-gym-v0'

    seeds = [1, 2, 3, 4, 5]*3
    use_bc = [0]*5+[1]*10 # if use bc expert
    deg = [0]*10+[1]*5     # if dynamically update expert
    print(seeds, use_bc, deg)
    for index in range(10, 15):
        env = gym.make(env_name, seed=seeds[index])

        if deg[index]:
            model_type = "_dynamic_bc"

        elif use_bc[index]:
            model_type = "_static_bc"
        else:
            model_type = "_no_bc"

        callback = UpdateExpertCallback(check_freq=check_freq, log_dir=tmp_path,
                                        verbose=1, seed=seeds[index], env_name=env_name, reward_threshold=20) if use_bc[index] else None
        tb_log_name = 'cliff_training_' + str(seeds[index]) + model_type
        model_save_name = 'cliff_training_' + str(seeds[index]) + '_140k' + model_type

        model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=10, verbose=0, env_name=env_name,
                    tensorboard_log=tb_log_dir)
        model.use_action_loss = use_bc[index]

        model.learn(total_timesteps=train_ppo_ep, progress_bar=False, tb_log_name=tb_log_name, callback=callback)

        model.save(model_save_name)
        print(f'Model {model_save_name} has been trained and saved!')
        env.close()


if __name__ == '__main__':
    train()
