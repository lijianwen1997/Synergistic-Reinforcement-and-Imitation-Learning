"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""

import gym
import numpy as np
from ppo import PPO
#from stable_baselines3 import PPO, TD3

from evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.logger import configure

from train_utils import *

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import bc
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from imitation.data.types import Transitions


class UpdateExpertCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1, seed=0, mirl = False, use_bad=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.reward_threshold = 4
        self.reward = 0
        self.reward_max = 10
        self.seed = seed
        self.env_name = "unity_river"
        self.mirl = mirl
        self.use_bad = use_bad


    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:
            print("+++++++++++++")
            print("callback", self.n_calls)
            if self.use_bad:
                test_type = "mix"
            else:
                test_type = "good"
            if self.mirl == False:
                test_type = "static"

            self.model.save(
                "weight/mirl/" + "PPO_MEDIUM_" + self.env_name + '_' + str(self.n_calls) + '_seed_' + str(
                    self.seed) + test_type)
            if self.mirl:

                print("evaluate PPO")
                reward, ep_reward, traj_good, traj_bad = evaluate_policy(
                    self.model.policy,  # type: ignore[arg-type]
                    self.model.env,
                    n_eval_episodes=20,
                    render=False,
                    reward_threshold=self.reward_threshold
                )
                if np.mean(ep_reward) > 0.5:
                    self.model.action_eff = 0.2
                    print("+"*20)
                    print("Update action loss weight!!")
                print("PPO reward",np.mean(ep_reward), ep_reward)
                # print("evaluate BC")
                # self.reward, ep_reward, _, _ = evaluate_policy(
                #     self.model.bc,  # type: ignore[arg-type]
                #     self.model.env,
                #     n_eval_episodes=5,
                #     render=False,
                # )
                # print("BC reward", ep_reward)
                # #self.reward_threshold = max(self.reward,self.reward_threshold)
                if len(traj_good) > 0:
                    print("New good traj")
                    for i in range(len(traj_good)):
                        append_to_csv("success", self.env_name, traj_good[i],seed=self.seed,test_type=test_type)
                    self.model.new_success = True
                    self.model.train_IL = True
                if len(traj_bad) > 0 and self.use_bad:
                    print("New bad traj")
                    for i in range(len(traj_bad)):
                        append_to_csv("failure", self.env_name, traj_bad[i], seed=self.seed,test_type=test_type)
                    self.model.new_failure = True
                    self.model.train_IL = True
                print("reward_threshold", self.reward_threshold)

                self.model.n_calls = self.n_calls

                # if self.n_calls / self.check_freq<4.0:
                #     self.model.train_IL = False
                if self.model.train_IL:
                    rng = np.random.default_rng(0)
                    # if self.model.new_failure and self.use_bad:
                    #     print("new bad, start training")
                    #
                    #     d_f = read_csv("failure", self.env_name + '_',seed=self.seed,test_type=test_type)
                    #     bc_trainer = bc.BC(
                    #         observation_space=self.model.env.observation_space,
                    #         action_space=self.model.env.action_space,
                    #         demonstrations=d_f,
                    #         policy=self.model.bc,
                    #         rng=rng,
                    #     )
                    #     bc_trainer.success = False
                    #
                    #     bc_trainer.train(n_epochs=5)
                    #     self.model.bc = bc_trainer.policy

                    if self.model.new_success:

                        # breakpoint()
                        print("new good")

                        # print("Here")
                        d_s = read_csv("success", self.env_name + '_',seed=self.seed,test_type=test_type)

                        bc_trainer = bc.BC(
                            observation_space=self.model.env.observation_space,
                            action_space=self.model.env.action_space,
                            demonstrations=d_s,
                            policy=self.model.bc,
                            rng=rng,
                        )
                        bc_trainer.success = True
                        if self.model.new_failure:
                            bc_trainer.train(n_epochs=30)
                        else:
                            bc_trainer.train(n_epochs=20)
                        self.new_failure = False
                        self.new_success = False
                        bc_trainer.save_policy("weight/trained/" + "BC_MEDIUM_" + self.env_name + "_" + str(self.n_calls)+'_seed_' + str(
                        self.seed)+test_type)

                        self.model.bc = bc_trainer.policy


                        print("Finish BC training")

                print("+++++++++++++")

        return True


if __name__ == "__main__":
    train_rl = True
    use_callback = False
    rl_name = "PPO"
    il_name = "BC"
    tmp_path = "/tmp/sb3_log/"
    env_name = "LunarLanderContinuous-v2" #"CartPole-v1" #"LunarLander-v2" # "CartPole-v1" #

    seeds = [0,1,2,3,4]

    train_ppo_ep = 500000
    train_il_ep = 50
    train_il_bad_ep = 1
    failure_steps = 10
    sample_ep = 5
    check_freq = int(train_ppo_ep/20)


    rng = np.random.default_rng(0)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    np.set_printoptions(suppress=True)
    for i in range(len(seeds)):
        env = gym.make(env_name)
        env.seed(seeds[i])
        #callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        expert = PPO(
            policy=MlpPolicy,
            env=env,
            seed= seeds[i],
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64,
            tensorboard_log="./ppo_" + env_name + "_tensorboard/",  # ppo_cartpole_tensorboard
        )
        if use_callback:
            callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=tmp_path, verbose=1)
        else:
            callback = None

        if train_rl:
            print("Start training of the expert")
            expert.learn(train_ppo_ep, tb_log_name="train_PPOIL_seed_"+str(seeds[i]),callback=callback)  # Note: change this to 100000 to train a decent expert.

        else:
            print("load the trained agent")
            expert = PPO.load("weight/"+rl_name+"_"+env_name+"_"+str(train_ppo_ep), env=env, observation_space=env.observation_space, action_space=env.action_space)

        reward, _, traj_good, traj_bad = evaluate_policy(
            expert.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=10,
            render=True,
        )


        print(f"Reward of PPO: {reward}")

