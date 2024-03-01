from evaluation import evaluate_policy

from train_utils import *

from stable_baselines3.common.callbacks import BaseCallback

import bc as bc
import shutil


class UpdateExpertCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1, seed=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.reward_threshold = 4
        self.reward = 0
        self.reward_max = 10
        self.seed = seed
        self.env_name = "unity_river"

        directory = "./trajectory/" + self.env_name + "/success"
        csv_file = directory + "/transitions_deg" + str(seed) + ".csv"
        csv_file_demo = directory + "/transitions_merge.csv"

        if not os.path.exists(csv_file):
            try:
                shutil.copyfile(csv_file_demo, csv_file)
                print(f"File '{csv_file}' created by copying from '{csv_file_demo}'.")
            except Exception as e:
                print(f"Error creating file: {e}")
        weight_directory = "./weight/expert/"
        if not os.path.exists(weight_directory):
            os.makedirs(weight_directory)
            print(f"Folder '{weight_directory}' created successfully.")
        else:
            print(f"Folder '{weight_directory}' already exists.")

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:
            print("+++++++++++++")
            print("callback", self.n_calls)

            self.model.save(
                "weight/mirl/" + "PPO_MEDIUM_" + self.env_name + '_' + str(self.n_calls) + '_seed_' + str(
                    self.seed) + "_deg")

            print("evaluate PPO")
            reward, ep_reward, traj_good, _ = evaluate_policy(
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
            print("PPO reward", np.mean(ep_reward), ep_reward)
            '''
            print("evaluate BC")
            self.reward, ep_reward, _, _ = evaluate_policy(
                self.model.bc,  # type: ignore[arg-type]
                self.model.env,
                n_eval_episodes=5,
                render=False,
            )
            print("BC reward", ep_reward)
            self.reward_threshold = max(self.reward,self.reward_threshold)
            '''

            if len(traj_good) > 0:
                print("New good traj")
                for i in range(len(traj_good)):
                    append_to_csv("success", self.env_name, traj_good[i],seed=self.seed,test_type="deg")
                self.model.new_success = True
                self.model.train_IL = True

            print("reward_threshold", self.reward_threshold)

            self.model.n_calls = self.n_calls
            ## Don't use bc at the begining
            # if self.n_calls / self.check_freq<4.0:
            #     self.model.train_IL = False
            if self.model.train_IL:
                rng = np.random.default_rng(0)

                if self.model.new_success:
                    d_s = read_csv("success", self.env_name, seed=self.seed, test_type="deg")

                    bc_trainer = bc.BC(
                        observation_space=self.model.env.observation_space,
                        action_space=self.model.env.action_space,
                        demonstrations=d_s,
                        policy=self.model.bc,
                        rng=rng,
                    )
                    bc_trainer.success = True
                    bc_trainer.train(n_epochs=20)
                    self.new_success = False
                    bc_trainer.save_policy("weight/expert/" + "BC_MEDIUM_" + self.env_name + "_" + str(self.n_calls) +
                                           '_seed_' + str(self.seed)+"_deg")
                    self.model.bc = bc_trainer.policy
                    print("Finish BC training")

                print("+++++++++++++")

        return True

