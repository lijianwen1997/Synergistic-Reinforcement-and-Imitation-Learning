"""
This is a training code for cloning the behavior of an expert for unity rivierine env.
"""

import gym

from stable_baselines3.common.logger import configure

import sril.bc as bc
from sril.train_utils import *

if __name__ == "__main__":
    il_name = "BC"
    encode_action = False
    train_il_good = True
    tmp_path = "/tmp/sb3_log/"
    env_name = "unity_riverine"
    np.set_printoptions(suppress=True)

    train_il_ep = 50
    n_eval_episodes = 20
    rng = np.random.default_rng(0)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    dataset = "RGB"
    observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1024,), dtype=np.float32)
    encode_str = ""
    action_space = gym.spaces.MultiDiscrete([3, 3, 3, 3])
    d_s = read_csv("success", env_name, seed="", test_type="merge")

    if train_il_good is True:
        bc_trainer = bc.BC(
            observation_space=observation_space,
            action_space=action_space,
            demonstrations=d_s,
            rng=rng,
            verbose=True,
        )
        bc_trainer.train(n_epochs=train_il_ep)
        bc_trainer.save_policy("sril/weight/" + il_name + "_medium_" + dataset + "_" + str(train_il_ep))
        bc_policy = bc_trainer.policy
    else:
        bc_policy = bc.reconstruct_policy("sril/weight/" + il_name + "_UnityRiverine_temp_" + str(train_il_ep))

    for _ in range(1):
        cnt = 0
        if encode_action:
            for i in range(d_s.acts.shape[0]):
                prediction, _ = bc_policy.predict(d_s.obs[i])

                # breakpoint()
                if np.array_equal(prediction, d_s.acts[i, 0]):
                    cnt += 1
        else:
            for i in range(d_s.acts.shape[0]):
                prediction, _ = bc_policy.predict(d_s.obs[i])

                # breakpoint()
                if np.array_equal(prediction, d_s.acts[i]):
                    cnt += 1
        print("Success rate: ", cnt / d_s.acts.shape[0])
