"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""

import gym

from stable_baselines3.common.logger import configure

import bc
from train_utils import *

if __name__ == "__main__":
    il_name = "BC"
    encode_action = False
    train_il_good = True
    train_il_bad = True
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
    #d_s = read_csv_unity()
    d_s = read_csv("success", env_name, seed="", test_type="merge")
    d_f = read_csv("failure", env_name, seed="", test_type="merge")
    #save_to_csv("success",env_name,d_s)
    # breakpoint()

    if train_il_good is True:
        bc_trainer = bc.BC(
            observation_space=observation_space,
            action_space=action_space,
            demonstrations=d_s,
            rng=rng,
            verbose=True,
        )
        bc_trainer.train(n_epochs=train_il_ep)
        bc_trainer.save_policy("weight/" + il_name + "_medium_" + dataset + "_" + str(train_il_ep))
        bc_policy = bc_trainer.policy
    else:
        bc_policy = bc.reconstruct_policy("weight/" + il_name + "_medium_" + dataset + "_" + str(train_il_ep))

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

    if train_il_bad is True:
        bc_trainer_bad = bc.BC(
            observation_space=observation_space,
            action_space=action_space,
            demonstrations=d_s,
            policy=bc_policy,
            rng=rng,
        )
        bc_trainer_bad.success = False
        bc_trainer_bad.train(n_epochs=5)
        bc_trainer_bad.save_policy("weight/" + il_name + "_medium_mix_" + dataset + "_" + str(train_il_ep))
        bc_policy = bc_trainer_bad.policy

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
        # no encode 0.98, 0.92,0.91
        # encode 0.76  0.73  0.70

    if True:
        bc_trainer = bc.BC(
            observation_space=observation_space,
            action_space=action_space,
            demonstrations=d_s,
            rng=rng,
            policy=bc_policy,
            verbose=True,
        )
        bc_trainer.success = True

        bc_trainer.train(n_epochs=50)
        bc_trainer.save_policy("weight/" + il_name + "_medium_retrain_" + dataset + "_" + str(train_il_ep))
        bc_policy = bc_trainer.policy
    else:
        bc_policy = bc.reconstruct_policy("weight/" + il_name + "_easy_retrain_" + dataset + "_" + str(train_il_ep))

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
        # no encode 0.98, 0.92,0.91
        # encode 0.76  0.73  0.70


