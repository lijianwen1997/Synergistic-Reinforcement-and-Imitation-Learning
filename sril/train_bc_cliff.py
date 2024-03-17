"""
This is a training code for cloning the behavior of an expert for Cliff Circular env.
"""
from sril.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
import sril.bc as bc
from sril.train_utils import *
from cliff_circular.cliffcircular_gym import *
if __name__ == "__main__":
    """
    Train IL agent in gym CircularCliff environment.
    """
    il_name = "BC"
    train_il = False
    render_mode = None  # or 'human'
    tmp_path = "/tmp/sb3_log/"
    seed = 0

    env_name = 'CliffCircular-gym-v0'
    env = gym.make(env_name, seed=seed)
    check_env(env)

    np.set_printoptions(suppress=True)
    train_il_ep = 50
    n_eval_episodes = 50
    rng = np.random.default_rng(seed=seed)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    dataset = 'success'

    # load trajectories
    print("Load good transitions.")
    d_s = read_csv(dataset, env_name, seed=seed, test_type='merge')
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=d_s,
        batch_size=8,
        rng=rng,
        verbose=1
    )

    # training or loading
    if train_il:
        print("Training a policy using Behavior Cloning")
        bc_trainer.train(n_epochs=train_il_ep, progress_bar=True)
        bc_trainer.save_policy("sril/weight/" + il_name + "_" + env_name + "_temp_" + str(train_il_ep))
        # evaluate the trained agent
        reward,  epoch_reward, _ = evaluate_policy(
            bc_trainer.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=n_eval_episodes,
            render=False,
            reward_threshold=18,
            env_name=env_name
        )
    else:
        print("Load a policy of Behavior Cloning")
        model_dir = "sril/weight/" + il_name + "_" + env_name + "_" + str(train_il_ep)
        bc_test = bc.reconstruct_policy(model_dir)
        # evaluate the trained agent
        reward, epoch_reward, _ = evaluate_policy(bc_test, env, n_eval_episodes=n_eval_episodes, render=False,
                                                  reward_threshold=18, env_name=env_name)
        print(reward, epoch_reward)

    print(f"Reward of IL after training using good demonstration: "+'{:.4f}'.format(reward))
