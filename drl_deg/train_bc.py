import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.logger import configure
import bc
from train_utils import *


def rl_agent():
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        verbose=0,
        # seed=1,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
        # env_name=env_name
    )
    return expert


def train_expert(train_ep):
    expert = rl_agent()
    #expert.set_logger(new_logger)
    expert.learn(train_ep)  # Note: change this to 100000 to train a decent expert.
    return expert


if __name__ == "__main__":
    train_rl = False
    rl_name = "PPO"
    il_name = "BC"
    sample_good = False
    sample_bad = False
    train_il_good = True
    train_il_bad = True
    tmp_path = "/tmp/sb3_log/"
    env_name = "LunarLanderContinuous-v2" #"CartPole-v1" #   #"LunarLander-v2" #
    env = gym.make(env_name)
    seed = 1
    env.seed(seed)
    np.set_printoptions(suppress=True)
    train_ppo_ep = 450000
    train_il_ep = 50
    train_il_bad_ep = 1
    failure_steps = 10
    sample_ep = 5
    sample_bad_ep = 100
    n_eval_episodes = 20
    rng = np.random.default_rng(0)
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    expert = rl_agent()
    reward, _ = evaluate_policy(
        expert.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=n_eval_episodes,
        render=False,
    )
    print(f"0. PPO Reward before training: "+'{:.4f}'.format(reward))

    if sample_bad:
        print("1. Sampling bad transitions.")
        d_f = sample_expert_transitions(expert,env,rng,sample_bad_ep)
        save_to_csv("failure",env_name,d_f,failure_steps,seed=seed)
        d_f = read_csv("failure",env_name,seed = seed)

    else:
        # load trajectories
        print("1. Load bad transitions.")
        d_f = read_csv("failure",env_name)

    if train_rl:
        print("2. start training of the expert")
        expert = train_expert(train_ppo_ep)
        expert.save("weight/"+rl_name+"_"+env_name+"_"+str(train_ppo_ep))
    else:
        print("2. load the trained agent")
        #breakpoint()
        expert = PPO.load("weight/"+rl_name+"_"+env_name+"_"+str(train_ppo_ep), env=env,observation_space=env.observation_space, action_space=env.action_space)

    reward, _ = evaluate_policy(
        expert.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=n_eval_episodes,
        render=False,
    )

    print(f"3. Reward of the trained PPO: "+'{:.4f}'.format(reward))
    dataset = "success"

    if sample_good:
        # transitions.obs/acts/next_obs/dones
        print("4. Sampling good transitions.")
        d_s = sample_expert_transitions(expert,env,rng,sample_ep)
        save_to_csv(dataset,env_name,d_s)
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=d_s,
            rng=rng,
        )
        d_s = read_csv(dataset,env_name)

    else:
        # load trajectories
        print("4. Load good transitions.")
        d_s = read_csv(dataset,env_name)
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=d_s,
            rng=rng,
        )

    # init IL agent

    # evaluate before training
    reward, _ = evaluate_policy( bc_trainer.policy, env, n_eval_episodes=n_eval_episodes, render=False,)
    print(f"5. Reward of IL before training: "+'{:.4f}'.format(reward))

    # training or loading
    if train_il_good:
        print("6. Training a policy using Behavior Cloning")
        bc_trainer.train(n_epochs=train_il_ep)
        bc_trainer.save_policy("weight/"+il_name + "_" + env_name + "_" + str(train_il_ep))
        # evaluate the trained agent
        reward, _ = evaluate_policy(
            bc_trainer.policy,  # type: ignore[arg-type]
            env,
            n_eval_episodes=n_eval_episodes,
            render=False,
        )
        bc_trainer_bad = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=d_f,
            policy=bc_trainer.policy,
            rng=rng,
        )
    else:
        print("6. Load a policy of Behavior Cloning")
        #breakpoint()
        bc_trainer = bc.reconstruct_policy("weight/"+il_name + "_" + env_name + "_" + str(train_il_ep))
        # evaluate the trained agent
        reward, _ = evaluate_policy( bc_trainer, env, n_eval_episodes=n_eval_episodes, render=False,)
        # Load the bc policy
        bc_trainer_bad = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=d_f,
            policy=bc_trainer,
            rng=rng,
        )

    print(f"7. Reward of IL after training using good demonstration: "+'{:.4f}'.format(reward))

    # Train from Failure
    bc_trainer_bad.success = False
    if train_il_bad:
        print("8. Training a policy using bad demonstrations")
        bc_trainer_bad.train(n_epochs=train_il_bad_ep)
        bc_trainer_bad.save_policy("weight/"+il_name + "_bad_" + env_name + "_" + str(train_il_ep))
        # evaluate the trained agent
        reward, _ = evaluate_policy( bc_trainer_bad.policy, env,n_eval_episodes=n_eval_episodes, render=True)
    else:
        print("8. Loading a policy using bad demonstrations")
        bc_trainer_bad = bc.reconstruct_policy("weight/"+il_name + "_bad_" + env_name + "_" + str(train_il_ep))
        # evaluate the trained agent
        reward, _ = evaluate_policy(bc_trainer_bad, env, n_eval_episodes=n_eval_episodes, render=True)
    print(f"9. Reward after training: "+'{:.4f}'.format(reward))
