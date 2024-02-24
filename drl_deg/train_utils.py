import os.path
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import Transitions

import csv
import re


def sample_expert_transitions(expert, env, rng, episodes=40):
    # generate a dataset
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=episodes),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)


def save_to_file(data, file_path):
    try:
        with open(file_path, "ab") as handle:
            np.savetxt(handle, data, fmt="%s")
    except FileNotFoundError:
        with open(file_path, "wb") as handle:
            np.savetxt(handle, data, fmt="%s")


def save_to_csv(dataset, env_name, transitions, failure_steps=20, test_type="mix", seed=0):
    directory = "./trajectory/" + env_name + "/" + dataset
    if not os.path.exists(directory):
        os.makedirs(directory)
    csv_columns = ['obs', 'acts', 'infos', 'next_obs', 'dones']
    csv_file = directory + "/transitions_"+test_type + str(seed)+".csv"
    np.set_printoptions(threshold=np.inf, linewidth=np.nan)

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        if dataset == "success":
            for data in transitions:
                writer.writerow(data)
        else:
            for i in range(len(transitions)-failure_steps):
                if transitions.dones[i+failure_steps-1] == True:
                    for j in range (failure_steps):
                        writer.writerow(transitions[i+j])


def append_to_csv(dataset, env_name, data, seed=0, test_type="mix"):
    directory = "./trajectory/" + env_name + "/" + dataset
    csv_columns = ['obs', 'acts', 'infos', 'next_obs', 'dones']
    csv_file = directory + "/transitions_"+test_type + str(seed)+".csv"
    if not os.path.exists(csv_file):
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        fp = open(csv_file, 'x')
        fp.close()
    np.set_printoptions(threshold=np.inf, linewidth=np.nan)
    with open(csv_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        #writer.writeheader()
        #for data in data:
        #    writer.writerow(data)
        for trans in data:
            writer.writerow(trans)


def read_csv(dataset, env_name, seed=1, test_type="mix"):
    states = []
    actions = []
    infos = []
    next_states = []
    dones = []
    directory = "./trajectory/" + env_name + "/" + dataset
    csv_file = directory + "/transitions_"+test_type + str(seed)+".csv"
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            #print(line)
            state_str = re.findall(r"[-+]?(?:\d*\.*\d+)", line[0])
            states.append([float(state) for state in state_str])
            actions_str = re.findall(r"[-+]?(?:\d*\.*\d+)", line[1])
            actions.append([float(action) for action in actions_str])
            infos.append({})  # eval(line[2])
            next_state_str = re.findall(r"[-+]?(?:\d*\.*\d+)", line[3])
            next_states.append([float(state) for state in next_state_str])
            dones.append(eval(line[4]))
        #breakpoint()
        #breakpoint()
        good_steps = 50*100
        bad_steps = 1000
        if dataset == "success":
            states = states[-good_steps:]
            actions = actions[-good_steps:]
            infos = infos[-good_steps:]
            next_states = next_states[-good_steps:]
            dones = dones[-good_steps:]
        else:
            states = states[-bad_steps:]
            actions = actions[-bad_steps:]
            infos = infos[-bad_steps:]
            next_states = next_states[-bad_steps:]
            dones = dones[-bad_steps:]
        new_transitions = Transitions(np.array(states), np.array(actions), np.array(infos),
                                       np.array(next_states), np.array(dones))

    return new_transitions


def read_csv_unity():
    states = []
    actions = []
    infos = []
    next_states = []
    dones = []
    for i in range(0, 84):
        if i != 17 and i != 21 and i != 40 and i != 64 and i != 58 and i != 73:
            csv_file = os.path.join(os.path.dirname(__file__),
                                    # "./dataset/images/sim/UnityRiverDataset/medium_new/demo"+str(i)+"/RGB_ONLY.csv")
                                    "../riverine_simulation/demonstration/demo"+str(i)+"/RGB_ONLY.csv")
            print(csv_file)
            with open(csv_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  # skip the headers
                for line in reader:
                    #print(line)
                    state_str = re.findall(r"[-+]?(?:\d*\.*\d+)", line[0])
                    states.append([float(state) for state in state_str[:1024]])
                    actions_str = re.findall(r"[-+]?(?:\d*\.*\d+)", line[1])
                    action_list = [int(action) for action in actions_str]
                    actions.append(action_list)
                    infos.append({})
                    #next_state_str = re.findall(r"[-+]?(?:\d*\.*\d+)", line[3])
                    dones.append(False)

        next_states = states[1:]
        next_states.append(next_states[-1])
    print(len(states))
    new_transitions = Transitions(np.array(states), np.array(actions), np.array(infos),
                                  np.array(next_states), np.array(dones))

    return new_transitions


def read_csv_cliff_circular():
    states = []
    actions = []
    infos = []
    next_states = []
    dones = []
    demo_path = '../cliff_circular/demo'
    assert os.path.exists(demo_path), f'{demo_path} does not exist!'

    demo_files = [f for f in os.listdir(demo_path) if f.endswith('.csv')]
    # print(f'{demo_files=}')

    step = 0
    for f in demo_files:
        demo_file = os.path.join(demo_path, f)
        with open(demo_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            ep_step = 0
            for line in reader:
                ep_step += 1
                obs = line[0][1:-1]
                obs = [int(o) for o in obs.split(' ')]
                act = int(line[1])
                done = bool(int(line[3]))
                states.append(obs)
                actions.append(act)
                infos.append({})
                dones.append(done)

        assert ep_step > 3, f'Episode length should be at least 3, given {ep_step}!'
        next_states += states[step + 1: step + ep_step]
        next_states.append(next_states[-1])
        step += ep_step

    new_transitions = Transitions(np.array(states), np.array(actions), np.array(infos),
                                  np.array(next_states), np.array(dones))

    return new_transitions


if __name__ == '__main__':
    # test Unity riverine env demo reading
    # ds = read_csv_unity()
    # save_to_csv('success', 'unity_river', ds)
    # # print(f'{ds.obs=}')
    # print(f'{ds.acts=}')

    # test CliffCircular demo reading
    transitions = read_csv_cliff_circular()
    save_to_csv('success', 'CliffCircular-gym-v0', transitions, test_type='merge')

