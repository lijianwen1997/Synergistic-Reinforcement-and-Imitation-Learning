import csv
import os
import os.path as osp

import gymnasium as gym

from cliff_circular.cliffcircular import CliffCircularEnv  # for env registering
from riverine_simulation.key2action import Key2Action


if __name__ == '__main__':
    max_episodes = 10
    log_dir = 'demo'
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make("CliffCircular-v0", render_mode='human')
    observation, info = env.reset()
    done = False
    key2action = Key2Action()

    for ep in range(max_episodes):
        demo_path = osp.join(log_dir, f'episode_{ep}.csv')
        if osp.exists(demo_path):
            print(f'{demo_path} already exist!')
            break

        print(f'Collecting demo in {demo_path} ...')
        while not done:
            action = key2action.get_cliff_circular_action()
            while action is None:  # wait for any meaningful action
                action = key2action.get_cliff_circular_action()

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            with open(demo_path, 'a+', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow([observation, action, reward, 1 if done else 0])

            observation = next_observation
        # reset env
        observation, info = env.reset()
        done = False
        print(f'Episode {ep} demo collection finished!')

    print(f'All {max_episodes} episodes demo collection finished!')
