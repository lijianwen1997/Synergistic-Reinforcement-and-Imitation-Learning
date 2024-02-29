from contextlib import closing
from io import StringIO
from os import path
from typing import Optional, Union, List, Dict, Tuple
import numpy as np

import gym
from gym import Env, spaces
from gym.error import DependencyNotInstalled
from gym.envs.registration import register
from gym.utils import seeding

try:
    import pygame
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gymnasium[toy-text]`"
    ) from e

NOOP = 0  # no_operation
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4


register(
     id="CliffCircular-gym-v0",
     entry_point="cliff_circular:CliffCircularGymEnv",
     max_episode_steps=128,
     kwargs={
         'render_mode': None,
         'extra_cliff_num': 2,
         'reset_extra_cliff_on_episode_begin': True,
         'obs_side_len': 5,
         'obs_include_loc': False,
         'punish_meaningless_action': False,
         'boost_finish_reward': False,
         'seed': 0,
     }
)


def categorical_sample(prob_n, np_random: np.random.Generator):
    """Sample from categorical distribution where each row specifies class probabilities."""
    assert np_random is not None, f'numpy rng is None!'
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return np.argmax(csprob_n > np_random.random())


class CliffCircularGymEnv(Env):
    """
    Gym version

    Cliff circular involves navigating in a gridworld circularly while avoiding falling off a cliff.

    ## Description
    The game starts with the player at a random location of the 12x12 grid world.
    If the player circulates the gridworld the episode ends.

    A cliff runs along gridworld boundary, as well as the center area. If the player moves to a cliff location it
    will be spawned randomly in a non-cliff grid.

    The player makes moves until they run around the gridworld.

    The grid world data structure is 2D numpy.array, where row number is array x dimension, column number
    is array y dimension.

    At most 2 extra cliff can be added to the gridworld with fixed cliff locations. Whether the extra cliff locations
    need to be randomly re-generated on episode begin can also be toggled on env creation.

    Adapted from [Gym CliffWalking](https://www.gymlibrary.dev/environments/toy_text/cliff_walking/)
    environment, which is adapted from Example 6.6 (page 132) from Reinforcement Learning: An Introduction
    by Sutton and Barto [<a href="#cliffwalk_ref">1</a>].

    With inspiration from:
    [https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py)

    ## Action Space
    The action shape is `(1,)` in the range `{0, 4}` indicating
    which direction to move the player.

    - 0: No operation
    - 1: Move up
    - 2: Move right
    - 3: Move down
    - 4: Move left

    Note this environment allows no_op movement, meaning standing still is also a valid movement for the agent.
    This makes the agent behavior more realistic considering future involvement of multiple agents, in which condition
    not moving can be an appropriate option in either cooperative or combating game.

    ## Observation Space
    The flattened 5 x 5 grids around agent's current location (row major, top to bottom, MultiBinary)
    indicating whether there is cliff or not.

    If agent's current location is included in the observation, this scalar value representing the player's
    current position as current_row * nrows + current_col (where both the row and col start at 0) will be
    the first element preceding the above dim-25 observation, resulting in 26-dim MultiDiscrete([144]+[2]*25)
    observation.

    ## Starting State
    The episode starts with the player in random non-cliff grid, for example, `[26]` (location [2, 2]).

    ## Reward
    The reward is calculated based on the number of newly visited central track grids, which are the central layer
    squared grids in all 3 valid layers of grids.
    In this 12 by 12 grid world, there are 20 central grids.
    Agent's current grid will be projected onto this central track to update visitedness.
    Any new visit of a central grid will incur +1 reward.
    If 'punish_meaningless_action' argument is toggled on, the occurrence of repeated visit of a central track
    grid will incur -1 reward, if no new central grids are visited in this step.
    If the agent stepped into the cliff, -100 reward will be incurred.

    ## Episode End
    The episode terminates when the agent has visited all central grids, or has stepped into any cliff grid.

    ## Information

    `step()` and `reset()` return a dict with the following keys:
    - "p" - transition probability for the state.

    As cliff circular is not stochastic, the transition probability returned is always 1.0.

    ## Arguments

    'render_mode': None, or 'human', or 'ansi', or 'rgb_array'
    'extra_cliff_num': 2, choose from {0, 1, 2}
    'reset_extra_cliff_on_episode_begin': True,
    'obs_side_len': 5,
    'obs_include_loc': False,
    'punish_meaningless_action': False,
    'boost_finish_reward': False, add +100 reward if the agent circulates gridworld successfully

    ```python
    import gymnasium as gym
    gym.make('CliffCircular-v0')
    ```

    ## References
    <a id="cliffwalk_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
    An Introduction” 2020. [Online]. Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)

    ## Version History
    - v0: Initial version release

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    def __init__(self,
                 render_mode: Optional[str] = None,
                 extra_cliff_num: int = 2,
                 reset_extra_cliff_on_episode_begin: bool = True,
                 obs_side_len: int = 5,
                 obs_include_loc: bool = False,
                 punish_meaningless_action: bool = True,
                 boost_finish_reward: bool = False,
                 seed: int = 0,
                 ):
        self.shape = (12, 12)  # the grid env size is fixed

        self.np_random, _ = seeding.np_random(seed)  # default rng

        # Initial state is not fixed, this state is only for game display
        self.start_state_index = None

        self.nS = np.prod(self.shape)
        self.nA = 5

        # Cliff Location
        self.extra_cliff_num: int = extra_cliff_num
        assert 0 <= self.extra_cliff_num <= 2, f'Invalid number {self.extra_cliff_num=}'
        self.reset_extra_cliff = reset_extra_cliff_on_episode_begin
        if self.extra_cliff_num == 0 and self.reset_extra_cliff:
            print(f'No need to reset cliff locations if there is no extra cliff!')
            self.reset_extra_cliff = False

        self._cliff = np.zeros(self.shape, dtype=bool)
        # fill surrounding and center grids as cliff
        self._cliff[:2, :] = True
        self._cliff[10:12, :] = True
        self._cliff[:, 0:2] = True
        self._cliff[:, 10:12] = True
        self._cliff[5: 7, 5: 7] = True
        self._original_cliff = self._cliff.copy()
        self._cliff_num: int = sum(np.array(self._cliff, dtype=int).flatten())
        self._safe_num: int = self.nS - self._cliff_num

        # Calculate initial state distribution
        self.initial_state_distrib = self._update_init_state_distribution()
        self.original_initial_state_distribution = self.initial_state_distrib.copy()

        # Update random cliff locations on episode begin
        self._update_extra_cliff_loc()
        self._cliff_num += self.extra_cliff_num
        self._safe_num -= self.extra_cliff_num
        self.initial_state_distrib = self._update_init_state_distribution()
        print(f'Total cliff number: {self._cliff_num}, safe number: {self._safe_num}')

        # Central track location
        self._central_track = []
        for j in range(3, 9):
            self._central_track.append(np.ravel_multi_index((3, j), self.shape))
        for i in range(4, 9):
            self._central_track.append(np.ravel_multi_index((i, 8), self.shape))
        for j in range(7, 2, -1):
            self._central_track.append(np.ravel_multi_index((8, j), self.shape))
        for i in range(7, 3, -1):
            self._central_track.append(np.ravel_multi_index((i, 3), self.shape))
        print(f'Central track grids: {self._central_track}')

        # Init central track visitedness
        self._track_visited_count: Dict[int, int] = {}
        for i in self._central_track:
            self._track_visited_count[i] = 0
        self._punish_visit_count_thr = 2
        # print(f'Track visited count: {self._track_visited_count}')

        # Calculate transition probabilities and rewards
        self.P = {}
        self._update_transition_prob()

        # Keep record of reset count of states
        self._state_reset_count: Dict[int, int] = {}
        self._reset_count = 0
        # Buffered mapping from observation tuple to the corresponding state(s)
        self.obs2state: Dict[Tuple[int, ...], List[int]] = {}

        self.obs_include_loc = obs_include_loc
        self.obs_side_len = obs_side_len
        assert self.obs_side_len > 0, f'Observation side length needs to be positive, given {self.obs_side_len}'
        assert self.obs_side_len <= 5, f'Observation side length needs to be smaller than 5, given {self.obs_side_len}'
        assert self.obs_side_len % 2 == 1, f'Observation side length needs to be odd, given {self.obs_side_len}'

        if self.obs_include_loc:
            self.observation_space = spaces.MultiDiscrete([self.nS] + [2] * self.obs_side_len * self.obs_side_len)
        else:
            self.observation_space = spaces.MultiBinary(self.obs_side_len * self.obs_side_len)

        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode

        self.punish_meaningless_action = punish_meaningless_action

        self.boost_finish_reward = boost_finish_reward

        self.ep_steps = 0

        assert 0 < self.obs_side_len < min(self.shape), f'Observation side length {self.obs_side_len} is invalid!'

        # pygame utils
        self.cell_size = (100, 100)
        self.window_size = (
            self.shape[1] * self.cell_size[1],
            self.shape[0] * self.cell_size[0],
        )
        self.window_surface = None
        self.clock = None
        self.elf_images = None
        self.start_img = None
        self.goal_img = None
        self.cliff_img = None
        self.mountain_bg_img = None
        self.near_cliff_img = None
        self.tree_img = None

    def _update_init_state_distribution(self) -> np.array:
        """
        Uniform initial state distribution according to number of valid grids
        Mainly used after the extra cliff locations are re-generated
        """
        initial_state_distribution = np.zeros(self.nS)
        uniform_init_prob = 1.0 / self._safe_num
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if not self._cliff[position]:
                initial_state_distribution[np.ravel_multi_index(position, self.shape)] = uniform_init_prob
        return initial_state_distribution

    def _update_extra_cliff_loc(self):
        """
        Randomly generate extra cliff locations
        """
        self._cliff = self._original_cliff.copy()  # reset cliff locations
        extra_cliff_states = []
        for _ in range(self.extra_cliff_num):
            cliff_state = categorical_sample(self.original_initial_state_distribution, self.np_random)
            while cliff_state in extra_cliff_states:
                cliff_state = categorical_sample(self.original_initial_state_distribution, self.np_random)
            extra_cliff_states.append(cliff_state)
            cliff_pos = np.unravel_index(cliff_state, self.shape)
            self._cliff[cliff_pos] = True

    def _update_transition_prob(self):
        """
        Mainly for storing the state transition dict with the next location after various actions for fast step query
        """
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][NOOP] = self._calculate_transition_prob(position, [0, 0])
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

    def _calc_dist(self, idx1: int, idx2: int):
        """
        Calculate manhattan distance between 2 grids
        """
        x1, y1 = np.unravel_index(idx1, self.shape)
        x2, y2 = np.unravel_index(idx2, self.shape)
        return abs(x1 - x2) + abs(y1 - y2)

    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """
        Does not have any meaning in this environment, since the episode will be reset if the agent steps into cliff
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _check_state_validity(self, state: Union[int, Tuple[int, int]]) -> bool:
        """
        For manual state reset, check whether the desired agent state is valid
        """
        pos = np.unravel_index(state, self.shape) if isinstance(state, int) else state
        if pos[0] < 0 or pos[0] > self.shape[0] - 1:
            return False
        if pos[1] < 0 or pos[1] > self.shape[1] - 1:
            return False
        if self._cliff[pos]:
            return False
        return True

    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        The termination flag indicates whether agent fell off cliff
        Args:
            current: Current position on the grid as (row, col)
            delta: Change in position for transition

        Returns:
            Tuple of ``(1.0, new_state, reward, terminated)``
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, new_state, -100, True)]
        return [(1.0, new_state, -1, False)]

    def step(self, a):
        assert NOOP <= a <= LEFT, f'Step action {a} is invalid!'

        # update next state by action
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.lastaction = a
        self.s = s

        # calculate observation and reward after action
        if not t:
            obs = self._calc_obs()
            self.update_obs_state_buffer(tuple(obs))
            # overwrite the original reward if not terminated
            r = self._calc_reward()
            t = self._completed()  # judge episode done after updated visitedness
            # give large final reward if agent successfully follows the circular track
            if self.boost_finish_reward and t:
                r += 100
        else:
            obs = self.observation_space.sample()  # observation at cliff is not the actual observation
            # reward is -100 if fell off cliff

        self.ep_steps += 1

        if self.render_mode is not None:
            self.render()

        return obs, r, bool(t), {"prob": p}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # reset extra cliff locations on episode begin
        if self.reset_extra_cliff:
            self._update_extra_cliff_loc()
            self.initial_state_distrib = self._update_init_state_distribution()  # for random reset
            self._update_transition_prob()  # for reward/termination judgement

        self.ep_steps = 0

        # reset agent in the extra cliff added env
        # sample reset state from initial state distribution or external assignment
        if options and 'state' in options:
            state = options['state']
            if self._check_state_validity(state):
                print(f'Manual reset!')
                self.s = state
            else:
                raise ValueError(f'Externally assigned state {state} is invalid!')
        else:
            # print(f'Random reset!')
            self.s = categorical_sample(self.initial_state_distrib, self.np_random)

        # print(f'Reset agent to {self.s}')

        self.start_state_index = self.s  # start_state_index is not used for this environment
        self.lastaction = None

        obs = self._calc_obs()
        # update mapping from observation to state
        self.update_obs_state_buffer(tuple(obs))

        # update state reset count
        self._reset_count += 1
        if self.s not in self._state_reset_count:
            self._state_reset_count[self.s] = 0
        else:
            self._state_reset_count[self.s] += 1

        # reset visit count of central track
        for k, _ in self._track_visited_count.items():
            self._track_visited_count[k] = 0

        # update visit count for the initial state
        # self.update_visit_count(self._project(self.s))

        if self.render_mode is not None:
            self.render()

        return obs

    def update_obs_state_buffer(self, obs: Tuple[int, ...]):
        """
        Update mapping from observation to agent state, for statistical purpose
        """
        if self.obs_include_loc:
            assert obs[0] == self.s, f'The first element in observation should be current agent state!'

        if obs not in self.obs2state:
            self.obs2state[obs] = [self.s]
        elif self.s not in self.obs2state[obs]:
            self.obs2state[obs].append(self.s)

    def state_reset_count(self) -> Dict[int, int]:
        """
        Return the dict that maps state to the number of reset counts at this state, for statistical purpose
        """
        return self._state_reset_count

    def set_state_reset_count(self, item: Optional[Dict[int, int]] = None):
        """
        Reset state-to-reset_count dict, for statistical purpose
        """
        if item is None:
            self._state_reset_count = {}
        else:
            print(f'Does not support external state reset count update!')

    def get_state_from_obs(self, obs: Union[Tuple[int, ...], np.ndarray]) -> Union[List[int], None]:
        """
        Get state from the corresponding observation, for manual state reset
        """
        obs_tuple = tuple(obs) if isinstance(obs, np.ndarray) else obs
        if obs_tuple in self.obs2state:
            return self.obs2state[obs_tuple]
        else:
            print(f'Observation {obs_tuple} does not exist in the current buffer!')
            print(f'{self.obs2state=}')
            return None

    def _calc_obs(self) -> np.array:
        """
        Calculate the 5x5 grids surrounding agent's current grid, return as a flattened array.
        """
        assert self.s is not None, 'Need to call reset first!'
        cur_x, cur_y = np.unravel_index(self.s, self.shape)
        half_obs_side_len = self.obs_side_len // 2
        surround_positions = np.array(self._cliff[cur_x - half_obs_side_len: cur_x + half_obs_side_len + 1,
                                      cur_y - half_obs_side_len: cur_y + half_obs_side_len + 1], dtype=np.int8).flatten()
        if self.obs_include_loc:
            return np.concatenate((np.array([self.s]), surround_positions))
        else:
            return surround_positions

    def _update_visit_count(self, central_track_grids: List[int]):
        """
        Update state visit count of central track grids
        """
        for grid in central_track_grids:
            assert grid in self._track_visited_count.keys(), f'Central track grid {grid} is invalid!'
            self._track_visited_count[grid] += 1

    def _project(self, index: int) -> List[int]:
        """
        Project any grid to the central track grids and return the nearest central track grids list
        """
        nearest_track_ids = []
        diagonal_idx = None
        for track_index in self._central_track:
            cur_dist = self._calc_dist(index, track_index)
            if cur_dist == 0:  # central track grids
                nearest_track_ids = [track_index]
                break
            if cur_dist == 1:
                nearest_track_ids.append(track_index)
            if cur_dist == 2:  # 4 outer corner grids
                diagonal_idx = track_index
        nearest_track_ids = nearest_track_ids if len(nearest_track_ids) > 0 else [diagonal_idx]
        # print(f'Current grid: {index}, projected central grids: {nearest_track_ids}')

        if len(nearest_track_ids) == 1:
            return nearest_track_ids
        elif len(nearest_track_ids) == 2:
            list_index1 = self._central_track.index(nearest_track_ids[0])
            list_index2 = self._central_track.index(nearest_track_ids[1])
            list_index_small = min(list_index1, list_index2)
            list_index_large = max(list_index1, list_index2)
            projected_track_indices = []
            if list_index_small == 1:
                projected_track_indices += [self._central_track[list_index_small - 1],
                                                 self._central_track[list_index_small],
                                                 self._central_track[list_index_large]]
            else:
                for i in range(list_index_small, list_index_large + 1):
                    projected_track_indices.append(self._central_track[i])
            assert len(
                projected_track_indices) == 3, f'Track visited number is wrong, {projected_track_indices=}!'
            return projected_track_indices
        else:
            print(f'Length greater than 2 projected track grids is not supported: {nearest_track_ids=}')
            raise NotImplementedError

    def _calc_reward(self) -> float:
        """
        Calculate the number of newly visited central track grids according to agent's current location, reward equals
        this number, ranges from 0 to 3, inclusively.
        Need to update visit count first!
        """
        track_indices = self._project(self.s)
        self._update_visit_count(track_indices)
        assert 0 < len(track_indices) <= 3, f'Track indices {track_indices} are invalid!'

        reward = 0
        have_new_visit = False
        have_over_visit = False
        for track_idx in track_indices:
            assert self._track_visited_count[track_idx] != 0, f'Need to update track visit count before getting reward!'

            if self._track_visited_count[track_idx] == 1:
                reward += 1  # increment reward by number of newly visited central track grids
                have_new_visit = True
            elif self._track_visited_count[track_idx] > self._punish_visit_count_thr:
                have_over_visit = True

        # punish meaningless action if required
        if self.punish_meaningless_action and not have_new_visit and have_over_visit:
            reward = -1  # fixed value punishment for any number of overly visited grids

        return reward

    def _completed(self) -> bool:
        """
        Whether all central track grids have been visited
        """
        return np.all(np.array(list(self._track_visited_count.values())) > 0)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            print(self._render_text())
        else:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        if self.window_surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CliffCircular")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.elf_images is None:
            hikers = [
                path.join(path.dirname(__file__), "img/elf_up.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(path.dirname(__file__), "img/mountain_bg1.png"),
                path.join(path.dirname(__file__), "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(path.dirname(__file__), "img/mountain_near-cliff1.png"),
                path.join(path.dirname(__file__), "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(path.dirname(__file__), "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nS):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            # if row < self.shape[0] - 1 and self._cliff[row + 1, col]:
            #     self.window_surface.blit(self.near_cliff_img[check_board_mask], pos)
            if s == self.start_state_index:
                self.window_surface.blit(self.start_img, pos)
            # if s == self.nS - 1:
            #     self.window_surface.blit(self.goal_img, pos)
            if s == self.s:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else DOWN
                last_action = DOWN - 1 if last_action == NOOP else last_action - 1
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _render_text(self):
        outfile = StringIO()

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            # elif position == (3, 11):
            #     output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()

# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# All other assets by ____


if __name__ == "__main__":
    # Play with CliffCircular environment with keyboard
    env = gym.make("CliffCircular-gym-v0", render_mode='human')
    # env = gym.make("CliffCircular-v0", render_mode='ansi')

    # Manual reset agent state/location
    # initial_state = 11
    # observation, info = env.reset(options={'state': initial_state})

    # Random reset
    observation, info = env.reset()

    manual_control: bool = True
    key2action = None
    if manual_control:
        from riverine_simulation.key2action import Key2Action
        key2action = Key2Action()

    steps = 0
    ep_rew = 0
    max_steps = 1000

    for i in range(max_steps):
        if key2action:
            action = key2action.get_cliff_circular_action()
            while action is None:  # wait for any meaningful action
                action = key2action.get_cliff_circular_action()
        else:
            action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        ep_rew += reward
        print(f'reward: {reward}')

        if terminated or truncated:
            print(f'Done! Steps: {steps}, episode reward: {ep_rew}')
            steps = 0
            ep_rew = 0
            observation, info = env.reset()

    print(f'Max steps {max_steps} reached, closing environment ...')
    env.close()
