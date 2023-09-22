
from mlagents_envs.environment import UnityEnvironment
from unity_gym_env import UnityToGymWrapper

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from imitation.data.wrappers import RolloutInfoWrapper

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def make_unity_env(env_directory, num_env, visual, seed, encode_obs=True, start_index=0, vae_model_name=''):
    """
    Create a wrapped, monitored Unity environment.
    """
    def make_env(rank, seed, use_visual=True):  # pylint: disable=C0111
        def _thunk():
            unity_env = UnityEnvironment(env_directory, base_port=5000 + rank, seed=seed)
            env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False, allow_multiple_obs=False,
                                    encode_obs=encode_obs, vae_model_name=vae_model_name)
            # new_logger = configure("/tmp/unity_sb3_ppo_log/", ["stdout", "csv", "tensorboard"])
            # env = Monitor(env, filename=new_logger.get_dir())
            # env = Monitor(env)
            # env = RolloutInfoWrapper(env)
            return env

        return _thunk

    if visual:
        return SubprocVecEnv([make_env(i + start_index, seed) for i in range(num_env)])
    else:
        rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        return DummyVecEnv([make_env(rank, use_visual=False, seed=seed)])
