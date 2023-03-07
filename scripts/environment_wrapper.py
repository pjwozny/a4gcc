from pathlib import Path
import numpy as np
from gym.spaces import Box, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from scripts.run_unittests import import_class_from_path
from rice import Rice

BIG_NUMBER = 1e20


class EnvWrapper(MultiAgentEnv):
    """
    The environment wrapper class.
    """

    def __init__(self, env_config: dict):
        assert isinstance(env_config, dict)

        env_config_copy = env_config.copy()

        if "source_dir" in env_config_copy:
            source_dir = Path(env_config_copy["source_dir"])
            rice_class = import_class_from_path("Rice", source_dir / "rice.py")
            del env_config_copy["source_dir"]
        else:
            rice_class = Rice

        self.env = rice_class(**env_config_copy)

        self.action_space = self.env.action_space

        self.observation_space = self.recursive_obs_dict_to_spaces_dict(
            self.env.reset()
        )

    def reset(self):
        """Reset the env."""
        obs = self.env.reset()
        return self.recursive_list_to_np_array(obs)

    def step(self, action_dict):
        """Step through the env."""
        assert isinstance(action_dict, dict)
        obs, rew, done, info = self.env.step(action_dict)
        return self.recursive_list_to_np_array(obs), rew, done, info

    def recursive_obs_dict_to_spaces_dict(self, obs):
        """Recursively return the observation space dictionary
        for a dictionary of observations

        Args:
            obs (dict): A dictionary of observations keyed by agent index
            for a multi-agent environment

        Returns:
            spaces.Dict: A dictionary of observation spaces
        """
        assert isinstance(obs, dict)
        dict_of_spaces = {}
        for key, val in obs.items():

            # list of lists are 'listified' np arrays
            _val = val
            if isinstance(val, list):
                _val = np.array(val)
            elif isinstance(val, (int, np.integer, float, np.floating)):
                _val = np.array([val])

            # assign Space
            if isinstance(_val, np.ndarray):
                large_num = float(BIG_NUMBER)
                box = Box(
                    low=-large_num, high=large_num, shape=_val.shape, dtype=_val.dtype
                )
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                # This loop avoids issues with overflow to make sure low/high are good.
                while not low_high_valid:
                    large_num = large_num // 2
                    box = Box(
                        low=-large_num,
                        high=large_num,
                        shape=_val.shape,
                        dtype=_val.dtype,
                    )
                    low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                dict_of_spaces[key] = box

            elif isinstance(_val, dict):
                dict_of_spaces[key] = self.recursive_obs_dict_to_spaces_dict(_val)
            else:
                raise TypeError
        return Dict(dict_of_spaces)

    def recursive_list_to_np_array(self, dictionary: dict):
        """
        Numpy-ify dictionary object to be used with RLlib.
        """
        assert isinstance(dictionary, dict)

        new_d = {}
        for key, val in dictionary.items():
            if isinstance(val, list):
                new_d[key] = np.array(val)
            elif isinstance(val, dict):
                new_d[key] = self.recursive_list_to_np_array(val)
            elif isinstance(val, (int, np.integer, float, np.floating)):
                new_d[key] = np.array([val])
            elif isinstance(val, np.ndarray):
                new_d[key] = val
            else:
                raise AssertionError
        return new_d
