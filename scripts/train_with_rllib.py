# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Training script for the rice environment using RLlib
https://docs.ray.io/en/latest/rllib-training.html
"""

from collections import defaultdict
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import json
import pickle as pkl

import numpy as np
from random import choice

import yaml
from train import get_rllib_config
from environment_wrapper import EnvWrapper
from desired_outputs import desired_outputs
from fixed_paths import PUBLIC_REPO_DIR

sys.path.append(PUBLIC_REPO_DIR)

# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.DEBUG)


def perform_other_imports():
    """
    RLlib-related imports.
    """
    import ray
    import torch
    from gym.spaces import Box, Dict
    if ray.__version__ == "1.0.0":
        from ray.rllib.agents.a3c import A2CTrainer
    else:
        from ray.rllib.algorithms.a2c import A2C as A2CTrainer
    # from ray.rllib.agents.a3c import A2CTrainer
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from ray.tune.logger import NoopLogger

    return ray, torch, Box, Dict, MultiAgentEnv, A2CTrainer, NoopLogger


try:
    other_imports = perform_other_imports()
except ImportError:
    print("Installing requirements...")

    # Install gym
    subprocess.call(["pip", "install", "gym==0.21.0"])
    # Install RLlib v1.0.0
    subprocess.call(["pip", "install", "ray[rllib]==1.0.0"])
    # Install PyTorch
    subprocess.call(["pip", "install", "torch==1.9.0"])

    other_imports = perform_other_imports()

ray, torch, Box, Dict, MultiAgentEnv, A2CTrainer, NoopLogger = other_imports

from torch_models import TorchLinear

_BIG_NUMBER = 1e20


def recursive_obs_dict_to_spaces_dict(obs):
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
            large_num = float(_BIG_NUMBER)
            box = Box(
                low=-large_num, high=large_num, shape=_val.shape, dtype=_val.dtype
            )
            low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            # This loop avoids issues with overflow to make sure low/high are good.
            while not low_high_valid:
                large_num = large_num // 2
                box = Box(
                    low=-large_num, high=large_num, shape=_val.shape, dtype=_val.dtype
                )
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            dict_of_spaces[key] = box

        elif isinstance(_val, dict):
            dict_of_spaces[key] = recursive_obs_dict_to_spaces_dict(_val)
        else:
            raise TypeError
    return Dict(dict_of_spaces)


def recursive_list_to_np_array(dictionary):
    """
    Numpy-ify dictionary object to be used with RLlib.
    """
    if isinstance(dictionary, dict):
        new_d = {}
        for key, val in dictionary.items():
            if isinstance(val, list):
                new_d[key] = np.array(val)
            elif isinstance(val, dict):
                new_d[key] = recursive_list_to_np_array(val)
            elif isinstance(val, (int, np.integer, float, np.floating)):
                new_d[key] = np.array([val])
            elif isinstance(val, np.ndarray):
                new_d[key] = val
            else:
                raise AssertionError
        return new_d
    raise AssertionError


def get_rllib_config_(exp_run_config=None, env_class=None, seed=None):
    """
    Reference: https://docs.ray.io/en/latest/rllib-training.html
    """

    assert exp_run_config is not None
    assert env_class is not None

    env_config = exp_run_config["env"]
    assert isinstance(env_config, dict)
    env_object = env_class(env_config=env_config)

    # Define all the policies here
    policy_config = exp_run_config["policy"]["regions"]

    # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
    # of (policy_cls, obs_space, act_space, config). This defines the
    # observation and action spaces of the policies and any extra config.
    policies = {
        "regions": (
            None,  # uses default policy
            env_object.observation_space[0],
            env_object.action_space[0],
            policy_config,
        ),
    }

    # Function mapping agent ids to policy ids.
    def policy_mapping_fn(agent_id=None):
        assert agent_id is not None
        return "regions"

    # Optional list of policies to train, or None for all policies.
    policies_to_train = None

    # Settings for Multi-Agent Environments
    multiagent_config = {
        "policies": policies,
        "policies_to_train": policies_to_train,
        "policy_mapping_fn": policy_mapping_fn,
    }

    train_config = exp_run_config["trainer"]
    rllib_config = {
        # Arguments dict passed to the env creator as an EnvContext object (which
        # is a dict plus the properties: num_workers, worker_index, vector_index,
        # and remote).
        "env_config": exp_run_config["env"],
        "framework": train_config["framework"],
        "multiagent": multiagent_config,
        "num_workers": train_config["num_workers"],
        "num_gpus": train_config["num_gpus"],
        "num_envs_per_worker": train_config["num_envs"] // train_config["num_workers"],
        "train_batch_size": train_config["train_batch_size"],
    }
    if seed is not None:
        rllib_config["seed"] = seed

    return rllib_config


def save_model_checkpoint(trainer_obj=None, save_directory=None, current_timestep=0):
    """
    Save trained model checkpoints.
    """
    assert trainer_obj is not None
    assert save_directory is not None
    assert os.path.exists(save_directory), (
        "Invalid folder path. "
        "Please specify a valid directory to save the checkpoints."
    )
    model_params = trainer_obj.get_weights()
    for policy in model_params:
        filepath = os.path.join(
            save_directory,
            f"{policy}_{current_timestep}.state_dict",
        )
        logging.info("Saving the model checkpoints for policy %s to %s.", policy, filepath)
        torch.save(model_params[policy], filepath)


def load_model_checkpoints(trainer_obj=None, save_directory=None, ckpt_idx=-1):
    """
    Load trained model checkpoints.
    """
    assert trainer_obj is not None
    assert save_directory is not None
    assert os.path.exists(save_directory), (
        "Invalid folder path. "
        "Please specify a valid directory to load the checkpoints from."
    )
    
    policy_name_files = defaultdict(list)
    for file in Path(save_directory).glob("*.state_dict"):
        policy_name = file.stem.split("_")[0]
        policy_name_files[policy_name].append(file)

    assert len(policy_name_files) == len(trainer_obj.config["multiagent"]["policies"])

    model_params = trainer_obj.get_weights()
    for policy_name in model_params:
        policy_models = policy_name_files[policy_name]
        # If there are multiple files, then use the ckpt_idx to specify the checkpoint
        assert ckpt_idx < len(policy_models)
        sorted_policy_models = sorted(policy_models, key=lambda x: int(x.stem.split("_")[-1]))
        policy_model_file = sorted_policy_models[ckpt_idx]
        model_params[policy_name] = torch.load(policy_model_file)
        logging.info(f"Loaded model checkpoints {policy_model_file}.")

    trainer_obj.set_weights(model_params)


def create_trainer(exp_run_config=None, source_dir=None, results_dir=None, seed=None):
    """
    Create the RLlib trainer.
    """
    assert exp_run_config is not None
    if results_dir is None:
        # Use the current time as the name for the results directory.
        results_dir = f"{time.time():10.0f}"

    # Directory to save model checkpoints and metrics

    save_config = exp_run_config["saving"]
    results_save_dir = os.path.join(
        save_config["basedir"],
        save_config["name"],
        save_config["tag"],
        results_dir,
    )

    ray.init(ignore_reinit_error=True)

    # Create the A2C trainer.
    exp_run_config["env"]["source_dir"] = source_dir
    rllib_trainer = A2CTrainer(
        env=EnvWrapper,
        config=get_rllib_config(
            exp_run_config, env_class=EnvWrapper, seed=seed
        ),
    )
    return rllib_trainer, results_save_dir


def part_of(state, global_state):
    if state not in global_state:
        print(f"WARNING: {state} not part of global state, removing from evaluation")
        return False
    return True


def fetch_episode_states(trainer_obj=None, episode_states=None):
    """
    Helper function to rollout the env and fetch env states for an episode.
    """
    assert trainer_obj is not None
    assert episode_states is not None
    assert isinstance(episode_states, list)
    assert len(episode_states) > 0

    outputs = {}

    # Fetch the env object from the trainer
    env_object: EnvWrapper = trainer_obj.workers.local_worker().env
    obs = env_object.reset()

    env = env_object.env

    episode_states = [s for s in episode_states if part_of(s, env.global_state)]

    for state in episode_states:
        assert state in env.global_state, f"{state} is not in global state!"
        # Initialize the episode states
        array_shape = env.global_state[state]["value"].shape
        outputs[state] = np.nan * np.ones(array_shape)

    agent_states = {}
    policy_ids = {}
    policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
    for region_id in range(env.num_regions):
        policy_ids[region_id] = policy_mapping_fn(region_id)
        agent_states[region_id] = trainer_obj.get_policy(
            policy_ids[region_id]
        ).get_initial_state()

    for timestep in range(env.episode_length):
        for state in episode_states:
            outputs[state][timestep] = env.global_state[state]["value"][timestep]

        actions = {}
        # TODO: Consider using the `compute_actions` (instead of `compute_action`)
        # API below for speed-up when there are many agents.
        for region_id in range(env.num_agents):
            if (
                len(agent_states[region_id]) == 0
            ):  # stateless, with a linear model, for example
                actions[region_id] = trainer_obj.compute_single_action(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )
            else:  # stateful
                (
                    actions[region_id],
                    agent_states[region_id],
                    _,
                ) = trainer_obj.compute_single_action(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )
        obs, _, done, _ = env_object.step(actions)
        if done["__all__"]:
            for state in episode_states:
                outputs[state][timestep + 1] = env.global_state[state]["value"][
                    timestep + 1
                ]
            break

    return outputs


def fetch_episode_states_freerider(trainer_obj=None, episode_states=None):
    """
    Helper function to rollout the env and fetch env states for an episode.
    """
    assert trainer_obj is not None
    assert episode_states is not None
    assert isinstance(episode_states, list)
    assert len(episode_states) > 0

    conditions = ["soft_defect","hard_defect", "cooperate", "control"]

    metrics = [{} for i in range(trainer_obj.workers.local_worker().env.env.episode_length)]

    for condition in conditions:

        outputs = {}

        # Fetch the env object from the trainer
        env_object = trainer_obj.workers.local_worker().env
        obs = env_object.reset()

        env = env_object.env
        episode_states = [s for s in episode_states if part_of(s, env.global_state)]
        
        #choose one agent to be the freerider
        fr_id = np.random.randint(0,env.num_agents) 

        for state in episode_states:
            assert state in env.global_state, f"{state} is not in global state!"
            # Initialize the episode states
            array_shape = env.global_state[state]["value"].shape
            outputs[state] = np.nan * np.ones(array_shape)

        agent_states = {}
        policy_ids = {}
        policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
        for region_id in range(env.num_regions):
            policy_ids[region_id] = policy_mapping_fn(region_id)
            agent_states[region_id] = trainer_obj.get_policy(
                policy_ids[region_id]
            ).get_initial_state()

        #get action offset index for defect action
        action_offset_index = len(
            env.savings_action_nvec
            + env.mitigation_rate_action_nvec
            + env.export_action_nvec
            + env.import_actions_nvec
            + env.tariff_actions_nvec
            + env.proposal_actions_nvec
            + env.negotiator.stages[1]["numberActions"]
        )
        num_defect_actions = len(env.negotiator.stages[2]["numberActions"])

        #mitigation offset
        mitigation_offset = len(env.savings_action_nvec)

        #action offset index for tariff actions
        tariff_offset = len(env.savings_action_nvec
            + env.mitigation_rate_action_nvec
            + env.export_action_nvec
            + env.import_actions_nvec)
        number_tariff_actions = len(env.tariff_actions_nvec)

        for timestep in range(env.episode_length):
            for state in episode_states:
                outputs[state][timestep] = env.global_state[state]["value"][timestep]

            actions = {}
            # TODO: Consider using the `compute_actions` (instead of `compute_action`)
            # API below for speed-up when there are many agents.
            for region_id in range(env.num_agents):
                if (
                    len(agent_states[region_id]) == 0
                ):  # stateless, with a linear model, for example
                    actions[region_id] = trainer_obj.compute_single_action(
                        obs[region_id],
                        agent_states[region_id],
                        policy_id=policy_ids[region_id],
                    )
                else:  # stateful
                    (
                        actions[region_id],
                        agent_states[region_id],
                        _,
                    ) = trainer_obj.compute_single_action(
                        obs[region_id],
                        agent_states[region_id],
                        policy_id=policy_ids[region_id],
                    )

                #agents defect, but can still mitigate to whatever degree the find 
                if condition == "soft_defect":

                    if region_id == fr_id:
                        #set region to defect regardless of previous decision.
                        actions[region_id][action_offset_index : action_offset_index + num_defect_actions] = 1
                #agent mitigates 0
                elif condition == "hard_defect":
                    if region_id == fr_id:
                        #set region to defect regardless of previous decision.
                        actions[region_id][action_offset_index : action_offset_index + num_defect_actions] = 1
                        #set agent mitigation to 0
                        actions[region_id][mitigation_offset] = 0

                #make no change for control group
                elif condition == "control":
                    pass

                elif condition == "cooperate":
                    if region_id == fr_id:
                        #set region to cooperate regardless of previous decision.
                        actions[region_id][action_offset_index : action_offset_index + num_defect_actions] = 0


            #get tariffs of agent
            average_fr_tariffs = np.mean([actions[region_id][tariff_offset+fr_id] for region_id in range(env.num_agents) if region_id !=fr_id])
            #get reward
            reward = env.get_global_state("reward_all_regions", timestep, fr_id)
            #get labor
            labor = env.get_global_state("labor_all_regions", timestep, fr_id)
            #get abatement cost
            abatement = env.get_global_state("abatement_cost_all_regions", timestep, fr_id)

            metrics[timestep] = {**{
                    f"{condition}_average_tariffs":float(average_fr_tariffs),
                    f"{condition}_reward":float(reward),
                    f"{condition}_labor":float(labor),
                    f"{condition}_abatement":float(abatement)
                },**metrics[timestep]}


            
            
            obs, _, done, _ = env_object.step(actions)
            if done["__all__"]:
                for state in episode_states:
                    outputs[state][timestep + 1] = env.global_state[state]["value"][
                        timestep + 1
                    ]
                break
    


    current_time = time.strftime("%H:%M:%S", time.localtime())
    file_name = f"fr_{fr_id}_{current_time}.json"

    with open(os.path.join(PUBLIC_REPO_DIR,"scripts","experiments", "fr", file_name), "w") as f:
        json.dump(metrics, f)

    return outputs



def fetch_episode_states_tariff(trainer_obj=None, episode_states=None):
    """
    Helper function to rollout the env and fetch env states for an episode.
    + runs the tariff comparison experiment
    + for a range of tariff rates on a common seed, a single agent is treated
    as a pariah or control, where pariah indicates that all agents tariff at a given level.
    """
    assert trainer_obj is not None
    assert episode_states is not None
    assert isinstance(episode_states, list)
    assert len(episode_states) > 0

    outputs = {}

    metrics = [{} for i in range(trainer_obj.workers.local_worker().env.env.episode_length)]
    #choose one agent to be the freerider
    pariah_id = np.random.randint(0,trainer_obj.workers.local_worker().env.env.num_agents) 
    tariff_rates = [5,6,7,8,9]
    groups = ["pariah", "control"]


    for tariff_rate in tariff_rates:

        for group in groups:
    
            # Fetch the env object from the trainer
            env_object = trainer_obj.workers.local_worker().env
            obs = env_object.reset()

            env = env_object.env

            episode_states = [s for s in episode_states if part_of(s, env.global_state)]

            for state in episode_states:
                assert state in env.global_state, f"{state} is not in global state!"
                # Initialize the episode states
                array_shape = env.global_state[state]["value"].shape
                outputs[state] = np.nan * np.ones(array_shape)

            agent_states = {}
            policy_ids = {}
            policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
            for region_id in range(env.num_regions):
                policy_ids[region_id] = policy_mapping_fn(region_id)
                agent_states[region_id] = trainer_obj.get_policy(
                    policy_ids[region_id]
                ).get_initial_state()

            #action offset index for tariff actions
            tariff_offset = len(env.savings_action_nvec
                + env.mitigation_rate_action_nvec
                + env.export_action_nvec
                + env.import_actions_nvec)
            number_tariff_actions = len(env.tariff_actions_nvec)

            

            for timestep in range(env.episode_length):
                for state in episode_states:
                    outputs[state][timestep] = env.global_state[state]["value"][timestep]

                actions = {}
                # TODO: Consider using the `compute_actions` (instead of `compute_action`)
                # API below for speed-up when there are many agents.
                for region_id in range(env.num_agents):
                    if (
                        len(agent_states[region_id]) == 0
                    ):  # stateless, with a linear model, for example
                        actions[region_id] = trainer_obj.compute_single_action(
                            obs[region_id],
                            agent_states[region_id],
                            policy_id=policy_ids[region_id],
                        )
                    else:  # stateful
                        (
                            actions[region_id],
                            agent_states[region_id],
                            _,
                        ) = trainer_obj.compute_single_action(
                            obs[region_id],
                            agent_states[region_id],
                            policy_id=policy_ids[region_id],
                        )
                    
                    if group == "pariah":
                        #all other regions
                        if region_id != pariah_id:
                            #heavily tariff the pariah
                            actions[region_id][tariff_offset+pariah_id] = tariff_rate

                        
                #get tariffs aginst agents
                average_tariffs = np.mean([actions[region_id][tariff_offset:tariff_offset+number_tariff_actions][pariah_id] for region_id in range(env.num_agents) if region_id !=pariah_id])
                reward = env.get_global_state("reward_all_regions", timestep, pariah_id)
                labor = env.get_global_state("labor_all_regions", timestep, pariah_id)
                metrics.append({
                    f"{group}_tariffs_{tariff_rate}":float(average_tariffs),
                    f"{group}_reward_{tariff_rate}":float(reward),
                    f"{group}_labor_{tariff_rate}":float(labor)
                })

                
                
                obs, _, done, _ = env_object.step(actions)
                if done["__all__"]:
                    for state in episode_states:
                        outputs[state][timestep + 1] = env.global_state[state]["value"][
                            timestep + 1
                        ]
                    break

            outputs = {}


   
    current_time = time.strftime("%H:%M:%S", time.localtime())
    file_name = f"fr_{pariah_id}_{current_time}.json"

    with open(os.path.join(PUBLIC_REPO_DIR,"scripts","experiments", "tariff", file_name), "w") as f:
        json.dump(metrics, f)

    return outputs


def trainer(
    negotiation_on=0,
    num_envs=100,
    train_batch_size=1024,
    num_episodes=30000,
    lr=0.0005,
    model_params_save_freq=5000,
    desired_outputs=desired_outputs,
    num_workers=4,
):
    print("Training with RLlib...")

    # Read the run configurations specific to the environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # -----------------------------------------------------------------------------
    config_path = os.path.join(PUBLIC_REPO_DIR, "scripts", "rice_rllib.yaml")
    if not os.path.exists(config_path):
        raise ValueError(
            "The run configuration is missing. Please make sure the correct path "
            "is specified."
        )

    with open(config_path, "r", encoding="utf8") as fp:
        run_config = yaml.safe_load(fp)
    # replace the default setting
    run_config["env"]["negotiation_on"] = negotiation_on
    run_config["trainer"]["num_envs"] = num_envs
    run_config["trainer"]["train_batch_size"] = train_batch_size
    run_config["trainer"]["num_workers"] = num_workers
    run_config["trainer"]["num_episodes"] = num_episodes
    run_config["policy"]["regions"]["lr"] = lr
    run_config["saving"]["model_params_save_freq"] = model_params_save_freq

    # Create trainer
    # --------------
    trainer, save_dir = create_trainer(run_config)

    # Copy the source files into the results directory
    # ------------------------------------------------
    os.makedirs(save_dir)
    # Copy source files to the saving directory
    for file in ["rice.py", "rice_helpers.py"]:
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, file),
            os.path.join(save_dir, file),
        )
    for file in ["rice_rllib.yaml"]:
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, "scripts", file),
            os.path.join(save_dir, file),
        )

    # Add an identifier file
    with open(os.path.join(save_dir, ".rllib"), "x", encoding="utf-8") as fp:
        pass
    fp.close()

    # Perform training
    # ----------------
    trainer_config = run_config["trainer"]
    # num_episodes = trainer_config["num_episodes"]
    # train_batch_size = trainer_config["train_batch_size"]
    # Fetch the env object from the trainer
    env_obj = trainer.workers.local_worker().env.env
    episode_length = env_obj.episode_length
    num_iters = (num_episodes * episode_length) // train_batch_size

    for iteration in range(num_iters):
        print(f"********** Iter : {iteration + 1:5d} / {num_iters:5d} **********")
        result = trainer.train()
        total_timesteps = result.get("timesteps_total")
        if (
            iteration % run_config["saving"]["model_params_save_freq"] == 0
            or iteration == num_iters - 1
        ):
            save_model_checkpoint(trainer, save_dir, total_timesteps)
            logging.info(result)
        print(f"""episode_reward_mean: {result.get('episode_reward_mean')}""")

    # Create a (zipped) submission file
    # ---------------------------------
    subprocess.call(
        [
            "python",
            os.path.join(PUBLIC_REPO_DIR, "scripts", "create_submission_zip.py"),
            "--results_dir",
            save_dir,
        ]
    )

    # Close Ray gracefully after completion
    outputs_ts = fetch_episode_states(trainer, desired_outputs)
    ray.shutdown()
    return trainer, outputs_ts


if __name__ == "__main__":
    print("Training with RLlib...")

    # Read the run configurations specific to the environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # -----------------------------------------------------------------------------
    config_path = os.path.join(PUBLIC_REPO_DIR, "scripts", "rice_rllib.yaml")
    if not os.path.exists(config_path):
        raise ValueError(
            "The run configuration is missing. Please make sure the correct path "
            "is specified."
        )

    with open(config_path, "r", encoding="utf8") as fp:
        run_config = yaml.safe_load(fp)
    # Create trainer
    # --------------
    trainer, save_dir = create_trainer(run_config)

    # Copy the source files into the results directory
    # ------------------------------------------------
    os.makedirs(save_dir)
    # Copy source files to the saving directory
    negotiator_file_location=run_config["env"]["negotiator_class_config"]["file_name"]
    for file in ["rice.py",
                 "rice_helpers.py",
                f"{negotiator_file_location}.py"]:
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, file),
            os.path.join(save_dir, file),
        )
    for file in ["rice_rllib.yaml"]:
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, "scripts", file),
            os.path.join(save_dir, file),
        )

    # Add an identifier file
    with open(os.path.join(save_dir, ".rllib"), "x", encoding="utf-8") as fp:
        pass
    fp.close()

    # Perform training
    # ----------------
    trainer_config = run_config["trainer"]
    num_episodes = trainer_config["num_episodes"]
    train_batch_size = trainer_config["train_batch_size"]
    # Fetch the env object from the trainer
    env_obj = trainer.workers.local_worker().env.env
    episode_length = env_obj.episode_length
    num_iters = (num_episodes * episode_length) // train_batch_size

    for iteration in range(num_iters):
        print(f"********** Iter : {iteration + 1:5d} / {num_iters:5d} **********")
        result = trainer.train()
        total_timesteps = result.get("timesteps_total")
        if (
            iteration % run_config["saving"]["model_params_save_freq"] == 0
            or iteration == num_iters - 1
        ):
            save_model_checkpoint(trainer, save_dir, total_timesteps)
            logging.info(result)
        print(f"""episode_reward_mean: {result.get('episode_reward_mean')}""")

    # Create a (zipped) submission file
    # ---------------------------------
    subprocess.call(
        [
            "python",
            os.path.join(PUBLIC_REPO_DIR, "scripts", "create_submission_zip.py"),
            "--results_dir",
            save_dir,
        ]
    )

    # Close Ray gracefully after completion
    ray.shutdown()
