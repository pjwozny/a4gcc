# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Training script for the rice environment using RLlib
https://docs.ray.io/en/latest/rllib-training.html
"""

import logging
import os
import sys
import shutil
import time
from pathlib import Path

import ray
import torch
import yaml
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.models import ModelCatalog

from scripts.fixed_paths import PUBLIC_REPO_DIR
sys.path.append(PUBLIC_REPO_DIR)

from scripts.create_submission_zip import prepare_submission
from scripts.environment_wrapper import EnvWrapper
from scripts.torch_models import TorchLinear

ModelCatalog.register_custom_model("torch_linear", TorchLinear)



def train():
    # Read the run configurations specific to the environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # ------------------------------------------------
    config_path = Path() / "scripts" / "rice_rllib.yaml"
    if not config_path.exists():
        raise ValueError(
            "The run configuration is missing. Please make sure the correct path is specified."
        )

    with open(config_path, "r", encoding="utf8") as fp:
        run_config = yaml.safe_load(fp)

    # Create an experiment directory and copy relevant source files
    # ------------------------------------------------
    save_dir = Path("experiments") / time.strftime("%Y-%m-%d_%H%M%S")
    save_dir.mkdir(parents=True)
    copy_files = ["rice.py", "negotiator.py", "rice_helpers.py", "scripts/rice_rllib.yaml"]
    for file in copy_files:
        shutil.copy(file, save_dir)
    # shutil.copytree("scripts_submit", save_dir)
    # Add an identifier file
    (save_dir / ".rllib").touch()

    # Create trainer
    # ------------------------------------------------
    ray.init(ignore_reinit_error=True)
    trainer = create_trainer(run_config)

    # Perform training
    # ------------------------------------------------
    num_episodes = run_config["trainer"]["num_episodes"]
    train_batch_size = run_config["trainer"]["train_batch_size"]
    model_save_freq = run_config["saving"]["model_params_save_freq"]
    # Fetch the env object from the trainer
    env_obj = trainer.workers.local_worker().env.env
    episode_length = env_obj.episode_length
    num_iters = (num_episodes * episode_length) // train_batch_size

    for iteration in range(num_iters):
        print(f"********** Iter : {iteration + 1:5d} / {num_iters:5d} **********")
        result = trainer.train()
        print(f"episode_reward_mean: {result.get('episode_reward_mean')}")

        if iteration % model_save_freq == 0 or iteration + 1 == num_iters:
            total_timesteps = result.get("timesteps_total")
            save_model_checkpoint(trainer, save_dir, total_timesteps)
            logging.info(result)

    # Create a (zipped) submission file
    # ---------------------------------
    prepare_submission(save_dir)

    # Close Ray gracefully after completion
    ray.shutdown()


def get_rllib_config(run_config=None, env_class=None, seed=None):
    """
    Reference: https://docs.ray.io/en/latest/rllib-training.html
    """

    assert run_config is not None
    assert env_class is not None

    env_config = run_config["env"]
    assert isinstance(env_config, dict)
    env_object = env_class(env_config=env_config)

    # Define all the policies here
    policy_config = run_config["policy"]["regions"]

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

    train_config = run_config["trainer"]
    rllib_config = {
        # Arguments dict passed to the env creator as an EnvContext object (which
        # is a dict plus the properties: num_workers, worker_index, vector_index,
        # and remote).
        "env_config": run_config["env"],
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
        logging.info(
            "Saving the model checkpoints for policy %s to %s.", policy, filepath
        )
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
    files = [f for f in os.listdir(save_directory) if f.endswith("state_dict")]

    assert len(files) == len(trainer_obj.config["multiagent"]["policies"])

    model_params = trainer_obj.get_weights()
    for policy in model_params:
        policy_models = [
            os.path.join(save_directory, file) for file in files if policy in file
        ]
        # If there are multiple files, then use the ckpt_idx to specify the checkpoint
        assert ckpt_idx < len(policy_models)
        sorted_policy_models = sorted(policy_models, key=os.path.getmtime)
        policy_model_file = sorted_policy_models[ckpt_idx]
        model_params[policy] = torch.load(policy_model_file)
        logging.info("Loaded model checkpoints %s.", policy_model_file)

    trainer_obj.set_weights(model_params)


def create_trainer(run_config, seed=None):
    """
    Create the RLlib trainer.
    """
    # Create the A2C trainer.
    # run_config["env"]["source_dir"] = source_dir
    trainer_config = get_rllib_config(
        run_config=run_config, env_class=EnvWrapper, seed=seed
    )
    rllib_trainer = A2CTrainer(
        env=EnvWrapper,
        config=trainer_config,
    )
    return rllib_trainer


if __name__ == "__main__":
    train()
