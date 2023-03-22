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
import shutil
import sys
import time
from pathlib import Path

import ray
import torch
import wandb
import yaml
from ray.rllib.algorithms.a2c import A2C
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog

from scripts.fixed_paths import PUBLIC_REPO_DIR

sys.path.append(PUBLIC_REPO_DIR)

from scripts.environment_wrapper import EnvWrapper
from scripts.torch_models import TorchLinear
from scripts.evaluate_submission import perform_evaluation
from negotiator import TorchLinearMasking

ModelCatalog.register_custom_model("torch_linear", TorchLinear)
ModelCatalog.register_custom_model("torch_linear_masking", TorchLinearMasking)


def train(run_config, save_dir):
    # Copy relevant source files
    # ------------------------------------------------
    copy_files = [
        "rice.py",
        "negotiator.py",
        "rice_helpers.py",
        "scripts/rice_rllib.yaml",
    ]
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
    model_save_freq = run_config["saving"]["model_params_save_freq"]
    next_model_checkpoint = model_save_freq

    result = {"episodes_total": 0}
    while result["episodes_total"] < num_episodes:
        print(
            f"********** Episodes : {result['episodes_total']:5d} / {num_episodes:5d} **********"
        )
        result = trainer.train()
        print(f"episode_reward_mean: {result.get('episode_reward_mean')}")

        if run_config["logging"]["enabled"]:
            wandb.log(
                {
                    "episode_reward_min": result["episode_reward_min"],
                    "episode_reward_mean": result["episode_reward_mean"],
                    "episode_reward_max": result["episode_reward_max"],
                },
                step=result["episodes_total"],
            )
            wandb.log(
                result["info"]["learner"]["regions"]["learner_stats"],
                step=result["episodes_total"],
            )

        if result["episodes_total"] >= next_model_checkpoint:
            next_model_checkpoint += model_save_freq
            total_episodes = result["episodes_total"]
            save_model_checkpoint(trainer, save_dir, total_episodes)
            logging.info(result)

    total_episodes = result["episodes_total"]
    save_model_checkpoint(trainer, save_dir, total_episodes)
    logging.info(result)

    if run_config["logging"]["enabled"]:
        wandb.finish()

    # Create a (zipped) submission file
    # ---------------------------------
    # prepare_submission(save_dir)

    # Close Ray gracefully after completion
    ray.shutdown()


def get_rllib_config(run_config, env_class, seed=None):
    """
    Reference: https://docs.ray.io/en/latest/rllib-training.html
    """

    env_config = run_config["env"]
    assert isinstance(env_config, dict)
    env_object = env_class(env_config=env_config)
    episode_length = env_object.env.episode_length

    # Define all the policies here
    policy_config = run_config["policy"]["regions"]

    policy_config["entropy_coeff_schedule"] = [[episode * episode_length, coeff] for episode, coeff in policy_config["entropy_coeff_schedule"]]

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
    def policy_mapping_fn(agent_id):
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
        "batch_mode": train_config["batch_mode"],
        "rollout_fragment_length": episode_length,
        "env_config": run_config["env"],
        "framework": train_config["framework"],
        "multiagent": multiagent_config,
        "num_workers": train_config["num_workers"],
        "num_gpus": train_config["num_gpus"],
        "num_envs_per_worker": train_config["num_envs_per_worker"],
        "num_cpus_per_worker": train_config["num_cpus_per_worker"],
        "train_batch_size": train_config["train_batch_size_in_episodes"]
        * episode_length,
        "ignore_worker_failures": train_config["ignore_worker_failures"],
        "recreate_failed_workers": train_config["recreate_failed_workers"],
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


def create_trainer(run_config, source_dir=None, seed=None) -> Algorithm:
    """
    Create the RLlib trainer.
    """
    # Create the A2C trainer.
    if source_dir:
        run_config["env"]["source_dir"] = source_dir
    trainer_config = get_rllib_config(
        run_config=run_config, env_class=EnvWrapper, seed=seed
    )
    rllib_trainer = A2C(
        env=EnvWrapper,
        config=trainer_config,
    )
    return rllib_trainer


if __name__ == "__main__":
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

    # Create the save directory
    save_dir = Path("experiments") / time.strftime("%Y-%m-%d_%H%M%S")
    save_dir.mkdir(parents=True)

    # Initialize wandb
    if run_config["logging"]["enabled"]:
        wandb_config = run_config["logging"]["wandb_config"]
        wandb.login(key=wandb_config["login"])
        wandb.init(
            project=wandb_config["project"],
            name=f'{wandb_config["run"]}_train',
            entity=wandb_config["entity"],
        )

    # Write timestamp of start training
    with open(save_dir / "timestamp.txt", "a") as f:
        f.write(f"START:\t{time.strftime('%Y-%m-%d_%H:%M:%S')}\n")
    train(run_config, save_dir)
    # Write timestamp of finished training
    with open(save_dir / "timestamp.txt", "a") as f:
        f.write(f"STOP:\t{time.strftime('%Y-%m-%d_%H:%M:%S')}\n")

    framework_used, succeeded, metrics, comments = perform_evaluation(save_dir)
    print(f"Framework used: {framework_used}")
    print(f"Succeeded: {succeeded}")
    print(f"Metrics: {metrics}")
    print(f"Comments: {comments}")
