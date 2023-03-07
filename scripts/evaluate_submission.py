# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Evaluation script for the rice environment
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict

import numpy as np
import wandb
import yaml
import pickle as pkl
from pathlib import Path
from visualizeOutputs import construct_stacked_bar_chart
from run_unittests import fetch_base_env, run_unittests
from create_submission_zip import prepare_submission, validate_dir

_path = Path(os.path.abspath(__file__))

from fixed_paths import PUBLIC_REPO_DIR
from gym.spaces import MultiDiscrete
sys.path.append(os.path.join(PUBLIC_REPO_DIR, "scripts"))
print("Using PUBLIC_REPO_DIR = {}".format(PUBLIC_REPO_DIR))

_PRIVATE_REPO_DIR = os.path.join(_path.parent.parent.parent.absolute(), "private-repo-clone")
sys.path.append(os.path.join(_PRIVATE_REPO_DIR, "backend"))
print("Using _PRIVATE_REPO_DIR = {}".format(_PRIVATE_REPO_DIR))

import pickle as pkl
import numpy as np

# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)

_SEED = 1234567890  # seed used for evaluation

_INDEXES_FILENAME = "climate_economic_min_max_indices.txt"

_METRICS_TO_LABEL_DICT = OrderedDict()
# Read the dict values below as
# (label, decimal points used to round off value: 0 becomes an integer)
_METRICS_TO_LABEL_DICT["global_temperature"] = ("Temperature Rise", 2)
_METRICS_TO_LABEL_DICT["global_carbon_mass"] = ("Carbon Mass", 2)
_METRICS_TO_LABEL_DICT["capital_all_regions"] = ("Capital", 2)
_METRICS_TO_LABEL_DICT["labor_all_regions"] = ("Labor", 2)
_METRICS_TO_LABEL_DICT["production_factor_all_regions"] = ("Production Factor", 2)
_METRICS_TO_LABEL_DICT["production_all_regions"] = ("Production", 2)
_METRICS_TO_LABEL_DICT["intensity_all_regions"] = ("Intensity", 2)
# _METRICS_TO_LABEL_DICT["global_exegenous_emissions"] = ("Exogenous Emissions", 2)
_METRICS_TO_LABEL_DICT["global_land_emissions"] = ("Land Emissions", 2)
# _METRICS_TO_LABEL_DICT["capital_deprication_all_regions"] = ("Capital Deprication", 2)
_METRICS_TO_LABEL_DICT["savings_all_regions"] = ("Savings", 2)
_METRICS_TO_LABEL_DICT["mitigation_rate_all_regions"] = ("Mitigation Rate", 0)
_METRICS_TO_LABEL_DICT["max_export_limit_all_regions"] = ("Max Export Limit", 2)
_METRICS_TO_LABEL_DICT["mitigation_cost_all_regions"] = ("Mitigation Cost", 2)
_METRICS_TO_LABEL_DICT["damages_all_regions"] = ("Damages", 2)
_METRICS_TO_LABEL_DICT["abatement_cost_all_regions"] = ("Abatement Cost", 2)
_METRICS_TO_LABEL_DICT["utility_all_regions"] = ("Utility", 2)
_METRICS_TO_LABEL_DICT["social_welfare_all_regions"] = ("Social Welfare", 2)
_METRICS_TO_LABEL_DICT["reward_all_regions"] = ("Reward", 2)
_METRICS_TO_LABEL_DICT["consumption_all_regions"] = ("Consumption", 2)
_METRICS_TO_LABEL_DICT["current_balance_all_regions"] = ("Current Balance", 2)
_METRICS_TO_LABEL_DICT["gross_output_all_regions"] = ("Gross Output", 2)
_METRICS_TO_LABEL_DICT["investment_all_regions"] = ("Investment", 2)
_METRICS_TO_LABEL_DICT["production_all_regions"] = ("Production", 2)
_METRICS_TO_LABEL_DICT["minimum_mitigation_rate_all_regions"] = ("Minimum Mitigation Rate", 0)
# _METRICS_TO_LABEL_DICT["
# _METRICS_TO_LABEL_DICT["

# _METRICS_TO_LABEL_DICT["gross_output_all_regions"] = ("Gross Output", 0)
# _METRICS_TO_LABEL_DICT["investment_all_regions"] = ("Investment", 0)
# _METRICS_TO_LABEL_DICT["abatement_cost_all_regions"] = ("Abatement Cost", 2)
# _METRICS_TO_LABEL_DICT["mitigation_rate_all_regions"] = ("Mitigation Rate", 0)

# _METRICS_TO_LABEL_DICT["consumption_all_regions"] = ("Consumption", 0)
# # _METRICS_TO_LABEL_DICT["current_balance_all_regions"]
# # _METRICS_TO_LABEL_DICT["current_balance_all_regions"]
# # _METRICS_TO_LABEL_DICT["current_balance_all_regions"]
# # _METRICS_TO_LABEL_DICT["current_balance_all_regions"]
# # _METRICS_TO_LABEL_DICT["current_balance_all_regions"]

# _METRICS_TO_LABEL_DICT["reward_all_regions"] = ("Episode Reward", 2)

def get_imports(framework, experiment_id):
    """
    Fetch relevant imports.
    """
    assert framework is not None
    if framework == "rllib":
        from train_with_rllib import (
            create_trainer,
            fetch_episode_states,
            fetch_episode_states_freerider,
            fetch_episode_states_tariff,
            load_model_checkpoints,
        )
    elif framework == "warpdrive":
        from train_with_warp_drive import (
            create_trainer,
            fetch_episode_states,
            load_model_checkpoints,
        )
    else:
        raise ValueError(f"Unknown framework {framework}!")

    episode_fetchers = {
        "fr":fetch_episode_states_freerider,
        "tariff":fetch_episode_states_tariff,
        "none":fetch_episode_states
    }

    fetch_episode_states = episode_fetchers[experiment_id]

    return create_trainer, load_model_checkpoints, fetch_episode_states


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        help="the directory where all the submission files are saved. Can also be "
        "the zipped file containing all the submission files.",
        required=True,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="which experiment to run",
        default="none",
        required=False,
    )

    args = parser.parse_args()
    results_dir = args.results_dir
    experiment_id = args.experiment

    return results_dir, experiment_id, parser


def compute_metrics(fetch_episode_states, trainer, framework, submission_file, log_config=None, num_episodes=1, include_c_e_idx=True):
    """
    Generate episode rollouts and compute metrics.
    """
    assert trainer is not None
    available_frameworks = ["rllib", "warpdrive"]
    assert (
        framework in available_frameworks
    ), f"Invalid framework {framework}, should be in f{available_frameworks}."

    if log_config and log_config["enabled"]:
        wandb_config = log_config["wandb_config"]
        wandb.login(key=wandb_config["login"])
        wandb.init(project=wandb_config["project"],
            name=f'{wandb_config["run"]}_eval',
            entity=wandb_config["entity"])

    # Fetch all the desired outputs to compute various metrics.
    desired_outputs = list(_METRICS_TO_LABEL_DICT.keys())

    episode_states = {}
    eval_metrics = {}
    try:
        for episode_id in range(num_episodes):
            if fetch_episode_states is not None:
                episode_states[episode_id] = fetch_episode_states(
                    trainer, desired_outputs
                )
            else:
                episode_states[episode_id] = trainer.fetch_episode_global_states(
                    desired_outputs
                )
        
        with open("episode_states.pkl", "wb") as f:
            pkl.dump(episode_states[0], f, protocol=pkl.HIGHEST_PROTOCOL)

        if log_config and log_config["enabled"]:
        
            #log mitigation rate counts of each country over time
            wandb.log({"mitigation_rate Counts Across Time":construct_stacked_bar_chart(episode_states[0],
                                        field="mitigation_rate_all_regions")})

            #log mitigation rate counts of each country over time
            wandb.log({"minimum_mitigation_rate Counts Across Time":construct_stacked_bar_chart(episode_states[0],
                                        field="minimum_mitigation_rate_all_regions")})

        for feature in desired_outputs:
            feature_values = [None for _ in range(num_episodes)]

            if feature == "global_temperature":
                # Get the temp rise for upper strata
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = (
                        episode_states[episode_id][feature][-1, 0]
                        - episode_states[episode_id][feature][0, 0]
                    )

            elif feature == "global_carbon_mass":
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = episode_states[episode_id][feature][
                        -1, 0
                    ]

            else:
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = np.sum(
                        episode_states[episode_id][feature]
                    )

            # Compute mean feature value across episodes
            mean_feature_value = np.mean(feature_values)

            # Formatting the values
            metrics_to_label_dict = _METRICS_TO_LABEL_DICT[feature]

            eval_metrics[metrics_to_label_dict[0]] = perform_format(
                mean_feature_value, metrics_to_label_dict[1]
            )


            if log_config and log_config["enabled"]:
                #TODO: fix dirty method to remove negotiation steps from results
                interval = (len(episode_states[episode_id][feature]) - 1) // 20
                ys = episode_states[episode_id][feature][0::interval].T


                xs = list(range(len(ys[0])))
                wandb.log({feature : wandb.plot.line_series(
                       xs=xs,
                       ys=ys.tolist(),
                       keys=[f"region_{x}" for x in range(len(ys))],
                       title=feature,
                       xname="Steps")})

                if feature.endswith("_all_regions"):
                    title = f"mean_{feature.rsplit('_', 2)[0]}_over_regions"
                    ys_mean = np.mean(ys, axis=0)
                    data = [[x, y] for (x, y) in zip(xs, ys_mean.tolist())]
                    table = wandb.Table(data=data, columns = ["Steps", "y"])
                    wandb.log({title : wandb.plot.line(table, "Steps", "y",
                            title=title)})
        if include_c_e_idx:
            if not os.path.exists(_INDEXES_FILENAME):
                # Write min, max climate and economic index values to a file
                # for use during evaluation.
                indices_dict = generate_min_max_climate_economic_indices()
                # Write indices to a file
                with open(_INDEXES_FILENAME, "w", encoding="utf-8") as file_ptr:
                    file_ptr.write(json.dumps(indices_dict))
            with open(_INDEXES_FILENAME, "r", encoding="utf-8") as file_ptr:
                index_dict = json.load(file_ptr)
            eval_metrics["climate_index"] = np.round(
                (eval_metrics["Temperature Rise"] - index_dict["min_ci"])
                / (index_dict["max_ci"] - index_dict["min_ci"]),
                2,
            )
            eval_metrics["economic_index"] = np.round(
                (eval_metrics["Gross Output"] - index_dict["min_ei"])
                / (index_dict["max_ei"] - index_dict["min_ei"]),
                2,
            )
        success = True
        comment = "Successful submission"
    except Exception as err:
        logging.error(err)
        success = False
        comment = "Could not obtain an episode rollout!"
        eval_metrics = {}

    if log_config and log_config["enabled"]:
        # attach submission file as artifact (needs to be named after the nego class)
        artifact = wandb.Artifact("submission", type="model")
        artifact.add_file(submission_file)
        wandb.log_artifact(artifact)
        wandb.finish()

    return success, comment, eval_metrics


def val_metrics(trainer, logged_ts, framework, num_episodes=1, include_c_e_idx=True):
    """
    Generate episode rollouts and compute metrics.
    """
    assert trainer is not None
    available_frameworks = ["rllib", "warpdrive"]
    assert (
        framework in available_frameworks
    ), f"Invalid framework {framework}, should be in f{available_frameworks}."

    # Fetch all the desired outputs to compute various metrics.
    desired_outputs = list(_METRICS_TO_LABEL_DICT.keys())
    episode_states = {}
    eval_metrics = {}
    try:
        for episode_id in range(num_episodes):
            episode_states[episode_id] = logged_ts
            
        for feature in desired_outputs:
            feature_values = [None for _ in range(num_episodes)]

            if feature == "global_temperature":
                # Get the temp rise for upper strata
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = (
                        episode_states[episode_id][feature][-1, 0]
                        - episode_states[episode_id][feature][0, 0]
                    )

            elif feature == "global_carbon_mass":
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = episode_states[episode_id][feature][
                        -1, 0
                    ]

            else:
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = np.sum(
                        episode_states[episode_id][feature]
                    )

            # Compute mean feature value across episodes
            mean_feature_value = np.mean(feature_values)

            # Formatting the values
            metrics_to_label_dict = _METRICS_TO_LABEL_DICT[feature]

            eval_metrics[metrics_to_label_dict[0]] = perform_format(
                mean_feature_value, metrics_to_label_dict[1]
            )
        if include_c_e_idx:
            if not os.path.exists(_INDEXES_FILENAME):
                # Write min, max climate and economic index values to a file
                # for use during evaluation.
                indices_dict = generate_min_max_climate_economic_indices()
                # Write indices to a file
                with open(_INDEXES_FILENAME, "w", encoding="utf-8") as file_ptr:
                    file_ptr.write(json.dumps(indices_dict))
            with open(_INDEXES_FILENAME, "r", encoding="utf-8") as file_ptr:
                index_dict = json.load(file_ptr)
            eval_metrics["climate_index"] = np.round(
                (eval_metrics["Temperature Rise"] - index_dict["min_ci"])
                / (index_dict["max_ci"] - index_dict["min_ci"]),
                2,
            )
            eval_metrics["economic_index"] = np.round(
                (eval_metrics["Gross Output"] - index_dict["min_ei"])
                / (index_dict["max_ei"] - index_dict["min_ei"]),
                2,
            )
        success = True
        comment = "Successful submission"
    except Exception as err:
        logging.error(err)
        success = False
        comment = "Could not obtain an episode rollout!"
        eval_metrics = {}

    return success, comment, eval_metrics


def perform_format(val, num_decimal_places):
    """
    Format value to the number of desired decimal points.
    """
    if np.isnan(val):
        return val
    assert num_decimal_places >= 0
    rounded_val = np.round(val, num_decimal_places)
    if num_decimal_places == 0:
        return int(rounded_val)
    return rounded_val


def perform_evaluation(
    results_directory,
    experiment_id,
    num_episodes=1,
    eval_seed=None,
):
    """
    Create the trainer and compute metrics.
    """
    assert results_directory is not None
    eval_metrics = {}
    assert num_episodes > 0

    framework, success, comment = validate_dir(Path(results_directory))
    submission_file = prepare_submission(Path(results_directory))

    if success:
        logging.info("Running unit tests...")

        try:
            run_unittests(results_directory)
            logging.info("DONE")

            if success:
                (
                    create_trainer,
                    load_model_checkpoints,
                    fetch_episode_states,
                ) = get_imports(framework, experiment_id)

                logging.info("Performing eval...")

                # Load a run configuration
                config_file = os.path.join(results_directory, f"rice_{framework}.yaml")

                if not os.path.exists(config_file):
                    success = False
                    comment = (
                        f"The run configuration is missing in {results_directory}."
                    )

                else:
                    with open(config_file, "r", encoding="utf-8") as file_ptr:
                        run_config = yaml.safe_load(file_ptr)

                    #TODO: create better method of setting workers for eval
                    run_config["trainer"]["num_envs_per_worker"] = 0
                    run_config["trainer"]["num_workers"] = 0 # 0 workers -> head process does the rollout
                    run_config["trainer"]["num_gpus"] = 0

                    # Create trainer object
                    try:
                        trainer, _ = create_trainer(
                            run_config, source_dir=results_directory, seed=eval_seed
                        )

                        # Load model checkpoints
                        try:
                            load_model_checkpoints(trainer, results_directory)

                            # Compute metrics
                            try:
                                success, comment, eval_metrics = compute_metrics(
                                    fetch_episode_states,
                                    trainer,
                                    framework,
                                    submission_file,
                                    run_config["logging"],
                                    num_episodes=num_episodes,
                                )

                                if framework == "warpdrive":
                                    trainer.graceful_close()
                                logging.info("DONE!")

                            except Exception as err:
                                logging.error(err)
                                success = False
                                comment = "Count not fetch episode and compute metrics."

                        except Exception as err:
                            logging.error(err)
                            success = False
                            comment = "Could not load model checkpoints."

                    except Exception as err:
                        logging.error(err)
                        success = False
                        comment = (
                            "Could not create trainer with the run_config provided."
                        )
        except subprocess.CalledProcessError as err:
            logging.error(err)
            success = False
            comment = "Unit tests were not successful."

    return framework, success, eval_metrics, comment


def get_temp_rise_and_gross_output(env, actions):
    env.reset()
    for _ in range(env.episode_length):
        env.step(actions)
    temperature_array = env.global_state["global_temperature"]["value"]
    temperature_rise = temperature_array[-1, 0] - temperature_array[0, 0]

    total_gross_production = np.sum(
        env.global_state["gross_output_all_regions"]["value"]
    )
    return temperature_rise, total_gross_production


def generate_min_max_climate_economic_indices():
    """
    Generate min and max climate and economic indices for the leaderboard.
    0% savings, 100% mitigation => best climate index, worst economic index
    100% savings, 0% mitigation => worst climate index, best economic index
    """
    env = fetch_base_env()
    assert isinstance(
        env.action_space[0], MultiDiscrete
    ), "Unknown action space for env."
    all_zero_actions = {
        agent_id: np.zeros(
            len(env.action_space[agent_id].nvec),
            dtype=np.int32,
        )
        for agent_id in range(env.num_agents)
    }

    # 0% savings, 100% mitigation
    low_savings_high_mitigation_actions = {}
    savings_action_idx = 0
    mitigation_action_idx = 1
    for agent_id in range(env.num_agents):
        low_savings_high_mitigation_actions[agent_id] = all_zero_actions[
            agent_id
        ].copy()
        low_savings_high_mitigation_actions[agent_id][
            mitigation_action_idx
        ] = env.num_discrete_action_levels
    # Best climate index, worst economic index
    best_ci, worst_ei = get_temp_rise_and_gross_output(
        env, low_savings_high_mitigation_actions
    )

    high_savings_low_mitigation_actions = {}
    for agent_id in range(env.num_agents):
        high_savings_low_mitigation_actions[agent_id] = all_zero_actions[
            agent_id
        ].copy()
        high_savings_low_mitigation_actions[agent_id][
            savings_action_idx
        ] = env.num_discrete_action_levels
    worst_ci, best_ei = get_temp_rise_and_gross_output(
        env, high_savings_low_mitigation_actions
    )

    index_dict = {
        "min_ci": float(worst_ci),
        "max_ci": float(best_ci),
        "min_ei": float(worst_ei),
        "max_ei": float(best_ei),
    }
    return index_dict


if __name__ == "__main__":
    logging.info("This script performs evaluation of your code.")
    results_dir, experiment_id, _ = parse_args()

    try:
        # Also handle a zipped file
        if results_dir.endswith(".zip"):
            unzipped_results_dir = Path(tempfile.gettempdir()) / str(time.time())
            shutil.unpack_archive(results_dir, unzipped_results_dir)
            results_dir = unzipped_results_dir
    except Exception as err:
        raise ValueError("Cannot obtain the results directory") from err

    framework_used, succeeded, metrics, comments = perform_evaluation(
        results_dir, experiment_id, eval_seed=_SEED
    )
    print(f"Framework used: {framework_used}")
    print(f"Succeeded: {succeeded}")
    print(f"Metrics: {metrics}")
    print(f"Comments: {comments}")
