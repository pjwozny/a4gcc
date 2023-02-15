# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Evaluation script for the rice environment
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict

import numpy as np
import yaml

from pathlib import Path

from train_with_rllib import (
            create_trainer,
            fetch_episode_states_freerider,
            fetch_episode_states_tariff,
            fetch_episode_states,
            load_model_checkpoints,
        )

EXP = {"fr":fetch_episode_states_freerider,
        "tariff":fetch_episode_states_tariff,
        "none":fetch_episode_states}

_path = Path(os.path.abspath(__file__))

from fixed_paths import PUBLIC_REPO_DIR
sys.path.append(os.path.join(PUBLIC_REPO_DIR, "scripts"))
print("Using PUBLIC_REPO_DIR = {}".format(PUBLIC_REPO_DIR))

_PRIVATE_REPO_DIR = os.path.join(_path.parent.parent.parent.absolute(), "private-repo-clone")
sys.path.append(os.path.join(_PRIVATE_REPO_DIR, "backend"))
print("Using _PRIVATE_REPO_DIR = {}".format(_PRIVATE_REPO_DIR))


# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)

_SEED = np.random.randint(0,1000) #1234567890  # seed used for evaluation

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


def get_results_dir():
    """
    Obtain the 'results' directory from the system arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        default=".",
        help="the directory where all the submission files are saved. Can also be "
        "the zipped file containing all the submission files.",
    )

    parser.add_argument(
        "--exp",
        "-e",
        type=str,
        default="none",
        help="experiment_type",
    )

    args = parser.parse_args()

    try:
        results_dir = args.results_dir
        experiment = args.exp

        # Also handle a zipped file
        if results_dir.endswith(".zip"):
            unzipped_results_dir = os.path.join("/tmp", str(time.time()))
            shutil.unpack_archive(results_dir, unzipped_results_dir)
            results_dir = unzipped_results_dir
        return results_dir, parser, experiment
    except Exception as err:
        raise ValueError("Cannot obtain the results directory") from err

def perform_evaluation(
    results_directory=None,
    eval_seed=None,
    experiment = "none"
):
    """
    Create the trainer and compute metrics.
    """
    assert results_directory is not None

    framework = 'rllib'
    config_file = os.path.join(results_directory, f"rice_{framework}.yaml")
    with open(config_file, "r", encoding="utf-8") as file_ptr:
                        run_config = yaml.safe_load(file_ptr)

    trainer, _ = create_trainer(
            run_config, source_dir=results_directory, seed=eval_seed
        )
    load_model_checkpoints(trainer, results_directory)
    desired_outputs = list(_METRICS_TO_LABEL_DICT.keys())

    #run experiment
    EXP[experiment](trainer, desired_outputs)
    


if __name__ == "__main__":
    logging.info("This script performs evaluation of your code.")
    results_dir, _, experiment = get_results_dir()

    perform_evaluation(
        results_dir, eval_seed=_SEED, experiment = experiment
    )