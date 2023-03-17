# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Script to create the zipped submission file from the results directory
"""
import logging
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import yaml

BACKWARDS_COMPAT_CONFIG = """
trainer:
    num_envs: 1 # number of environment replicas
    rollout_fragment_length: 100 # divide episodes into fragments of this many steps each during rollouts.
    train_batch_size: 2000 # total batch size used for training per iteration (across all the environments)
    num_episodes: 100 # number of episodes to run the training for
    framework: torch # framework setting.
    # Note: RLlib supports TF as well, but our end-to-end pipeline is built for Pytorch only.
    # === Hardware Settings ===
    num_workers: 1 # number of rollout worker actors to create for parallel sampling.
    # Note: Setting the num_workers to 0 will force rollouts to be done in the trainer actor.
    num_gpus: 0 # number of GPUs to allocate to the trainer process. This can also be fractional (e.g., 0.3 GPUs).
"""


def prepare_submission(results_dir: Path) -> Path:
    """
    # Validate all the submission files and compress into a .zip.
    Note: This method is also invoked in the trainer script itself!
    So if you ran the training script, you may not need to re-run this.
    Args results_dir: the directory where all the training files were saved.
    """
    assert isinstance(results_dir, Path)

    # Validate the results directory
    _, success, comment = validate_dir(results_dir)
    if not success:
        raise FileNotFoundError(comment)

    # Remove all the checkpoint state files from the tmp directory except for the last one
    policy_models = list(results_dir.glob("*.state_dict"))
    policy_models = sorted(policy_models, key=lambda x: x.stat().st_mtime)

    # assemble list of files to copy
    files_to_copy = list(results_dir.glob("*.py"))
    files_to_copy.extend(list(results_dir.glob(".*")))
    files_to_copy.append(results_dir / "rice_rllib.yaml")
    files_to_copy.append(policy_models[-1])

    # Make a temporary copy of the results directory for zipping
    results_dir_copy = results_dir.parent / "tmp_copy"
    results_dir_copy.mkdir(parents=True)

    for file in files_to_copy:
        shutil.copy(file, results_dir_copy / file.name)

    # Create the submission file and delete the temporary copy
    submission_file = Path("submissions") / results_dir.name
    shutil.make_archive(submission_file, "zip", results_dir_copy)
    print("NOTE: The submission file is created at:\t\t\t", submission_file.with_suffix(".zip"))

    # open rice config yaml file in copied directory
    config_path = results_dir_copy / "rice_rllib.yaml"
    with open(config_path, "r", encoding="utf8") as fp:
        run_config = yaml.safe_load(fp)

    # modify the rice_config yaml to work with the original code
    backwards_config = yaml.safe_load(BACKWARDS_COMPAT_CONFIG)
    run_config["trainer"] = backwards_config["trainer"]
    del run_config["logging"]

    # write rice_config yaml file to tmp directory
    with open(config_path, "w", encoding="utf8") as fp:
        yaml.dump(run_config, fp, default_flow_style=False)

    # Create the backwards compatible submission file and delete the temporary copy
    submission_file_bc = Path("submissions") / "backwards_compatible" / results_dir.name
    shutil.make_archive(submission_file_bc, "zip", results_dir_copy)
    print("NOTE: The backwards compatible submission file is created at:\t", submission_file_bc.with_suffix(".zip"))
    
    # delete temporary directory
    shutil.rmtree(results_dir_copy)

    return submission_file.with_suffix(".zip")


def validate_dir(results_dir: Path):
    """
    Validate that all the required files are present in the 'results' directory.
    """
    assert isinstance(results_dir, Path)
    framework = None
    files = set(os.listdir(results_dir))
    if ".warpdrive" in files:
        framework = "warpdrive"
        # Warpdrive was used for training
        for file in [
            "rice.py",
            "rice_helpers.py",
            "rice_cuda.py",
            "rice_step.cu",
            "rice_warpdrive.yaml",
        ]:
            if file not in files:
                success = False
                logging.error(
                    "%s is not present in the results directory: %s!", file, results_dir
                )
                comment = f"{file} is not present in the results directory!"
                break
            success = True
            comment = "Valid submission"
    elif ".rllib" in files:
        framework = "rllib"
        # RLlib was used for training
        for file in ["rice.py", "rice_helpers.py", "rice_rllib.yaml"]:
            if file not in files:
                success = False
                logging.error(
                    "%s is not present in the results directory: %s!", file, results_dir
                )
                comment = f"{file} is not present in the results directory!"
                break
            success = True
            comment = "Valid submission"
    else:
        success = False
        logging.error(
            "Missing identifier file! Either the .rllib or the .warpdrive "
            "file must be present in the results directory: %s",
            results_dir,
        )
        comment = "Missing identifier file!"

    return framework, success, comment

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        help="the directory where all the submission files are saved. Can also be "
        "the zipped file containing all the submission files.",
        required=True,
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    prepare_submission(results_dir)
