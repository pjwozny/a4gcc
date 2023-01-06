# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Script to create the zipped submission file from the results directory
"""
import shutil
from pathlib import Path

from scripts.evaluate_submission import get_results_dir, validate_dir


def prepare_submission(results_dir: Path):
    """
    # Validate all the submission files and compress into a .zip.
    Note: This method is also invoked in the trainer script itself!
    So if you ran the training script, you may not need to re-run this.
    Args results_dir: the directory where all the training files were saved.
    """
    assert isinstance(results_dir, Path)

    # Validate the results directory
    validate_dir(results_dir)

    # Make a temporary copy of the results directory for zipping
    results_dir_copy = results_dir.parent / "tmp_copy"
    shutil.copytree(results_dir, results_dir_copy)

    # Remove all the checkpoint state files from the tmp directory except for the last one
    policy_models = list(results_dir_copy.glob("*.state_dict"))
    policy_models = sorted(policy_models, key=lambda x: x.stat().st_mtime)
    _ = [policy_model.unlink() for policy_model in policy_models[:-1]]

    # Create the submission file and delete the temporary copy
    submission_file = Path("submissions") / results_dir.name
    shutil.make_archive(submission_file, "zip", results_dir_copy)
    print("NOTE: The submission file is created at:", submission_file.with_suffix(".zip"))
    shutil.rmtree(results_dir_copy)


if __name__ == "__main__":
    prepare_submission(results_dir=get_results_dir()[0])
