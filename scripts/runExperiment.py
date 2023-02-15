import os
from fixed_paths import PUBLIC_REPO_DIR
import subprocess
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--exp",
    "-e",
    type=str,
    default="none",
    help="experiment_type",
)

parser.add_argument(
    "--iterations",
    "-i",
    type=int,
    default="none",
    help="experiment_type",
)

args = parser.parse_args()
exp = args.exp
iterations = args.iterations

EXP = {"fr":"c393c2556b7c8df664f27286a3c63901.zip", #BasicClubDiscreteDefect FreeRider 
        "tariff":"d630fd2ed25f1a493eca3f5e17609aaf.zip", #NoNegotiator Tariff 
        "none":"d630fd2ed25f1a493eca3f5e17609aaf.zip"} #NoNegotiator Debugging

if __name__ == "__main__":

    for i in tqdm(range(iterations)):
        subprocess.call(
        [
            "python",
            os.path.join(PUBLIC_REPO_DIR, "scripts", "conductExperiment.py"),
            "-r",
            f"Submissions/{EXP[exp]}",
            "-e",
            exp
        ]
        )