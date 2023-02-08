import os
from zipfile import ZipFile
import yaml
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

#30k bilateralnegotiator
# 1670227114.zip

for file in os.listdir(os.path.join(__location__,"Submissions")):
    if ".zip" in file:
        z = ZipFile(os.path.join(__location__,"Submissions", file))
        if "rice_rllib.yaml" in z.namelist():
            yml = z.extract("rice_rllib.yaml")
            with open(yml, "r") as f:
                config = yaml.safe_load(f.read())
                if config["trainer"]["num_episodes"] > 500:

                    print(file)
                

