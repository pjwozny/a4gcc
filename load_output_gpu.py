import pickle as pkl
import pandas as pd
import os
import json
import wandb
import numpy as np
from ast import literal_eval
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


episode_length = 60
num_discrete_action_levels=10
num_regions = 27
wandb.login(key=wandb_config["login"])
with open(os.path.join(__location__,  "output.json"), "r") as f:
    output = json.loads(f.read())
    dict = json.loads(output)
for type in ['nego_on']:
    print()
            
    
    wandb.init(project=wandb_config["project"],
        name=f'{wandb_config["run"]}_{type}',
        entity=wandb_config["entity"])

    country_data =[
        "capital_depreciation_all_regions",
        "savings_all_regions",
        "mitigation_rate_all_regions",
        "max_export_limit_all_regions",
        "mitigation_cost_all_regions",
        "damages_all_regions",
        "abatement_cost_all_regions",
        "utility_all_regions",
        "social_welfare_all_regions",
        "reward_all_regions",
        ]

    global_features = [
        "global_temperature",
        "global_carbon_mass",
        "global_exogenous_emissions",
        "global_land_emissions",
    ]

    for key in global_features:
        current_data = []
        time_steps_for_plotting = []
        multi_line = False
        for step in range(1,episode_length-1):

            step_data = dict[type][key][step]
            
            if len(dict[type][key][step]) == 0:
                wandb.log({key:step_data[0]})
            else:
                multi_line = True
                current_data.append(step_data)
                time_steps_for_plotting.append(step)
        if multi_line:
            
            #multi-line plot
            wandb.log({key : wandb.plot.line_series(
                    xs=time_steps_for_plotting, 
                    ys=np.array(current_data).transpose().tolist(),
                    title=key,
                    xname="timesteps")})

        
    #average mitigation_rate
    for key in country_data:

        current_data = []
        time_steps_for_plotting = []
        for step in range(1,len(dict[type][key])):
            
            
            

            


            
            #get values for this timestep
            rates = [dict[type][key][step][idx] * num_discrete_action_levels for idx in range(len(dict[type][key][step]))]


            #store data for line plot
            current_data.append(rates)
            time_steps_for_plotting.append(step)

            #average across all countries
            wandb.log({f"average{key}":np.mean(rates)})

            #histogram p country
            wandb.log({
                f"mitigation{key}Histogram":wandb.Histogram(
                    np.array(rates) / step
                )
            })

        #multi-line plot
        wandb.log({key : wandb.plot.line_series(
                xs=time_steps_for_plotting, 
                ys=np.array(current_data).transpose().tolist(),
                keys=[f"region_{idx}" for idx in range(num_regions)],
                title=key,
                xname="timesteps")})


    wandb.finish()
        
