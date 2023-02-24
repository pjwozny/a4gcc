import pickle as pkl
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import os
from fixed_paths import PUBLIC_REPO_DIR

def constructStackedBarChart(global_states, wandb,
                            num_discrete_actions = 10, 
                            field = "mitigation_rate_all_regions"):

    """
    Constructs a Stacked Bar Chart for each timestep of a given run
    Note: this is only useful where there is a single value per country, 
        ie self-directed country actions, such as mitigation. 

    Args:
        global_state(dict): the global state dictionary of a completed run
        field(str): the name of the field to extract, defualt is minimum_mitigation_rates
        wandb(wandb-object): an already initialized and open wandb object
                indicating where the plot should be sent
    """


    rates_over_time = global_states[field]
    possible_rates = range(0,num_discrete_actions-1)
    to_plot = {rate:[] for rate in possible_rates}

    #simplify output to 1 per timestep otherwise the data is repeated per negotiation step
    run_length = 20
    step_size = int(rates_over_time.shape[0] / run_length)

    #per timestep get rate counts
    for timestep in range(0,rates_over_time.shape[0], step_size):

        #cast floats to ints and scale
        current_rates = [int(rate*num_discrete_actions) for rate in rates_over_time[timestep,:]]
        current_counter = Counter(current_rates)
        for rate in possible_rates:
            #count countries with a particular rate
            if rate in current_counter.keys():
                to_plot[rate].append(current_counter[rate])
            #if no countries have that particular rate
            else:
                to_plot[rate].append(0)

    #construct plot
    pdf = pd.DataFrame(to_plot)
    pdf.plot(kind='bar', stacked=True).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
    y_axis_field = field.replace("_all_regions", "")
    plt.ylabel(f"# of Countries of a Given {y_axis_field}")
    plt.xlabel("Timesteps")
    plt.title(f"{field} Distribution")
    wandb.log({f"{y_axis_field} Counts Across Time":plt})
       



