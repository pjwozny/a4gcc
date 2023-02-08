import json, os
import pandas as pd
from scipy.stats import f_oneway 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation


path_to_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fr" )
conditions = ["soft_defect","hard_defect", "cooperate", "control"]


features = ["reward", "abatement", "average_tariffs"]

json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
aggregate_data = []
for file in json_files:
    with open(os.path.join(path_to_json, file), "r") as f:
        pdf = pd.DataFrame(json.load(f))
        print(pdf.head())
        for group in conditions:
            pdf[f"normalized_reward_{group}"] = pdf.apply(lambda x: x[f"{group}_reward"] / x[f"{group}_labor"], axis = 1)
        aggregate_data.append(
            {
                **{"id":file.split("_")[1]},
                **{f"{group}_reward": pdf.loc[:,f"normalized_reward_{group}"].mean() for group in conditions},
                **{f"{group}_abatement":pdf.loc[:,f"{group}_abatement"].mean() for group in conditions},
                **{f"{group}_average_tariffs":pdf.loc[:,f"{group}_average_tariffs"].mean() for group in conditions},
            }
        )
agg_pdf = pd.DataFrame(aggregate_data)

sns.set(style="whitegrid")

for feature in features:

    mean_box = agg_pdf[[col for col in agg_pdf.columns if feature in col]]#.boxplot()
    print(pd.melt(mean_box).head())
    print("asdf")
    x="variable"
    y="value"
    ax = sns.boxplot(x="variable", y="value", data=pd.melt(mean_box))
    box_pairs = []
    for group in conditions:
        for group_ in conditions:
            if group_ != group:
                box_pairs.append((f"{group}_{feature}", f"{group_}_{feature}"))
            else:
                pass
    fset = set(frozenset(x) for x in box_pairs)
    box_pairs = [tuple(x) for x in fset]
    ax, test_results = add_stat_annotation(ax, data=pd.melt(mean_box), x=x, y=y,order=[col for col in agg_pdf.columns if feature in col],
                                box_pairs = box_pairs, test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    

    plt.title(feature)
    plt.show()

