import json, os
import pandas as pd
from scipy.stats import f_oneway 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation

path_to_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tariff" )
tariff_rates = [5,6,7,8,9]
groups = ["pariah", "control"]

json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
aggregate_data = []
for file in json_files:
    with open(os.path.join(path_to_json, file), "r") as f:

        pdf = pd.DataFrame([d for d in json.load(f) if d != {}])
        for group in ["pariah", "control"]:
            for tariff_rate in tariff_rates:
                pdf[f"normalized_reward_{group}_{tariff_rate}"] = pdf.apply(lambda x: \
                        x[f"{group}_reward_{tariff_rate}"] \
                                / x[f"{group}_labor_{tariff_rate}"], axis = 1)
        aggregate_data.append(
            {
                **{"id":file.split("_")[1]},
                **{f"r_{group[0]}_{tariff_rate}":pdf.loc[:,f"normalized_reward_{group}_{tariff_rate}"].mean() 
                                        for group in groups for tariff_rate in tariff_rates},
                **{f"t_{group[0]}_{tariff_rate}":pdf.loc[:,f"{group}_tariffs_{tariff_rate}"].mean() 
                                        for group in groups for tariff_rate in tariff_rates}

            }
        )
agg_pdf = pd.DataFrame(aggregate_data)
print(agg_pdf.head())
features = ["r_", "t_"]
for feature in features:
    col_order = [col for col in agg_pdf.columns if feature in col]
    col_order.sort(key = lambda x: int(x.split("_")[-1]))
    mean_box = agg_pdf[col_order]
    print(mean_box.columns)

    x="variable"
    y="value"
    ax = sns.boxplot(x="variable", y="value", order = col_order, data=pd.melt(mean_box))
    box_pairs = []
    for tariff_rate in tariff_rates:
        for group in groups:
            box_pairs.append((f"{feature}p_{tariff_rate}", f"{feature}c_{tariff_rate}"))


    fset = set(frozenset(x) for x in box_pairs)
    box_pairs = [tuple(x) for x in fset]
    print(box_pairs)
    print(mean_box.columns)
    ax, test_results = add_stat_annotation(ax, data=pd.melt(mean_box), x=x, y=y,order=col_order,
                                box_pairs = box_pairs, test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    

    plt.title(feature)
    plt.show()
