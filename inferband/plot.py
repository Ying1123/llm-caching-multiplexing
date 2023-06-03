import argparse
from collections import defaultdict
import csv
import json

from exp_suite import BenchmarkConfig, get_all_suites

column_names = list(BenchmarkConfig._fields) + ["tot_cost"]


def get_table_format(scenario, dataset=None):
    online = int(scenario == "offline")
    if dataset == "lambda" or dataset == "oasst":
        table_format = "\\begin{tabular}{ c" + "p{3.5em}" * online
    elif dataset is None:
        table_format = "\\begin{tabular}{ cc" + "p{3.5em}" * online
    else:
        raise Exception("unrecognized dataset")
    return table_format


def get_col_names(scenario, dataset=None):
    online = int(scenario == "offline")
    if dataset == "lambda" or dataset == "oasst":
        col_names = f"  $\\alpha$ & " + "selector accuracy &" * online
    elif dataset is None:
        col_names = f"  $\\alpha$ & cost ratio & " + "selector accuracy &" * online
    else:
        raise Exception("unrecognized dataset")
    return col_names


def get_param_cols(configs, scenario, dataset=None):
    online = int(scenario == "offline")
    acc = configs['selector_acc']
    if dataset == "lambda" or dataset == "oasst":
        cols = f"  {configs['alpha']} " + f"& \\multicolumn{{1}}{{r}}{{ {acc} }} " * online
    elif dataset is None:
        cols = f"  {configs['alpha']} & {configs['cost_ratio']} " + f"& \\multicolumn{{1}}{{r}}{{ {acc} }} " * online
    else:
        raise Exception("unrecognized dataset")
    return cols


def get_caption(scenario, dataset=None, small_model=None, large_model=None):
    caption = f"{scenario}"
    if dataset is not None:
        caption += f" {dataset}"
        if small_model is not None:
            caption += f" {small_model} {large_model}"
    else:
        caption += f" synthetic"
    return caption


def get_label(scenario, dataset=None, small_model=None, large_model=None):
    label = f"{scenario}"
    if dataset is not None:
        label += f"_{dataset}"
        if small_model is not None:
            label += f"_{small_model}_{large_model}"
    else:
        label += f"_synthetic"
    return label


def dumptex(results, scenario, dataset=None, small_model=None, large_model=None):
    texfile = scenario
    if dataset is not None:
        texfile += f"_{dataset}"
        if small_model is not None:
            texfile += f"_{small_model}_{large_model}"
    else:
        texfile += f"_synthetic"
    texfile += ".tex"

    if dataset is None:
        B1 = "LFU+ model 1"
        B2 = "LFU+ model 2"
        B3 = "LFU+ selector"
        B4 = "LEC+ model 1"
        B5 = "LEC+ model 2"
        B6 = "LEC+ selector"
    else:
        B1 = "LFU+ large"
        B2 = "LFU+ cascade"
        B3 = "LFU+ selector"
        B4 = "LEC+ large"
        B5 = "LEC+ cascade"
        B6 = "LEC+ selector"
    acc = -1
    with open(texfile, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\begin{center}\n")
        f.write(get_table_format(scenario, dataset=dataset) + 
                "p{3.2em}p{3.2em}p{3.2em}p{3.2em}p{3.2em}p{3.2em} }\n")
        f.write("  \\toprule\n")
        f.write(get_col_names(scenario, dataset=dataset) + 
                f" {B1} & {B2} & {B3} & {B4} & {B5} & {B6} \\\\ \n")
        for key, value in results.items():
            configs = json.loads(key)
            if configs["scenario"] != scenario:
                continue
            if dataset is None and configs["dataset"] != "":
                continue
            if dataset is not None and (configs["dataset"] != dataset or
               configs["small_model"] != small_model or configs["large_model"] != large_model):
                continue

            if configs["selector_acc"] != acc:
                f.write("  \\midrule\n")
                acc = configs["selector_acc"]

            f.write(get_param_cols(configs, scenario, dataset=dataset))
            key_of_min_value = min(value, key=value.get)
            for key, cost in value.items():
                if key == key_of_min_value or int(cost) == int(value[key_of_min_value]):
                    f.write(f"& \\multicolumn{{1}}{{r}}{{ \\textbf{{ {cost / 1000:.2f} }} }} ")
                else:
                    f.write(f"& \\multicolumn{{1}}{{r}}{{ {cost / 1000:.2f} }} ")
            f.write("\\\\ \n")
        f.write("  \\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{center}\n")
        f.write(f"\\caption{{{get_caption(scenario, dataset=dataset, small_model=small_model, large_model=large_model)}}}\n")
        f.write(f"\\label{{tab:{get_label(scenario, dataset=dataset, small_model=small_model, large_model=large_model)}}}\n")
        f.write("\\end{table}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-file", type=str, default="all_exp_results.csv")
    args = parser.parse_args()

    data = []
    with open(args.res_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[-1] != '':
                data.append(row)
    assert len(column_names) == len(data[0])
    print("number of data points:", len(data))

    strategies = [
        ("LFU", "large"),
        ("LFU", "cascade"),
        ("LFU", "ours"),
        ("ours", "large"),
        ("ours", "cascade"),
        ("ours", "ours"),
    ]
    results = {}
    for suite in data:
        config = dict(zip(column_names, suite))
        stg = (config["cache_strategy"], config["selector"])
        cost = float(config["tot_cost"])
        config.pop("cache_strategy")
        config.pop("selector")
        config.pop("tot_cost")
        config = json.dumps(config)

        if config not in results:
            results[config] = {}
        results[config][stg] = cost

    for key, value in results.items():
        print(key)
        for strategy, cost in value.items():
            print(f"{strategy}: {float(cost):.2f}")

    dumptex(results, "offline")
    dumptex(results, "online")

    dumptex(results, "offline", dataset="lambda", small_model="opt-1.3b", large_model="opt-13b")
    dumptex(results, "offline", dataset="oasst", small_model="flan-t5", large_model="vicuna")
    dumptex(results, "offline", dataset="oasst", small_model="fastchat-t5", large_model="vicuna")

    dumptex(results, "online", dataset="lambda", small_model="opt-1.3b", large_model="opt-13b")
    dumptex(results, "online", dataset="oasst", small_model="flan-t5", large_model="vicuna")
    dumptex(results, "online", dataset="oasst", small_model="fastchat-t5", large_model="vicuna")

