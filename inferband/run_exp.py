import argparse
import csv
import numpy as np
import time
from tqdm import tqdm
import os

from exp_suite import synthetic_suite, dataset_suite, get_all_suites
from inferband.common import Stage
from inferband.server import Server
from inferband.trace import generate_requests, generate_requests_from_file


def get_data_file(dataset, small_model=None, large_model=None):
    if "lambda" in dataset:
        return "../data/lambda_opt_flops_clean_train.json"
    elif "oasst" in dataset:
        if small_model.startswith("flan-t5") and large_model.startswith("vicuna"):
            return "../data/oasst_a100_flan-t5_vicuna_clean_train.json"
        if small_model.startswith("fastchat-t5") and large_model.startswith("vicuna"):
            return "../data/oasst_a100_fastchat-t5_vicuna_clean_train.json"
    return None


def simulate(config, seed):
    (num_round, num_query, cache_size,
    cost_base, cost_ratio, cost_var, selector_acc,
    success_ratio, alpha, align_type, scenario,
    cache_strategy, selector,
    dataset, small_model, large_model,
    cost_dist,) = config

    # Generate requests.
    if dataset is None:
        requests = generate_requests(
                      num_round=num_round,
                      num_query=num_query,
                      cost_base=cost_base,
                      cost_ratio=cost_ratio,
                      cost_var=cost_var,
                      success_ratio=success_ratio,
                      alpha=alpha,
                      align_type=align_type,
                      seed=seed,
                      cost_dist=cost_dist,
                   )
    else:
        data_file = get_data_file(dataset, small_model, large_model)
        requests = generate_requests_from_file(
                      data_file=data_file,
                      num_round=num_round,
                      num_query=num_query,
                      success_ratio=success_ratio,
                      alpha=alpha,
                      align_type=align_type,
                      seed=seed,
                   )
    if len(requests) == 0:
        return None

    # Create a server.
    server = Server(
                 cache_size=cache_size,
                 cache_strategy=cache_strategy,
                 selector_acc=selector_acc,
                 selector=selector,
                 scenario=scenario,
                 requests=requests,
                 seed=seed,
             )

    # Start benchmarking.
    cost = 0
    for t in range(num_round):
        server.receive_request(requests[t])
        cost += server.step(requests[t], cost_dist)

    # Dump results
    # server.print_log(83, 84)
    return cost


def main(args):
    suites = get_all_suites(debug=args.debug)

    results = []
    for config in tqdm(suites, desc="suites"):
        cost = 0
        for i in range(args.N):
            if args.N > 0:
                seed = int(time.time())
            else:
                seed = args.seed
            cost += simulate(config, seed) / args.N
        results.append(list(config) + [cost])

    with open("all_exp_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--N", type=int, default=100)
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(time.time())
    main(args)
