from collections import namedtuple
import itertools

BenchmarkConfig = namedtuple(
    "BenchmarkConfig",
    [
     "num_round", "num_query", "cache_size",
     "cost_base", "cost_ratio", "cost_var", "selector_acc", # synthetic exp only
     "success_ratio", "alpha", "align_type", "scenario",
     "cache_strategy", "selector",
     "dataset", "small_model", "large_model", # dataset exp only
     "cost_dist",
    ]
)

synthetic_suite = {
    "default": BenchmarkConfig(
        num_round = [10000],
        num_query = [20],
        cache_size = [8],

        cost_base = [None],
        cost_ratio = [1.5, 100],
        cost_var = [None],
        selector_acc = [0.8, 1],

        # success_ratio = [0.2, 0.4, 0.5, 0.6, 0.8],
        success_ratio = [None],
        alpha = [0.5, 0.8],
        align_type = ["random"],
        scenario = ["offline", "online"],

        cache_strategy = ["LFU", "ours"],
        selector = ["large", "cascade", "ours"],

        dataset = [None],
        small_model = [None],
        large_model = [None],

        cost_dist = ["binomial"]
    ),
}

dataset_suite = {
    "lambda": BenchmarkConfig(
        num_round = [10000],
        num_query = [100],
        cache_size = [40],

        cost_base = [None],
        cost_ratio = [None],
        cost_var = [None],
        selector_acc = [0.8, 1],

        # success_ratio = [0.2, 0.4, 0.5, 0.6, 0.8],
        success_ratio = [None],
        alpha = [0.2, 0.8],
        align_type = ["random"],
        scenario = ["offline"],

        cache_strategy = ["LFU", "ours"],
        selector = ["large", "cascade", "ours"],

        dataset = ["lambda"],
        small_model = ["opt-1.3b"],
        large_model = ["opt-13b"],

        cost_dist = [None]
    ),
    "oasst": BenchmarkConfig(
        num_round = [10000],
        num_query = [100],
        cache_size = [40],

        cost_base = [None],
        cost_ratio = [None],
        cost_var = [None],
        selector_acc = [1],

        # success_ratio = [0.2, 0.4, 0.5, 0.6, 0.8],
        success_ratio = [None],
        alpha = [0.2, 0.5, 0.8],
        align_type = ["random"],
        scenario = ["online"],

        cache_strategy = ["LFU", "ours"],
        selector = ["large", "cascade", "ours"],

        dataset = ["oasst"],
        small_model = ["fastchat-t5"],
        large_model = ["vicuna"],

        cost_dist = [None]
    ),
}


debug_suite = {
    "default": BenchmarkConfig(
        num_round = [10000],
        num_query = [20],
        cache_size = [8],

        cost_base = [None],
        cost_ratio = [100],
        cost_var = [None],
        selector_acc = [1],

        # success_ratio = [0.2, 0.4, 0.5, 0.6, 0.8],
        success_ratio = [None],
        alpha = [0.8],
        align_type = ["random"],
        scenario = ["offline"],

        cache_strategy = ["ours"],
        selector = ["large", "cascade", "ours"],

        dataset = [None],
        small_model = [None],
        large_model = [None],

        cost_dist = ["binomial"]
    ),
}


def get_all_suites(debug=False):
    if debug:
        exps = [debug_suite]
    else:
        exps = [synthetic_suite, dataset_suite]

    suites = []
    for exp in exps:
        for workload in exp:
            (num_round_list, num_query_list, cache_size_list,
            cost_base_list, cost_ratio_list, cost_var_list, selector_acc_list,
            success_ratio_list, alpha_list, align_type_list, scenario_list,
            cache_strategy_list, selector_list,
            dataset_list, small_model_list, large_model_list,
            cost_dist,) = exp[workload]

            for combination in itertools.product(
                                   num_round_list, num_query_list, cache_size_list,
                                   cost_base_list, cost_ratio_list, cost_var_list, selector_acc_list,
                                   success_ratio_list, alpha_list, align_type_list, scenario_list,
                                   cache_strategy_list, selector_list,
                                   dataset_list, small_model_list, large_model_list,
                                   cost_dist,):
                # skip if scenario == "online" and selector == "ours" and selector_acc != 1
                if combination[10] == "online" and combination[6] < 1:
                    continue
                suites.append(combination)
    return suites

