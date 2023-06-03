from collections import Counter
import json
import logging
import numpy as np
import pickle
from typing import List, Tuple, Any

import numpy as np

MIN = 0.1
FAIL_PENALTY = 2

logging.basicConfig(level=logging.ERROR)

class Query:
    def __init__(
        self,
        qid: int,
        cost_s: float,
        cost_l: float,
        cost_cas: float,
        cost_opt: float,
        success,
    ) -> None:
        self.qid = qid
        self.cost_s = cost_s
        self.cost_l = cost_l
        self.cost_cas = cost_cas
        self.cost_opt = cost_opt
        self.success = success

    def __repr__(self):
        return f"Query(qid={self.qid}, cost_s={self.cost_s:.2f}, cost_l={self.cost_l:.2f})"


class Request:
    def __init__(
        self,
        rid: int,
        query: Query,
        cost_s=None,
        cost_l=None,
        cost_cas=None,
        cost_opt=None,
        success=None,
    ):
        self.rid = rid
        self.qid = query.qid
        self.query = query
        self.cost_s = cost_s if cost_s is not None else query.cost_s
        self.cost_l = cost_l if cost_l is not None else query.cost_l
        self.cost_cas = cost_cas if cost_cas is not None else query.cost_cas
        self.cost_opt = cost_opt if cost_opt is not None else query.cost_opt
        self.success = success if success is not None else query.success

    def __repr__(self):
        return f"Request(rid={self.rid}, qid={self.qid}, " \
               f"cost_s={self.cost_s:.2f}, cost_l={self.cost_l:.2f})"


def generate_requests(
        num_round=None,
        num_query=None,
        cost_base=None,
        cost_ratio=None,
        cost_var=None,
        success_ratio=None, # does not support
        alpha=None,
        align_type=None,
        seed=0,
        cost_dist="binomial",
    ):

    np.random.seed(seed)

    queries = []
    if cost_dist == "binomial":
        cost_s = np.random.binomial(n=1, p=0.5, size=num_query) * cost_ratio + 1
        cost_l = np.random.binomial(n=1, p=0.5, size=num_query) * cost_ratio + 1
    for i in range(num_query):
        if cost_dist == "normal":
            success = np.random.randint(2)
            cost_s = np.maximum(MIN, np.random.normal(cost_base, np.sqrt(cost_var)))
            cost_l = np.maximum(MIN, np.random.normal(cost_base * cost_ratio, np.sqrt(cost_var)))
            cost_cas = cost_s + (success ^ 1) * cost_l * FAIL_PENALTY
            cost_opt = np.minimum(cost_cas, cost_l)
            queries.append(Query(i, cost_s, cost_l, cost_cas, cost_opt, success))

        elif cost_dist == "binomial":
            queries.append(Query(i, cost_s[i], cost_l[i], cost_s[i], min(cost_s[i], cost_l[i]), None))
        else:
            raise Exception("unrecognized distribution")

    # construct requests according to: num_round, alpha, align_type
    if align_type == "best":
        queries = sorted(queries, key=lambda query: query.real_cost())
    elif align_type == "worst":
        queries = sorted(queries, key=lambda query: query.real_cost(), reverse=True)
    elif align_type == "random":
        pass
    else:
        raise Exception("unrecognized align type")

    probs = np.random.power(alpha, num_round)
    indices = (probs * num_query).astype(int)

    requests = []
    for rid, qid in enumerate(indices):
        if cost_dist == "normal":
            success = np.random.randint(2)
            cost_s = np.maximum(MIN, np.random.normal(queries[qid].cost_s, np.sqrt(cost_var)))
            cost_l = np.maximum(MIN, np.random.normal(queries[qid].cost_l, np.sqrt(cost_var)))
            cost_cas = cost_s + (success ^ 1) * cost_l * FAIL_PENALTY
            cost_opt = np.minimum(cost_cas, cost_l)
            requests.append(Request(rid, queries[qid], cost_s, cost_l, cost_cas, cost_opt, success))
        elif cost_dist == "binomial":
            cost_s = np.maximum(MIN, queries[qid].cost_s + np.random.normal(0, 1))
            cost_l = np.maximum(MIN, queries[qid].cost_l + np.random.normal(0, 1))
            requests.append(Request(rid, queries[qid], cost_s, cost_l, cost_s, min(cost_s, cost_l), None))
        else:
            raise Exception("unrecognized distribution")

    # P = Counter([x.qid for x in requests])
    # n = 10
    # C = sorted(queries,
    #            key=lambda x : P[x.qid] * x.cost_opt,
    #            reverse=True)[:n]
    # print([x.qid for x in C])
    # print([P[C[i].qid] for i in range(n)])
    # print([C[i].cost_s for i in range(n)])
    # print([C[i].cost_l for i in range(n)])
    # print([P[C[i].qid] * min(C[i].cost_s, C[i].cost_l) for i in range(n)])

    # print()
    # print(sorted([x.qid for x in C]))
    # print(sorted([P[C[i].qid] * min(C[i].cost_s, C[i].cost_l) for i in range(n)]))

    # print()
    # C = sorted(queries,
    #            key=lambda x : P[x.qid],
    #            reverse=True)[:n]
    # print(sorted([x.qid for x in C]))
    # print(sorted([P[C[i].qid] * min(C[i].cost_s, C[i].cost_l) for i in range(n)]))

    return requests


def generate_requests_from_file(
        data_file=None,
        num_round=None,
        num_query=None,
        success_ratio=None,
        alpha=None,
        align_type=None,
        seed=0,
    ):

    np.random.seed(seed)
    assert success_ratio is None, "success_ratio has not been supported yet"

    if data_file is None:
        logging.warning("dataset file is None")
        return []

    with open(data_file, "r") as f:
        data = f.readlines()

    # extract queries
    cnt = [0, 0]
    queries = []
    for sample in data:
        query = json.loads(sample)
        if ((cnt[1] < num_query // 2 and query["correct_small"]) or
            (cnt[0] < num_query - num_query // 2 and not query["correct_small"])):
            cnt[query["correct_small"]] += 1
            cost_cas = query["cost_small"] + (query["correct_small"] ^ 1) * query["cost_large"] * FAIL_PENALTY
            cost_opt = np.minimum(cost_cas, query["cost_large"])
            queries.append(Query(len(queries),
                                 query["cost_small"],
                                 query["cost_large"],
                                 cost_cas,
                                 cost_opt,
                                 query["correct_small"]))
        if len(queries) == num_query: break

    # construct requests according to: num_round, alpha, align_type
    if align_type == "best":
        queries = sorted(queries, key=lambda query: query.real_cost())
    elif align_type == "worst":
        queries = sorted(queries, key=lambda query: query.real_cost(), reverse=True)
    elif align_type == "random":
        np.random.shuffle(queries)
    else:
        raise Exception("unrecognized align type")

    probs = np.random.power(alpha, num_round)
    indices = (probs * num_query).astype(int)
    requests = [queries[i] for i in indices]
    return requests


def generate_requests_debug(num_query, num_round):
    queries = []
    for i in range(num_query):
        if i < num_query // 2:
            query = Query(qid=i, cost_s=1, cost_l=2)
        else:
            query = Query(qid=i, cost_s=3, cost_l=2)
        queries.append(query)
    requests = []
    for i in range(num_round):
        requests.append(Request(i, queries[i % num_query]))
    return requests


def generate_requests_arbitrary(num_query: int, num_round: int,
                      debug=False):
    if debug:
        return generate_requests_debug(num_query, num_round)

    queries = []
    for i in range(num_query):
        cost_s = np.random.rand() + 1
        cost_l = np.random.rand() * 2 + 2
        query = Query(qid=i,
                      cost_s=cost_s + cost_l * np.random.randint(0, 2),
                      cost_l=cost_l)
        queries.append(query)

    requests = []
    for i in range(num_round):
        requests.append(Request(i, queries[np.random.randint(0, num_query)]))
    return requests
