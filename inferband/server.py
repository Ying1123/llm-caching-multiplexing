from collections import Counter, defaultdict
import numpy as np
import os
import time
from typing import List, Optional, Tuple

from inferband.common import Stage, Choice
from inferband.trace import Query


class Server:
    def __init__(
            self,
            cache_size=None,
            cache_strategy=None,
            selector_acc=None,
            selector=None,
            scenario=None,
            requests=None,
            seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)
        # params
        self.cache_size = cache_size
        self.cache_strategy = cache_strategy
        self.selector_acc = selector_acc
        self.selector = selector
        self.scenario = scenario

        if scenario == "offline":
            self._init_offline(requests)
        elif scenario == "online":
            self._init_online()
        else:
            raise Exception("unrecognized scenario")

        # server log
        self.log = []


    def _init_online(self):
        # init cost: qid -> {cost, cnt}
        self.cs = {}
        self.cl = {}
        self.cas = {}
        self.opt = {}

        # init cache
        self.C = [] # [qid,...]
        # init past observations
        # self.H = []
        # init policy
        self.pi = {} # qid -> {SMALL, LARGE}
        # init density function
        self.tot_cnt = 0
        self.P = {} # qid -> cnt


    def _init_offline(self, requests):
        # TODO currently using ground truth

        # density function
        self.P = Counter([x.qid for x in requests])
        self.tot_cnt = len(requests)

        # query cost
        self.cs = defaultdict(lambda: [0, 0])
        self.cl = defaultdict(lambda: [0, 0])
        self.cas = defaultdict(lambda: [0, 0])
        self.opt = defaultdict(lambda: [0, 0])
        for request in requests:
            qid = request.qid

            cnt = self.cs[qid][1]
            self.cs[qid] = ((self.cs[qid][0] * cnt + request.cost_s) / (cnt + 1), cnt + 1)

            cnt = self.cl[qid][1]
            self.cl[qid] = ((self.cl[qid][0] * cnt + request.cost_l) / (cnt + 1), cnt + 1)

            cnt = self.cas[qid][1]
            self.cas[qid] = ((self.cas[qid][0] * cnt + request.cost_cas) / (cnt + 1), cnt + 1)

            cnt = self.opt[qid][1]
            self.opt[qid] = ((self.opt[qid][0] * cnt + request.cost_opt) / (cnt + 1), cnt + 1)

        # cache
        if self.cache_strategy == "LFU":
            self.C = sorted(self.P, key=self.P.get, reverse=True)[:self.cache_size]
            # print("count", [self.P[x] for x in sorted(self.C)])
            # print([int(self.cs[x][0]) for x in sorted(self.C)])
            # print([int(self.cl[x][0]) for x in sorted(self.C)])
            # print([int(self.cascade[x][0]) for x in sorted(self.C)])
            # print("optimal", [int(self.optimal[x][0]) for x in sorted(self.C)])
            # print("cache", sorted(self.C))

        elif self.cache_strategy == "ours":
            if self.selector == "ours":
                self.C = sorted(self.P.items(),
                                key=lambda x : x[1] * self.opt[x[0]][0],
                                reverse=True
                               )[:self.cache_size]
            elif self.selector == "large":
                self.C = sorted(self.P.items(),
                                key=lambda x : x[1] * self.cl[x[0]][0],
                                reverse=True
                               )[:self.cache_size]
            elif self.selector == "cascade":
                self.C = sorted(self.P.items(),
                                key=lambda x : x[1] * self.cas[x[0]][0],
                                reverse=True
                               )[:self.cache_size]
            self.C = [x[0] for x in self.C]
            # print("count", [self.P[x] for x in sorted(self.C)])
            # print([int(self.cs[x][0]) for x in sorted(self.C)])
            # print([int(self.cl[x][0]) for x in sorted(self.C)])
            # print([int(self.cascade[x][0]) for x in sorted(self.C)])
            # print("optimal", [int(self.optimal[x][0]) for x in sorted(self.C)])
            # print("cache", sorted(self.C))
        else:
            raise Exception("unrecognized cache strategy")


    def receive_request(
        self,
        request,
    ):
        if self.scenario == "offline":
            return
        elif self.scenario == "online":
            qid = request.qid
            # update density estimation
            self.tot_cnt += 1
            if qid not in self.P:
                self.P[qid] = 1
            else:
                self.P[qid] += 1
        else:
            raise Exception("unrecognized scenario")


    def step(self, request: Query, cost_dist, stage: Stage=None):
        if self.scenario == "offline":
            # hit cache
            if self.hit_cache(request):
                self.log.append((request, stage, "hit cache", 0))
                return 0
            # get cost
            if self.selector == "large":
                self.log.append((request, stage, Choice.LARGE, request.cost_l))
                return request.cost_l

            elif self.selector == "cascade":
                if request.success:
                    self.log.append((request, stage, Choice.SMALL, request.cost_cas))
                else:
                    self.log.append((request, stage, Choice.BOTH, request.cost_cas))
                return request.cost_cas

            elif self.selector == "ours":
                assert self.selector_acc is not None
                coin = (np.random.uniform(0, 1) < self.selector_acc)
                if coin == 1:
                    cost = request.cost_opt
                else:
                    cost = max(request.cost_cas, request.cost_l)
                self.log.append((request, stage, None, cost))
            else:
                raise Exception("unrecognized selector")

            return cost

        elif self.scenario == "online":
            # hit cache
            if self.hit_cache(request):
                self.log.append((request, stage, "hit cache", 0))
                return 0
            # get cost
            choice, cost = self.select(request)
            # update past observation
            # self.H.append((request, choice, cost))
            self.log.append((request, stage, choice, cost))
            # update RegressionOracle
            self.update_RegressionOracle(request, choice, cost_dist)
            self.update_policy(request, cost_dist)
            # update cache
            self.update_cache(request, cost_dist)

            return cost


    def select(self, request: Query):
        if self.selector == "large":
            return Choice.LARGE, request.cost_l

        elif self.selector == "cascade":
            if request.success:
                return Choice.SMALL, request.cost_cas
            else:
                return Choice.BOTH, request.cost_cas

        elif self.selector == "ours":
            if request.qid not in self.pi:
                self.pi[request.qid] = Choice.SMALL
    
            if self.pi[request.qid] == Choice.SMALL:
                if request.success:
                    return Choice.SMALL, request.cost_cas
                else:
                    return Choice.BOTH, request.cost_cas
            else:
                return Choice.LARGE, request.cost_l
        else:
            raise Exception("unrecognized selector")


    def update_RegressionOracle(self, request: Query, choice: Choice, cost_dist):
        qid = request.qid
        if qid not in self.cs:
            self.cs[qid] = (0, 0)
            self.cl[qid] = (0, 0)
            self.cas[qid] = (0, 0)
            self.opt[qid] = (0, 0)

        if cost_dist == "binomial":
            if choice == Choice.SMALL or choice == Choice.BOTH:
                cnt = self.cs[qid][1]
                self.cs[qid] = ((self.cs[qid][0] * cnt + request.cost_s) / (cnt + 1), cnt + 1)
            elif choice == Choice.LARGE:
                cnt = self.cl[qid][1]
                self.cl[qid] = ((self.cl[qid][0] * cnt + request.cost_l) / (cnt + 1), cnt + 1)

        if choice == Choice.SMALL or choice == Choice.BOTH:
            cnt = self.cs[qid][1]
            self.cs[qid] = ((self.cs[qid][0] * cnt + request.cost_s) / (cnt + 1), cnt + 1)

            cnt = self.cas[qid][1]
            self.cas[qid] = ((self.cas[qid][0] * cnt + request.cost_cas) / (cnt + 1), cnt + 1)

            cnt = self.opt[qid][1]
            self.opt[qid] = ((self.opt[qid][0] * cnt + request.cost_opt) / (cnt + 1), cnt + 1)
    
        if choice == Choice.LARGE or not request.success:
            cnt = self.cl[qid][1]
            self.cl[qid] = ((self.cl[qid][0] * cnt + request.cost_l) / (cnt + 1), cnt + 1)


    def update_policy(self, request: Query, cost_dist):
        qid = request.qid
        if cost_dist == "binomial":
            if self.cl[qid][0] < self.cs[qid][0]:
                self.pi[qid] = Choice.LARGE
            else:
                self.pi[qid] = Choice.SMALL

        if self.cl[qid][1] > 0 and self.cas[qid][0] > self.cl[qid][0]:
            self.pi[qid] = Choice.LARGE 
        else:
            self.pi[qid] = Choice.SMALL


    def hit_cache(self, request):
        return request.qid in self.C


    def weight(self, qid, selector, cost_dist):
        if selector == "ours":
            if cost_dist == "binomial":
                if self.cl[qid][0] == 0:
                    return self.P[qid] / self.tot_cnt * self.cs[qid][0]
                return self.P[qid] / self.tot_cnt * min(self.cs[qid][0], self.cl[qid][0])

            if self.opt[qid][1] == 0:
                return self.P[qid] / self.tot_cnt * self.cs[qid][0]
            return self.P[qid] / self.tot_cnt * self.opt[qid][0]
        
        elif selector == "large":
            return self.P[qid] / self.tot_cnt * self.cl[qid][0]

        elif selector == "cascade":
            if cost_dist == "binomial":
                return self.P[qid] / self.tot_cnt * self.cs[qid][0]
            else:
                return self.P[qid] / self.tot_cnt * self.cas[qid][0]


    def update_cache(self, request, cost_dist):
        qid = request.qid
        # add if not full
        if len(self.C) < self.cache_size:
            self.C.append(qid)
            return
 
        # replace if full
        if self.cache_strategy == "LFU":
            ind, q = min(enumerate(self.C), key=lambda q: self.P[q[1]])
            if self.P[qid] > self.P[q]:
                self.C[ind] = qid

        elif self.cache_strategy == "ours":
            ind, q = min(enumerate(self.C), key=lambda q: self.weight(q[1], self.selector, cost_dist))
            if self.weight(qid, self.selector, cost_dist) > self.weight(q, self.selector, cost_dist):
                self.C[ind] = qid
 

    def print_log(self, n1=None, n2=None):
        if n1 is not None and n2 is not None:
            logs = self.log[n1:n2]
            n = n2
        elif n1 is not None:
            logs = self.log[:n1]
            n = n1
        else:
            logs = self.log
            n = len(logs)
        
        for req, stage, choice, cost in logs:
            print(f"P: {self.P[req.qid]}")
            print(f"cs: {self.cs[req.qid]} cl: {self.cl[req.qid]} cost: {self.cascade[req.qid]}")
            if self.scenario == "online":
                print(f"{self.pi[req.qid]}")
            print(f"{req}, {stage}, {choice}, {cost:.2f}")
        csum = sum([x[3] for x in self.log[:n]])
        print(f"sum till {n}: {csum:.2f}")

