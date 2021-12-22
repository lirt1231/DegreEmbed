#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @file     :   evaluate.py
# @brief    :   Evaluate model prediction results.
# @author   :   Haotian Li
# @email    :   lcyxlihaotian@126.com

import argparse
import os
from collections import defaultdict

import numpy as np


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="", type=str)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--rel', default=False, type=bool)
    parser.add_argument('--include_reverse', default=False, action="store_true")
    parser.add_argument('--filter', default=False, action="store_true")
    args = parser.parse_args()
    print(args)

    here = os.path.dirname(os.path.abspath(__file__))
    all_file = os.path.join(here, f"../datasets/{args.dataset}", "all.txt")
    preds = os.path.join("saved", args.dataset, "prediction.txt")

    # Read query to entities map.
    query2tails = defaultdict(set)
    invquery2heads = defaultdict(set)
    with open(all_file, 'r') as file:
        for line in file:
            h, q, t = line.strip().split('\t')
            query2tails[(h, q)].add(t)
            invquery2heads[(t, q)].add(h)

    # Read prediction file.
    lines = [line.strip().split(",") for line in open(preds).readlines()]
    line_cnt = len(lines)

    hits = 0
    hits_by_q = defaultdict(list)
    ranks = 0
    ranks_by_q = defaultdict(list)
    rranks = 0.

    for line in lines:
        assert(len(line) > 3)
        q, h, t = line[0:3]
        assert(line[-1] == t)
        this_preds = set(line[3:])

        if args.filter:
            if args.include_reverse and q.startswith("inv_"):
                q_ = q[len("inv_"):]
                also_correct = invquery2heads[(h, q_)]
            else:
                also_correct = query2tails[(h, q)]
            assert(t in also_correct)
            this_preds -= also_correct
            this_preds.add(t)

        hitted = 0.
        if len(this_preds) <= args.top_k:
            hitted = 1.
        rank = len(this_preds)

        hits += hitted
        ranks += rank
        rranks += 1. / rank
        hits_by_q[q].append(hitted)
        ranks_by_q[q].append(rank)

    print("Hits at %d is %0.4f" % (args.top_k, hits / line_cnt))
    print("Mean rank %0.2f" % (1. * ranks / line_cnt))
    print("Mean Reciprocal Rank %0.4f" % (1. * rranks / line_cnt))

    # Compute Hit@k and MRR for each relation.
    # [k, np.mean(v), len(v)]: [relation, hit@k, num_triplets]
    if args.rel:
        hits_by_q_mean = sorted([[k, np.mean(v), len(v)]
                                for k, v in hits_by_q.items()], key=lambda xs: xs[1], reverse=True)
        for xs in hits_by_q_mean:
            xs += [np.mean(ranks_by_q[xs[0]]), np.mean(1. / np.array(ranks_by_q[xs[0]]))]
            # print: relation, hit@k, num_triplets, mean rank, MRR
            print(", ".join([str(x) for x in xs]))


if __name__ == "__main__":
    evaluate()
