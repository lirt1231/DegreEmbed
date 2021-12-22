#!/usr/bin/env python
# -*-coding:utf-8 -*-
# @file    :   dataloader.py
# @author  :   Haotian Li
# @email   :   lcyxlihaotian@126.com
import os
from tqdm import tqdm

import numpy as np


class RTDataLoader(object):
    def __init__(self,
                 relations_file: str,
                 entities_file: str,
                 facts_file: str,
                 train_file: str,
                 valid_file: str,
                 test_file: str,
                 query_include_reverse: bool = True,
                 ) -> None:
        # Read relations and entities.
        self.rel2id, self.num_relations = self._read_relations_file(
            relations_file, query_include_reverse
        )
        self.id2rel = {ident: rel for rel, ident in self.rel2id.items()}
        self.num_queries = self.num_relations
        if query_include_reverse:
            self.num_queries = self.num_relations * 2
        self.num_operators = self.num_queries
        self.ent2id = self._read_entities_file(entities_file)
        self.id2ent = {ident: ent for ent, ident in self.ent2id.items()}
        self.num_entities = len(self.ent2id)
        self.query_include_reverse = query_include_reverse

        # Read dataset.
        self.facts = self._parse_triplets(facts_file)
        self.train = self._parse_triplets(train_file)
        self.valid = self._parse_triplets(valid_file)
        self.test = self._parse_triplets(test_file)
        self.triplets_eval = np.concatenate([self.facts, self.train])
        all_triplets = np.concatenate([self.facts, self.train, self.valid, self.test])

        # Type of edges incident to each entity for constructing DegreEmbed operators.
        self.entity_degrees = self._get_inout_degree(all_triplets)

    def one_epoch(self,
                  name: str,
                  batch_size: int,
                  num_sample_batches: int = 0,
                  shuffle: bool = False) -> tuple:
        """Load batch data for one epoch train | valid | test.

        :Args
            `name`: 'train' | 'valid' | 'test'
            `batch_size`: mini batch size
            `num_sample_batches`: max number of batches for one epoch
            `shuffle`: shuffle data inside batch

        :Returns
            `batch_size` of queries and corresponding matrices.
        """
        if name not in ["train", "valid", "test"]:
            raise Exception("{} cannot be loaded.".format(name))
        if (name == "valid" and self.valid is None) or\
                (name == "test" and self.test is None):
            raise Exception("{} not loaded.".format(name))

        samples = getattr(self, name)
        num_samples = len(samples)
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        batch_cnt = 0
        for batch_start in range(0, num_samples, batch_size):
            batch_cnt += 1
            if num_sample_batches != 0 and\
                    batch_cnt >= num_sample_batches and\
                    name == "train":
                break
            ids = indices[batch_start:batch_start+batch_size]
            this_batch = samples[ids]
            queries, heads, tails = self.__triplets_to_feed(this_batch)
            if name != "train":
                triplets = self.triplets_eval
            else:
                this_batch_set = set()
                for q, h, t in this_batch:
                    this_batch_set.add((q, h, t))
                extra_triplets = np.array([
                    (q, h, t)
                    for q, h, t in samples
                    if (q, h, t) not in this_batch_set
                ])
                triplets = np.concatenate([self.facts, extra_triplets])

            yield queries, heads, tails, triplets

    def _read_relations_file(self,
                             relations_file: str,
                             query_include_reverse: bool
                             ) -> dict:
        """Load relations and return relation-to-index map."""
        rel2id = {}
        with open(relations_file, 'r') as file:
            for line in tqdm(file, "Loading relations"):
                line = line.strip()
                rel2id[line] = len(rel2id)
        num_rel = len(rel2id)
        if query_include_reverse:
            for rel, ident in list(rel2id.items()):
                rel2id["inv_" + rel] = ident + num_rel
        return rel2id, num_rel

    def _read_entities_file(self, entities_file: str) -> dict:
        """Load entities and return entity-to-index map."""
        ent2id = {}
        with open(entities_file, 'r') as file:
            for line in tqdm(file, "Loading entities"):
                line = line.strip()
                ent2id[line] = len(ent2id)
        return ent2id

    def _parse_triplets(self, triplets_file: str):
        """Read triplets (relation, head, tail)."""
        triplets = []
        with open(triplets_file, 'r') as file:
            for line in tqdm(file, "Loading triplets"):
                line = line.strip().split('\t')
                assert(len(line) == 3)
                triplets.append(
                    (
                        self.rel2id[line[1]],
                        self.ent2id[line[0]],
                        self.ent2id[line[2]]
                    )
                )
        return np.array(triplets)

    def __triplets_to_feed(self, triplets: np.ndarray) -> tuple:
        """Separate samples to queries, heads and tails."""
        queries, heads, tails = zip(*triplets)
        queries = np.array(queries, dtype=np.int64)
        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        if self.query_include_reverse:
            queries = np.concatenate([queries, queries + self.num_relations])
            heads, tails = np.concatenate([heads, tails]), np.concatenate([tails, heads])
        return queries, heads, tails

    def _get_inout_degree(self, all_triplets: np.ndarray) -> list:
        """Get relational edges incident to each entity."""
        # (num_entities, 2, num_operators):
        # (num_entities, (in_degree, out_degree), num_operators)
        # Element indexed by `num_operators` for padding.
        degrees = np.full(
            (self.num_entities, 2, self.num_operators),
            fill_value=self.num_operators
        )
        for q, h, t in all_triplets:
            degrees[h, 1, q] = q
            degrees[t, 0, q] = q
        # (num_entities, num_operators*2)
        degrees = degrees.reshape((self.num_entities, -1))

        # Put paddings at the front.
        degrees = np.sort(degrees, axis=-1)[:, ::-1]

        return degrees.copy()


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "../datasets/")
    dataset = "family"
    dataset_dir = os.path.join(data_dir, dataset)
    entities_file = os.path.join(dataset_dir, "entities.txt")
    relations_file = os.path.join(dataset_dir, "relations.txt")
    facts_file = os.path.join(dataset_dir, "facts.txt")
    train_file = os.path.join(dataset_dir, "train.txt")
    valid_file = os.path.join(dataset_dir, "valid.txt")
    test_file = os.path.join(dataset_dir, "test.txt")

    dataloader = RTDataLoader(
        relations_file, entities_file,
        facts_file, train_file,
        valid_file, test_file, True
    )

    for bid, (qq, hh, tt, trips) in enumerate(dataloader.one_epoch(
        "train", 2, False
    )):
        print(qq.shape)
        print(hh.shape)
        print(tt.shape)
        print(hh)
        print(tt)
        print(trips[:5, :])
        print(dataloader.entity_degrees[:2, :])
        break
