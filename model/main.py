#!/usr/bin/env python
# -*-coding:utf-8 -*-
# @file    :   main.py
# @author  :   Haotian Li
# @email   :   lcyxlihaotian@126.com

import random

import numpy as np
from torch import optim

from configure import Configure
from dataloader import RTDataLoader
from framework import RTFramework
from framework import RTLoss
from rule_miner import RuleMiner


def main():
    configure = Configure()
    facts_file = configure.facts_file
    train_file = configure.train_file
    valid_file = configure.valid_file
    test_file = configure.test_file
    entities_file = configure.entities_file
    relations_file = configure.relations_file
    ckpt_dir = configure.checkpoint_dir
    prediction_file = configure.prediction_file
    top_k = configure.HP_top_k
    rank = configure.HP_rank
    num_steps = configure.HP_num_steps
    num_rnn_layers = configure.HP_num_rnn_layers
    entity_embedding_dim = configure.HP_entity_embedding_dim
    query_embedding_dim = configure.HP_query_embedding_dim
    query_rnn_hidden_size = configure.HP_query_rnn_hidden_size
    entity_rnn_hidden_size = configure.HP_entity_rnn_hidden_size
    seed = configure.HP_random_seed
    batch_size = configure.HP_batch_size
    threshold = configure.HP_threshold
    train_epochs = configure.HP_train_epochs
    num_sample_batches = configure.num_sample_batches
    lr = configure.HP_lr
    query_include_reverse = configure.query_include_reverse
    device = configure.device

    random.seed(seed)
    np.random.seed(seed)

    dataloader = RTDataLoader(
        relations_file, entities_file,
        facts_file, train_file,
        valid_file, test_file,
        query_include_reverse
    )
    print("Dataloader built.")

    configure.num_relations = dataloader.num_relations
    configure.num_operators = dataloader.num_operators
    configure.num_entities = dataloader.num_entities
    configure.save()
    print("Configuration saved.")

    miner = RuleMiner(
        rank, num_steps,
        configure.num_entities, configure.num_operators,
        dataloader.entity_degrees, query_include_reverse,
        entity_embedding_dim, query_embedding_dim,
        num_rnn_layers, query_rnn_hidden_size,
        entity_rnn_hidden_size
    ).to(device)
    print("Miner built.")

    optimizer = optim.Adam(miner.parameters(), lr=lr)
    loss_fn = RTLoss(threshold).to(device)
    framework = RTFramework(
        miner, optimizer, dataloader,
        loss_fn, device, ckpt_save_dir=ckpt_dir
    )
    print("Framework created.")

    framework.train(
        top_k, batch_size,
        num_sample_batches,
        train_epochs
    )

    if test_file:
        acc = framework.eval(
            "test", batch_size,
            top_k, prediction_file
        )
        print(f"[Testing finished] acc: {acc:.2f}")


if __name__ == "__main__":
    main()
