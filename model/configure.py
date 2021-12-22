#!/usr/bin/env python
# -*-coding:utf-8 -*-
# @file    :   configure.py
# @brief   :   Paths and hyperparameter.
# @author  :   Haotian Li
# @email   :   lcyxlihaotian@126.com

import argparse
import os

import torch

# Absolute path where this file `configure` lies.
here = os.path.dirname(os.path.abspath(__file__))


class Configure(object):
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="Configuration")
        parser.add_argument("--dataset", default="family", type=str)
        parser.add_argument("--top_k", default=10, type=int)
        parser.add_argument("--rank", default=3, type=int)
        parser.add_argument("--num_steps", default=2, type=int)
        parser.add_argument("--seed", default=210224, type=int)
        parser.add_argument("--batch_size", default=128, type=int)
        parser.add_argument("--train_epochs", default=20, type=int)
        parser.add_argument("--num_sample_batches", default=0, type=int)
        parser.add_argument("--lr", default=0.001, type=float)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        parser.add_argument("--device", default=device, type=str)
        args = parser.parse_args()
        """Dataset paths."""
        self.data_dir = os.path.join(here, "../datasets/")
        self.dataset = args.dataset
        self.dataset_dir = os.path.join(self.data_dir, self.dataset)
        self.facts_file = os.path.join(self.dataset_dir, "facts.txt")
        self.train_file = os.path.join(self.dataset_dir, "train.txt")
        self.valid_file = os.path.join(self.dataset_dir, "valid.txt")
        self.test_file = os.path.join(self.dataset_dir, "test.txt")
        self.entities_file = os.path.join(self.dataset_dir, "entities.txt")
        self.relations_file = os.path.join(self.dataset_dir, "relations.txt")
        """Saved paths"""
        experiment_dir = os.path.join(here, "../saved", self.dataset)
        # Model checkpoint for continuing training.
        self.checkpoint_dir = os.path.join(experiment_dir, "checkpoint/")
        self.checkpoint_file = None
        # Directory to save trained model.
        self.model_save_dir = os.path.join(experiment_dir, "model/")
        # Options file.
        self.option_file = os.path.join(experiment_dir, "option.txt")
        # Model prediction file.
        self.prediction_file = os.path.join(experiment_dir, "prediction.txt")
        if not os.path.exists(experiment_dir):
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.model_save_dir)
        """Hypterparameters"""
        self.HP_top_k = args.top_k
        self.HP_rank = args.rank
        self.HP_num_steps = args.num_steps
        self.HP_entity_embedding_dim = 128
        self.HP_query_embedding_dim = 128
        self.HP_num_rnn_layers = 1
        self.HP_entity_rnn_hidden_size = 128
        self.HP_query_rnn_hidden_size = 128
        self.HP_random_seed = args.seed
        self.HP_batch_size = args.batch_size
        self.HP_threshold = 1e-20
        self.HP_train_epochs = args.train_epochs
        self.num_sample_batches = args.num_sample_batches
        self.HP_lr = args.lr

        """Other configurations"""
        self.query_include_reverse = True
        self.device = args.device

    def save(self):
        with open(self.option_file, 'w') as file:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                file.write(f"{key}, {value}\n")
