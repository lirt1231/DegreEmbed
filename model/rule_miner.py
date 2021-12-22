#!/usr/bin/env python
# -*-coding:utf-8 -*-
# @file    :   rule_miner.py
# @brief   :   Rule miner model.
# @author  :   Haotian Li
# @email   :   lcyxlihaotian@126.com

import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataloader import RTDataLoader


class RuleMiner(nn.Module):

    def __init__(self,
                 rank: int,
                 num_steps: int,
                 num_entities: int,
                 num_operators: int,
                 entity_degrees: np.ndarray,
                 query_include_reverse: bool = True,
                 entity_embedding_dim: int = 128,
                 query_embedding_dim: int = 128,
                 num_rnn_layers: int = 1,
                 entity_rnn_hidden_size: int = 128,
                 query_rnn_hidden_size: int = 128
                 ) -> None:
        """
        Args:
            `rank`: rank of estimators.
            `num_step': number of RNN input time step
            `num_entities': total number of entities in the KG
            `num_operators': number of DegreEmbed operators (relations in KG)
            `entity_degrees`: type (relation) of incident edges of each entity
            `query_include_reverse`: whether include reverse queries
            `entity_embedding_dim`: dimension of each entity embedding 
            `query_embedding_dim`: dimension of each query embedding vector
            `num_rnn_layers`: number of RNN layers
            `entity_rnn_hidden_size`: hidden state size of RNN for entity
            `query_rnn_hidden_size`: hidden state size of RNN for query
        """
        super(RuleMiner, self).__init__()
        self.rank = rank
        self.num_entities = num_entities
        self.num_steps = num_steps
        self.num_operators = num_operators
        self.query_include_reverse = query_include_reverse
        self.num_rnn_layers = num_rnn_layers
        self.entity_embedding_dim = entity_embedding_dim
        self.query_embedding_dim = query_embedding_dim
        self.query_rnn_hidden_size = query_rnn_hidden_size
        self.entity_rnn_hidden_size = entity_rnn_hidden_size

        self.query_embedding = nn.Embedding(
            self.num_operators,
            self.query_embedding_dim
        )
        self.entity_embedding = nn.Embedding(
            self.num_operators+1,
            self.entity_embedding_dim,
            padding_idx=self.num_operators
        )
        self.entity_degrees = nn.Parameter(
            torch.from_numpy(entity_degrees).long(),
            requires_grad=False
        )

        self.query_rnns = nn.ModuleList([
            nn.LSTM(
                input_size=self.query_embedding_dim,
                hidden_size=self.query_rnn_hidden_size,
                num_layers=self.num_rnn_layers,
                bidirectional=True
            ) for _ in range(self.rank)
        ])
        self.entity_rnn = nn.LSTM(
            input_size=self.entity_embedding_dim,
            hidden_size=self.entity_rnn_hidden_size,
            num_layers=self.num_rnn_layers,
            bidirectional=True
        )

        self.query_linear = nn.Linear(
            self.query_rnn_hidden_size*2,
            self.num_operators+1
        )
        self.entity_linear = nn.Linear(
            self.entity_rnn_hidden_size*2,
            self.num_operators
        )

    def forward(self,
                queries: torch.LongTensor,
                heads: torch.LongTensor,
                facts_triplets: np.ndarray,
                ) -> torch.Tensor:
        """Forward calculation.

        Args:
            `queries`: query relations (batch_size, )
            `heads`: head entities (batch_size, )
            `facts_triplets`: factual triplets for constructing KG,
                we reconstruct DegreEmbed operators each batch.

        Returns:
            torch.Tensor(batch_size, num_entities).
        """
        device = "cuda" if heads.is_cuda else "cpu"
        # (num_steps, batch_size).
        queries = queries.view(1, -1)
        queries = torch.cat(
            [
                queries
                for _ in range(self.num_steps)
            ],
            dim=0
        )
        # (num_steps, batch_size, query_embedding_dim).
        query_embed = self.query_embedding(queries)

        # Do attention.
        self.query_attn_ops_list = []
        for r in range(self.rank):
            # `query_rnn_output`: (num_steps, batch_size, hidden_size*2).
            query_rnn_output, (_, _) = self.query_rnns[r](query_embed)
            # (num_steps, num_operators+1, batch_size, 1).
            query_attn = self.query_linear(query_rnn_output)
            query_attn = F.softmax(query_attn, -1)\
                .transpose(-2, -1).unsqueeze(-1)
            self.query_attn_ops_list.append(query_attn)

        # (num_entities, num_operators*2, entity_embedding_dim)
        entity_embed = self.entity_embedding(self.entity_degrees)
        # ==> (time_step, batch_size, H_in)
        entity_embed = entity_embed.transpose(0, 1)
        # `entity_rnn_output`: (2*num_layers, num_entities, hidden_size)
        _, (entity_rnn_output, _) = self.entity_rnn(entity_embed)
        # (num_entities, num_layers, hidden_size*2)
        entity_rnn_output = entity_rnn_output.transpose(0, 1)\
            .reshape((self.num_entities, self.num_rnn_layers, -1))

        # (num_entities, hidden_size*2)
        entity_rnn_output = torch.sum(entity_rnn_output, dim=1)
        self.entity_attention = F.softmax(self.entity_linear(entity_rnn_output), -1)

        operators_matrices = self._get_adjacency_matrices(
            facts_triplets, self.entity_attention, device
        )

        # A list of `rank` tensors,
        # each tensor: (batch_size, step, num_entities).
        memories_list = [
            F.one_hot(heads, self.num_entities)
             .float().unsqueeze(1).to(device)
            for _ in range(self.rank)
        ]

        # (batch_size, num_entities).
        logits = 0.0
        for r in range(self.rank):
            for t in range(self.num_steps):
                # (num_entities, batch_size).
                memory = memories_list[r][:, -1, :]

                # (num_operators+1, batch_size, 1).
                attn_ops = self.query_attn_ops_list[r][t]
                # (batch_size, num_entities).
                added_matrix_results = 0.0
                if self.query_include_reverse:
                    for op in range(self.num_operators // 2):
                        # `op_matrix`: (num_entities, num_entities).
                        # `op_attn`: (batch_size, 1).
                        for op_matrix, op_attn in zip(
                            [operators_matrices[op], operators_matrices[op].t()],
                            [attn_ops[op], attn_ops[op + self.num_operators // 2]]
                        ):
                            # (batch_size, num_entities).
                            product = torch.matmul(op_matrix.t(), memory.t()).t()
                            # (batch_size, num_entities).
                            added_matrix_results += product * op_attn
                else:
                    for op in range(self.num_operators):
                        # `op_matrix`: (num_entities, num_entities).
                        # `op_attn`: (batch_size, 1).
                        op_matrix = operators_matrices[op]
                        op_attn = attn_ops[op]
                        # (batch_size, num_entities).
                        product = torch.matmul(op_matrix.t(), memory.t()).t()
                        # (batch_size, num_entities).
                        added_matrix_results += product * op_attn

                added_matrix_results += memory * attn_ops[-1]
                norm = torch.maximum(torch.tensor(1e-20).to(device),
                                     torch.sum(added_matrix_results,
                                               dim=1, keepdim=True))
                added_matrix_results /= norm
                # each tensor: (batch_size, step, num_entities).
                memories_list[r] = torch.cat(
                    [memories_list[r],
                     added_matrix_results.unsqueeze(1)],
                    dim=1
                )
            logits += memories_list[r][:, -1, :]

        return logits

    def _get_adjacency_matrices(self,
                                triplets: np.ndarray,
                                entity_attns: torch.Tensor,
                                device: str,
                                ) -> dict:
        """Create DegreEmbed operators.

        :Args
            `triples`: (q, h, t) for constructing adjacency matrices
            `entity_attns`: (num_entities, num_operators)
            `device`: 'cuda' | 'cpu'
        """
        matrices = {
            r: ([[0, 0]], [0.], [self.num_entities, self.num_entities])
            for r in range(self.num_operators)
        }
        for rel, head, tail in triplets:
            value = entity_attns[head, rel]
            matrices[rel][0].append([head, tail])
            matrices[rel][1].append(value)

        operators_matrices = {
            rel: torch.sparse.FloatTensor(
                torch.LongTensor(matrices[rel][0]).t(),
                torch.FloatTensor(matrices[rel][1]),
                matrices[rel][2],
            ).to(device)
            for rel in matrices.keys()
        }
        return operators_matrices


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(here, "../datasets/family")
    facts_file = os.path.join(dataset_dir, "facts.txt")
    train_file = os.path.join(dataset_dir, "train.txt")
    valid_file = os.path.join(dataset_dir, "valid.txt")
    test_file = os.path.join(dataset_dir, "test.txt")
    entities_file = os.path.join(dataset_dir, "entities.txt")
    relations_file = os.path.join(dataset_dir, "relations.txt")

    dataloader = RTDataLoader(
        relations_file, entities_file,
        facts_file, train_file,
        valid_file, test_file, True
    )

    miner = RuleMiner(
        rank=3, num_steps=2,
        num_entities=dataloader.num_entities,
        num_operators=dataloader.num_queries,
        entity_degrees=dataloader.entity_degrees,
        query_include_reverse=True
    )

    for _, (qq, hh, tt, trips) in enumerate(
        dataloader.one_epoch("train", 2, 0, True)
    ):
        qq = torch.from_numpy(qq)
        hh = torch.from_numpy(hh)
        print(miner(qq, hh, trips).size())
        break
