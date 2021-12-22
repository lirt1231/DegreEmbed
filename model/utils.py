#!/usr/bin/env python
# -*-coding:utf-8 -*-
# @file    :   utils.py
# @brief   :   Utility functions.
# @author  :   Haotian Li
# @email   :   lcyxlihaotian@126.com
from typing import List

import numpy as np
import torch


def in_top_k(targets: torch.LongTensor,
             preds: torch.Tensor,
             k: int
             ) -> torch.BoolTensor:
    """Return whether `targets` is in topk of `preds`.

    Args:
        `targets`: (batch_size, )
        `preds`: (batch_size, classes)

    Returns:
        (batch_size, )
    """
    topk = preds.topk(k)[1]  # topk() returns (values, indices)
    return (targets.unsqueeze(1) == topk).any(dim=1)


def get_sample_prediction(target: int,
                          predictions: np.ndarray,
                          id2ent: dict
                          ) -> List[List]:
    """Get entities that rank higher that the answer entity (tail)."""
    p_target = predictions[target]
    # Get entities whose logit is larger than that of target.
    pred = filter(lambda x: x[1] > p_target, enumerate(predictions))
    pred = sorted(pred, key=lambda x: x[1], reverse=True)
    pred.append((target, p_target))
    pred = [id2ent[targ] for targ, _ in pred]

    return pred
