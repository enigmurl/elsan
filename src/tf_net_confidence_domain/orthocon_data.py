import torch
import torch.nn as nn
import numpy as np
import os

from util import get_device

device = get_device(no_mps=True)

BATCHES = 1024
BATCH = 1
DIMENSION = 64
SUBTRACTIONS = 128

# so it seems hypercones just arent going to cut it. we might want to take advantage of the symmetry of the system
# orthocon generation hyperparameters
MINIMUM_DOT = -0.8   # a value of negative -1 implies full space, 0 implies half space
MID_VARIATION = 0.0  # measure out how random the cone orientations are (with 0 being always point away from center)
NUM_SAMPLES = 2048
ORTHANT_BIAS = 2  # bias towards an orthant as opposed to a full space
BATCH_BIAS = 10  # bias towards a smaller region
SAVE = "../../data/orthocon/"


def rand_orthoconvex(batch):
    """
    start with unit hypercube and keep cutting off sections
    """
    # random directions we will use for subtration starting points
    heads = torch.rand((batch, SUBTRACTIONS, DIMENSION)).to(device)
    mids = (torch.rand((batch, SUBTRACTIONS, DIMENSION)) - 0.5) * MID_VARIATION + 0.5
    mids = mids.to(device)
    norms = nn.functional.normalize(heads - mids, dim=2)

    minimum = MINIMUM_DOT
    maximum = torch.min(torch.abs(norms), dim=2).values

    max_dot = (torch.rand((batch, SUBTRACTIONS)).to(device) ** ORTHANT_BIAS) * (minimum - maximum) + maximum
    return heads, norms, max_dot


def rand_queries(batch):
    """
    shape of (batch, DIMENSION, 2)
    """
    left = torch.rand((batch, DIMENSION, 1)).to(device)
    right = (torch.rand((batch, DIMENSION, 1)) ** BATCH_BIAS).to(device) * (1 - left) + left

    left = torch.zeros((batch, DIMENSION, 1))
    right = torch.ones((batch, DIMENSION, 1))

    indices = (torch.rand(batch) * DIMENSION).long()
    left[torch.arange(batch), indices] = 1
    right[torch.arange(batch), indices] = 0

    return torch.concat((left, right), dim=2), indices


def solve(heads, norms, dots, queries, off_indices):
    """
    manually solves the question of min maxes given certain data
    Given orthant-like subtractions, determine the
    min and max for everything using all other information
    Huge potential to be optimized (and more accurate), but since training data can be cached, no real point for now
    """

    batch = queries.shape[0]
    batch_range = torch.arange(batch)

    result = torch.zeros((batch, 2)).to(device)

    count = 0

    axis = off_indices
    queries[batch_range, axis, 0] = 0
    queries[batch_range, axis, 1] = 1

    width = torch.unsqueeze(queries[:, :, 1] - queries[:, :, 0], dim=1)
    samples = torch.rand((batch, NUM_SAMPLES, DIMENSION)).to(device) * width + \
        torch.unsqueeze(queries[:, :, 0], dim=1)

    # hyperclone clipping
    inside = torch.ones((batch, NUM_SAMPLES), dtype=torch.bool).to(device)
    for sub in range(SUBTRACTIONS):
        delta = nn.functional.normalize(samples - torch.unsqueeze(heads[:, sub], dim=1), dim=2)
        dot = torch.sum(delta * torch.unsqueeze(norms[:, sub], dim=1), dim=2)
        inside = torch.logical_and(inside, dot < torch.unsqueeze(dots[:, sub], dim=1))

    count += torch.sum(inside)

    # take the mins and maxes of the ones that are inside
    sentinel = samples[batch_range, :, axis]
    sentinel[torch.logical_not(inside)] = 1
    result[:, 0] = torch.min(sentinel, dim=1).values

    sentinel[torch.logical_not(inside)] = 0
    result[:, 1] = torch.max(sentinel, dim=1).values

    print("Proportion", count, batch * NUM_SAMPLES, result)

    return result


def load_data():
    for b in range(BATCHES):
        heads, norms, dots = rand_orthoconvex(BATCH)
        queries, indices = rand_queries(BATCH)

        expected = solve(heads, norms, dots, queries, indices)


if __name__ == '__main__':
    load_data()
