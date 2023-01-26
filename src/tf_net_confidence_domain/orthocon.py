import torch
import torch.nn as nn

DIMENSION = 64 * 64
SUBTRACTIONS = 64

# orthocon generation hyperparameters
MID_VARIATION = 0.05


def rand_orthoconvex(batch):
    """
    start with unit hypercube and keep cutting off sections
    """
    # random directions we will use for subtration starting points
    heads = torch.rand((batch, SUBTRACTIONS, DIMENSION))
    mids = (torch.rand((batch, SUBTRACTIONS, DIMENSION)) - 0.5) * MID_VARIATION + 0.5
    norms = torch.nn.functional.normalize(heads - mids)

    minimum = -1
    maximum = torch.min(torch.abs(norms), dim=2)

    min_dot = torch.rand((batch, SUBTRACTIONS)) * (maximum - minimum) + minimum

    return heads, norms, min_dot


def solve(cones, queries):
    """
    manually solves the question of min maxes given certain data
    Given orthant-like subtractions, determine the
    min and max for everything using all other information
    """
    # hmmmm it would be a huge optimization to get rid of the for loop, but this is a data processing thing that
    # seems like we inherently can't avoid


# Encoder (maps orthoconvex set to pruning vector)
class OrthoConEncoder(nn.Module):
    def __init__(self):
        super(OrthoConEncoder, self).__init__()

    def forward(self, convex, subtractions):
        pass


# Pruner (takes pruning vector and ranges and outputs range primes)
class OrthoConPruner(nn.Module):
    def __init__(self):
        super(OrthoConPruner, self).__init__()

    def forward(self, pruning_vector, in_ranges):
        pass


def train(model):
    pass


if __name__ == '__main__':
    train(None)
