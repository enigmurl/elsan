# find collisions
# we want them to be similar, but also from theoretically different timestamps to get some resemblance of variety
# cyclic data is actually really good for this, which appears to be in the data set so we should be good actually
import torch

ENTROPY = 12


def rand_vector(count, dimension):
    # use normal distributions
    vectors = torch.normal(0, 1, size=(count, dimension))
    mags = torch.norm(vectors, dim=1)
    return vectors / torch.unsqueeze(mags, dim=1)


def local_hash(frame, planes):
    vector = torch.flatten(frame)
    string = ""
    for p in planes:
        string += "1" if torch.dot(p, vector) > 0 else "0"

    return string


def lsh(frames):
    planes = rand_vector(ENTROPY, torch.numel(frames[0]))
    distances = torch.rand()  # could potentially use something more advanced here?
    # generate a bunch of hyperplanes
    # create a bit string based on hyperplane results
    map = {}
    for i, f in enumerate(frames):
        map.setdefault(local_hash(f, planes), [])
        map[local_hash(f, planes)].append(i)

    return map