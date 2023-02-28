# find collisions
# we want them to be similar, but also from theoretically different timestamps to get some resemblance of variety
# cyclic data is actually really good for this, which appears to be in the data set so we should be good actually
import torch

ENTROPY = 32
DIST_MAX = 2


def rand_vector(count, dimension):
    # use normal distributions
    vectors = torch.normal(0, 1, size=(count, dimension))
    mags = torch.norm(vectors, dim=1)
    return vectors / torch.unsqueeze(mags, dim=1)


def local_hash(frame, planes, distances):
    vector = torch.flatten(frame)
    string = ""
    for p, d in zip(planes, distances):
        string += "1" if torch.dot(p, vector) > d else "0"

    return string


def lsh(frames):
    planes = rand_vector(ENTROPY, torch.numel(frames[0]))
    distances = torch.rand(ENTROPY) * DIST_MAX  # could potentially use something more advanced here?
    # generate a bunch of hyperplanes
    # create a bit string based on hyperplane results
    dic = {}
    for i, f in enumerate(frames):
        key = local_hash(f, planes, distances)
        dic.setdefault(key, [])
        dic[key].append(i)

    return dic
