import numpy as np
import torch
import random
from src.lsh import lsh

STACK = 4
K = 50
MID = 36

MAX_INDEX = 6000

with torch.no_grad():
    # read data
    data = torch.load("rbc_data.pt")

    # standardization
    std = torch.std(data)
    avg = torch.mean(data)
    data = (data - avg)/std
    data = data[:, :, ::4, ::4]

    # divide each rectangular snapshot into 7 subregions
    # data_prep shape: num_subregions * time * channels * w * h
    data_prep = torch.stack([data[:, :, :, k*64:(k+1)*64] for k in range(7)])

    frames = []
    for i in range(7):
        hsh = lsh(data_prep[i])

        print([len(x) for x in hsh.values()])

        for groups in hsh.values():
            # discount any indices before mid
            # if group size is less than stack, then we necessarily pad it
            random.shuffle(groups)
            curr = list((filter(lambda x: x % data_prep.shape[1] >= MID and
                                                   x - MID + 1 <= data_prep.shape[1] - K, groups)))
            if len(curr) < STACK:
                curr = (curr * STACK)[:STACK]

            curr = (np.array(curr) - MID + 1) * 7 + i
            curr = list(filter(lambda x: x < MAX_INDEX, curr))
            if len(curr) < STACK:
                curr = (curr * STACK)[:STACK]
            for k in range(len(curr) - STACK + 1):
                frames.append(curr[k:k + STACK])

    frames = torch.tensor(np.array(frames))
    print(len(frames))
    # use sliding windows to generate 9870 samples
    # training 6000, validation 2000, test 1870
    for j in range(data_prep.shape[1] - K + 1):
        for i in range(7):
            torch.save(data_prep[i, j: j + K].clone(), "data/data_64/sample_" + str(j*7 + i) + ".pt")
    torch.save(frames, "data/lsh_indices.pt")
