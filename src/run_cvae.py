import torch
import numpy as np
import torch.nn as nn
from torch.utils import data
import time
import sys

from hyperparameters import *
from model import CVAE
from train import ClusteredDataset, ClippingDataset, train_orthonet_epoch, train_clipping_epoch, EnsembleDataset, \
    train_base_orthonet_epoch
from util import get_device, write_parameters_into_model, save_parameters_from_model

device = get_device()
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    model = CVAE().to(device)
    org_model = model

    if len(sys.argv) > 1 and sys.argv[1] == 'recover':
        write_parameters_into_model(model, 'cvae_state.pt')
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.5, 0.999))

    train_emse = []
    valid_emse = []
    test_emse = []

    start = 0 if len(sys.argv) < 3 else int(sys.argv[2])
    best = 1e6
    for i in range(start, 512):
        print("Epoch", i)

        start = time.time()

        torch.cuda.empty_cache()

        model.train()
        lc = model.train_epoch(128, optimizer, epoch_num=i)
        train_emse.append(np.mean(lc, axis=0))

        if np.mean(lc[:2]) < best or True:
            torch.save(model.state_dict(), "cvae.pt")
            save_parameters_from_model(model, 'cvae_state.pt')
            best = np.mean(lc[:2])

        end = time.time()

        print("train emse", train_emse[-1], "minutes", round((end - start) / 60, 5))
