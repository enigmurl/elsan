import torch
import numpy as np
import torch.nn as nn
from torch.utils import data
import time
import sys

from hyperparameters import *
from model import CGAN
from train import ClusteredDataset, ClippingDataset, train_orthonet_epoch, train_clipping_epoch, EnsembleDataset, \
    train_base_orthonet_epoch
from util import get_device, write_parameters_into_model, save_parameters_from_model

device = get_device()
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # model = Orthonet(input_channels=O_INPUT_LENGTH * 2,
    #                  pruning_size=O_PRUNING_SIZE,
    #                  kernel_size=O_KERNEL_SIZE,
    #                  dropout_rate=O_DROPOUT_RATE,
    #                  time_range=O_TIME_RANGE
    #                  ).to(device)

    model = CGAN().to(device)
    org_model = model

    if len(sys.argv) > 1 and sys.argv[1] == 'recover':
        write_parameters_into_model(model, 'cgan_state.pt')
        model = model.to(device)

    loss_fun = torch.nn.MSELoss()

    goptimizer = torch.optim.Adam(model.parameters(), O_LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-3)
    doptimizer = torch.optim.Adam(model.parameters(), O_LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-3)

    train_emse = []
    valid_emse = []
    test_emse = []

    start = 0 if len(sys.argv) < 3 else int(sys.argv[2])
    best = 1e6
    for i in range(start, O_MAX_EPOCH):
        print("Epoch", i)

        start = time.time()

        torch.cuda.empty_cache()

        model.train()
        lc = model.train_epoch(128, [goptimizer, doptimizer], max_out_frame=i // 2 + 1)
        train_emse.append(np.mean(lc, axis=0))

        if np.mean(lc) < best or True:
            torch.save(model.state_dict(), "cgan.pt")
            save_parameters_from_model(model, 'cgan_state.pt')
            best = np.mean(lc)

        end = time.time()

        print("train emse", train_emse[-1], "minutes", round((end - start) / 60, 5))
