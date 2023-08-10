import torch
import numpy as np
import torch.nn as nn
from torch.utils import data
import time
import sys

from hyperparameters import *
from model import ELSAN
from penalty import DivergenceLoss, BigErrorLoss, BaseErrorLoss
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

    model = ELSAN().to(device)
    org_model = model

    if len(sys.argv) > 1 and (sys.argv[1] == 'clipping' or sys.argv[1] == 'recover'):
        # state = torch.load('model.pt')
        # model.load_state_dict(state)
        write_parameters_into_model(model, 'model_state.pt')
        model = model.to(device)



    # train_set = EnsembleDataset(O_TRAIN_INDICES, O_RUN_SIZE, O_TRAIN_DIREC, O_INPUT_LENGTH)
    # base_set = ClusteredDataset(list(range(0, 32)), '../data/base/', O_INPUT_LENGTH)
    # clipping_set = ClippingDataset(C_TRAIN_INDICES, C_TRAIN_DIREC)

    # workers causing bugs on m1, likely due to lack of memory?
    # base_loader = data.DataLoader(base_set, batch_size=16, shuffle=True, num_workers=0)
    # train_loader = data.DataLoader(train_set, batch_size=O_BATCH_SIZE, shuffle=True, num_workers=0)
    # clipping_loader = data.DataLoader(clipping_set, batch_size=C_BATCH_SIZE, shuffle=True, num_workers=0)

    loss_fun = torch.nn.MSELoss()
    error_fun = BigErrorLoss()
    base_error = BaseErrorLoss()
    regularizer = DivergenceLoss(torch.nn.MSELoss())

    optimizer1 = torch.optim.Adam(model.parameters1(), O_LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-3)
    optimizer2 = torch.optim.Adam(model.parameters2(), O_LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-3)
    optimizer3 = torch.optim.Adam(model.parameters3(), O_LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

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
        # error = base(xx)
        # ran_sample(query, error, frames[:, 12:14])
        # prev_error = torch.zeros((64, 8, frames.shape[-2], frames.shape[-1]), device=device)
        # im = model(xx)
        # ran_sample(model, im, prev_error,
        #            frames[:, 60:62])
        lc = model.train_epoch(128, [optimizer1, optimizer2, optimizer3], max_out_frame=i // 2 + 1)
        train_emse.append(np.mean(lc))
        #
        # model.eval()
        # emse = eval_epoch(valid_loader, base, trans, query, hammer, con_list, error_fun)
        # valid_emse.append(emse)
        # valid_emse = [min_mse * 0.5]
        # test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, test_direc, True)

        if np.mean(lc) < best or True:
            torch.save(model.state_dict(), "model.pt")
            save_parameters_from_model(model, 'model_state.pt')
            best = np.mean(lc)
        #
        # if valid_emse[-1] < min_mse:
        #     min_mse = valid_emse[-1]
        #     best_model = model
        #     # test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
        #     # preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun, error_fun)
        #     #
        #     # torch.save({"preds": preds,
        #     #             "trues": trues,
        #     #             "loss_curve": loss_curve},
        #     #            "results.pt")

        end = time.time()

        print("train emse", train_emse[-1], "minutes", round((end - start) / 60, 5))
