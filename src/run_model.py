import torch
import torch.nn as nn
from torch.utils import data
import time
import sys

from hyperparameters import *
from model import Orthonet
from penalty import DivergenceLoss, BigErrorLoss
from train import ClusteredDataset, ClippingDataset, train_orthonet_epoch, train_clipping_epoch
from util import get_device, write_parameters_into_model, save_parameters_from_model
device = get_device()
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    model = Orthonet(input_channels=O_INPUT_LENGTH * 2,
                     pruning_size=O_PRUNING_SIZE,
                     kernel_size=O_KERNEL_SIZE,
                     dropout_rate=O_DROPOUT_RATE,
                     time_range=O_TIME_RANGE
                     ).to(device)

    if len(sys.argv) > 1 and (sys.argv[1] == 'clipping' or sys.argv[1] == 'recover'):
        write_parameters_into_model(model, 'model_state.pt')

    base = model.base
    trans = model.transition
    query = model.query
    clipping = model.clipping
    model = nn.DataParallel(model)

    train_set = ClusteredDataset(O_TRAIN_INDICES, O_TRAIN_DIREC, O_INPUT_LENGTH)
    clipping_set = ClippingDataset(C_TRAIN_INDICES, C_TRAIN_DIREC)

    # workers causing bugs on m1, likely due to lack of memory?
    train_loader = data.DataLoader(train_set, batch_size=O_BATCH_SIZE, shuffle=True, num_workers=0)
    clipping_loader = data.DataLoader(clipping_set, batch_size=C_BATCH_SIZE, shuffle=True, num_workers=0)

    loss_fun = torch.nn.MSELoss()
    error_fun = BigErrorLoss()
    regularizer = DivergenceLoss(torch.nn.MSELoss())

    optimizer = torch.optim.Adam(model.parameters(), O_LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    train_emse = []
    valid_emse = []
    test_emse = []

    if len(sys.argv) > 1 and sys.argv[1] == 'clipping':
        print("Training clipping")

        for i in range(O_MAX_EPOCH):
            print("Epoch", i)
            start = time.time()

            torch.cuda.empty_cache()

            model.train()

            emse = train_clipping_epoch(clipping_loader, clipping, optimizer)

            train_emse.append(emse)

            torch.save(model, "model.pt")
            save_parameters_from_model(model, 'model_state.pt')
            end = time.time()

            print("train emse", train_emse[-1], "minutes", round((end - start) / 60, 5))

    else:
        print("Training orthonet")

        for i in range(O_MAX_EPOCH):
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
            emse = train_orthonet_epoch(train_loader, base, trans, query, optimizer, error_fun)

            train_emse.append(emse)
            #
            # model.eval()
            # emse = eval_epoch(valid_loader, base, trans, query, hammer, con_list, error_fun)
            # valid_emse.append(emse)
            # valid_emse = [min_mse * 0.5]
            # test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, test_direc, True)

            torch.save(model, "model.pt")
            save_parameters_from_model(model, 'model_state.pt')
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
