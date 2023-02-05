import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import time
from model import CLES, LES, con_list
from penalty import DivergenceLoss, BigErrorLoss
from train import Dataset, train_epoch, eval_epoch, test_epoch
from util import get_device
from hammer_scheduler import HammerSchedule
from statistics import NormalDist

torch.multiprocessing.set_sharing_strategy('file_system')

device = get_device()

train_direc = "../data/data_64/sample_"
test_direc = "../data/data_64/sample_"

# best_params: kernel_size 3, learning_rate 0.001, dropout_rate 0, batch_size 120, input_length 25, output_length 4
time_range = 6
output_length = 4
input_length = 25
learning_rate = 1e-3
dropout_rate = 0
kernel_size = 3
batch_size = 64
pruning_size = 2
coef = 0

train_indices = list(range(0, 6000))
valid_indices = list(range(6000, 7700))
test_indices = list(range(7700, 9800))

if __name__ == '__main__':

    model = CLES(input_channels=input_length * 2, output_channels=2, kernel_size=kernel_size,
                 dropout_rate=dropout_rate, time_range=time_range, pruning_size=pruning_size,
                 orthos=len(con_list)).to(device)
    orthonet = model.ortho_cons
    model = nn.DataParallel(model)

    train_set = Dataset(train_indices, input_length + time_range - 1, 30, output_length, train_direc, True)
    valid_set = Dataset(valid_indices, input_length + time_range - 1, 30, 6, test_direc, True)
    # workers causing bugs on m1x, likely due to lack of memory
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

    loss_fun = torch.nn.MSELoss()
    error_fun = BigErrorLoss()
    regularizer = DivergenceLoss(torch.nn.MSELoss())
    hammer = HammerSchedule(lorris=1e-2, lorris_buffer=1e-3, lorris_decay=2e-3,
                            hammer=0e-1, hammer_buffer=5e-3, hammer_decay=2e-3)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    train_mse = []
    train_emse = []
    train_lorris = []
    train_p = []
    valid_mse = []
    valid_emse = []
    valid_p = []
    valid_lorris = []
    test_mse = []
    min_mse = 100

    for i in range(1000):
        print("Epoch", i)

        start = time.time()

        torch.cuda.empty_cache()

        model.train()
        mse, emse, p, lorris = train_epoch(train_loader, model, orthonet, optimizer, hammer, con_list,
                                           loss_fun, error_fun,
                                           pruning_size,
                                           coef, regularizer)

        train_mse.append(mse)
        train_lorris.append(lorris)
        train_emse.append(emse)
        train_p.append(p)

        model.eval()
        mse, emse, p_valid, lorris, preds, trues = eval_epoch(valid_loader, model, orthonet, hammer,
                                                              con_list, loss_fun, error_fun, pruning_size)
        valid_mse.append(mse)
        valid_emse.append(emse)
        valid_p.append(p_valid)
        valid_lorris.append(lorris)

        if valid_mse[-1] + valid_emse[-1] < min_mse:
            min_mse = valid_mse[-1] + valid_emse[-1]
            best_model = model
            torch.save(best_model, "model.pt")

            # test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, test_direc, True)
            # test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
            # preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun, error_fun)
            #
            # torch.save({"preds": preds,
            #             "trues": trues,
            #             "loss_curve": loss_curve},
            #            "results.pt")

        end = time.time()

        print("train mse", train_mse[-1], "train emse", train_emse[-1], "train p", train_p[-1],
              "train lorris", train_lorris[-1],
              "valid mse", valid_mse[-1], "valid emse", valid_emse[-1], "valid p", valid_p[-1],
              "valid lorris", valid_lorris[-1],
              "minutes", round((end - start) / 60, 5))

        if len(train_mse) > 75 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5]):
            break
