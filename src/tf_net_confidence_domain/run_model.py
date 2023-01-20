from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import time
from model import LES, CLES
from penalty import DivergenceLoss, ErrorLoss
from train import Dataset, train_epoch, eval_epoch, test_epoch
from util import get_device

torch.multiprocessing.set_sharing_strategy('file_system')

device = get_device()

train_direc = "../../data/data_64/sample_"
test_direc = "../../data/data_64/sample_"

if __name__ == '__main__':

    #best_params: kernel_size 3, learning_rate 0.001, dropout_rate 0, batch_size 120, input_length 25, output_length 4
    min_mse = 1
    time_range = 6
    output_length = 4
    input_length = 26
    learning_rate = 0.001
    dropout_rate = 0
    kernel_size = 3
    batch_size = 32
    c = 0.25

    train_indices = list(range(0, 6000))
    valid_indices = list(range(6000, 7700))
    test_indices = list(range(7700, 9800))

    model = CLES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size,
                dropout_rate = dropout_rate, time_range = time_range).to(device)
    model = nn.DataParallel(model)

    train_set = Dataset(train_indices, input_length + time_range - 1, 40, output_length, train_direc, True)
    valid_set = Dataset(valid_indices, input_length + time_range - 1, 40, 6, test_direc, True)
    train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers = 8)
    valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle=False, num_workers = 8)
    loss_fun = torch.nn.MSELoss()
    error_fun = ErrorLoss(c)
    regularizer = DivergenceLoss(torch.nn.MSELoss())
    coef = 0

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    train_mse = []
    train_emse = []
    valid_mse = []
    valid_emse = []
    test_mse = []

    for i in range(100):
        print("Epoch", i)

        start = time.time()

        torch.cuda.empty_cache()

        model.train()
        mse, emse = train_epoch(train_loader, model, optimizer, loss_fun, error_fun, coef, regularizer)
        train_mse.append(mse)
        train_emse.append(emse)

        model.eval()
        mse, emse, preds, trues = eval_epoch(valid_loader, model, loss_fun, error_fun)
        valid_mse.append(mse)
        valid_emse.append(emse)

        if valid_mse[-1] + valid_emse[-1] < min_mse:
            min_mse = valid_mse[-1] + valid_emse[-1]
            best_model = model 
            torch.save(best_model, "model.pt")

        end = time.time()

        if len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5]):
            break

        print("train mse", train_mse[-1], "train emse", train_emse[-1],
              "valid mse", valid_mse[-1], "valid emse", valid_emse[-1],
              "minutes", round((end - start) / 60, 5)
              )
        print(time_range, min_mse)

        loss_fun = torch.nn.MSELoss()

        best_model = torch.load("model.pt")
        test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, test_direc, True)
        test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
        preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun, error_fun)

        torch.save({"preds": preds,
                    "trues": trues,
                    "loss_curve": loss_curve}, 
                    "results.pt")
