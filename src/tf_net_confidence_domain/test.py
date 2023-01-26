import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import time
from model import CLES, LES
from penalty import DivergenceLoss, ErrorLoss
from train import Dataset, train_epoch, eval_epoch, test_epoch
from util import get_device

torch.multiprocessing.set_sharing_strategy('file_system')

device = get_device()

train_direc = "../../data/data_64/sample_"
test_direc = "../../data/data_64/sample_"

if __name__ == '__main__':

    # best_params: kernel_size 3, learning_rate 0.001, dropout_rate 0, batch_size 120, input_length 25, output_length 4
    min_mse = 1
    time_range = 6
    output_length = 4
    input_length = 25
    learning_rate = 0.001
    dropout_rate = 0
    kernel_size = 3
    batch_size = 64
    c = 0.25
    e_coef = 0.25

    valid_indices = list(range(6000, 6500))
    test_indices = list(range(7700, 9800))

    model = torch.load('model.pt')

    valid_set = Dataset(valid_indices, input_length + time_range - 1, 40, 6, test_direc, True)
    # workers causing bugs on m1x, likely due to lack of memory
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8)
    loss_fun = torch.nn.MSELoss()
    error_fun = ErrorLoss(c, e_coef)
    regularizer = DivergenceLoss(torch.nn.MSELoss())
    coef = 0

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    valid_mse = []
    valid_emse = []
    valid_p = []
    test_mse = []

    start = time.time()

    torch.cuda.empty_cache()

    model.eval()
    mse, emse, p_valid, preds, trues = eval_epoch(valid_loader, model, loss_fun, error_fun)
    valid_mse.append(mse)
    valid_emse.append(emse)
    valid_p.append(p_valid)

    end = time.time()

    print("valid mse", valid_mse[-1], "valid emse", valid_emse[-1], "valid p", valid_p[-1],
          "minutes", round((end - start) / 60, 5))
