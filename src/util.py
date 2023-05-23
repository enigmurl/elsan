import torch
import numpy as np
from functools import cache
from scipy.optimize import linear_sum_assignment

from hyperparameters import *


def get_device(no_mps=False):
    if no_mps:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def mask_indices(batch, elems):
    return (torch.rand(batch) * elems).long()


def orthonet_model():
    from model import Orthonet

    device = get_device()
    model = Orthonet(input_channels=O_INPUT_LENGTH * 2,
                     pruning_size=O_PRUNING_SIZE,
                     kernel_size=O_KERNEL_SIZE,
                     dropout_rate=O_DROPOUT_RATE,
                     time_range=O_TIME_RANGE
                     ).to(device)
    write_parameters_into_model(model, 'model_state.pt')
    return model.to(device)


# This makes it easier to test out models trained on different cpu/gpu configurations than local
# would like to see why regular torch.load isn't working perfectly right now
def write_parameters_into_model(model, filename):
    for param, src in zip(model.parameters(), torch.load(filename, map_location=next(model.parameters()).device)):
        param.data = torch.tensor(src)


def save_parameters_from_model(model, filename):
    torch.save(list(x.cpu().detach().numpy().copy() for x in model.parameters()), filename)


@cache
def mask_tensor(n=DATA_FRAME_SIZE):
    frm = int(np.ceil(np.log2(n)) - 1) * 2
    mask = torch.zeros((frm, n, n), dtype=torch.uint8).bool()
    prev = torch.zeros((frm, n, n), dtype=torch.uint8).bool()

    for i in range(frm):
        if i >= 1:
            prev[i] = torch.logical_or(mask[i - 1], prev[i - 1])

        reverse_index = frm // 2 - i // 2
        step = int(2 ** reverse_index)
        r, c = step // 2 - 1, step // 2 - 1
        if i % 2 == 1:
            step //= 2
        mask[i, r::step, c::step] = ~prev[i, r::step, c::step]

    return prev.to(get_device()), mask.to(get_device())


def ternary_decomp(index):
    ternary = []
    while index > 0:
        ternary.append(index % 3)
        index //= 3

    ret = []
    for i in range(len(ternary)):
        for j in range(ternary[i]):
            ret.append(3 ** i)

    np.random.shuffle(ret)
    return ret


def rmse_lsa_unary_frame(y_true, y_pred, ensemble_size):
    rmse = 0
    for k in range(y_true.shape[0] // ensemble_size):
        z_true = y_true[ensemble_size * k: ensemble_size * (k + 1)]
        z_pred = y_pred[ensemble_size * k: ensemble_size * (k + 1)]

        matrix = torch.zeros((z_true.shape[0], z_pred.shape[0]))
        for i in range(z_true.shape[0]):
            for j in range(z_pred.shape[0]):
                matrix[i][j] = torch.sqrt(torch.mean(torch.square(z_true[i] -
                                                                  z_pred[j])))
        row_ind, col_ind = linear_sum_assignment(matrix.detach().numpy())
        rmse += torch.sqrt(torch.mean(torch.square(z_true[row_ind] - z_pred[col_ind]))) * ensemble_size / y_true.shape[0]
    return rmse


if __name__ == '__main__':
    print(ternary_decomp(26))
