import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import time
from model import CLES, LES, con_list, ran_sample
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
pruning_size = 8
coef = 0

train_indices = list(range(0, 6000))
valid_indices = list(range(6000, 7700))
test_indices = list(range(7700, 9800))


def load_rand():
    index = np.random.random_integers(6000, 7700)
    ret = torch.load(train_direc + str(index) + ".pt")
    return torch.unsqueeze(ret.reshape(-1, ret.shape[-2], ret.shape[-1]), dim=0).to(device)


if __name__ == '__main__':
    model = CLES(input_channels=input_length * 2, output_channels=2, kernel_size=kernel_size,
                 dropout_rate=dropout_rate, time_range=time_range, pruning_size=pruning_size,
                 orthos=len(con_list)).to(device)
    # for param, src in zip(model.parameters(), torch.load('model_state.pt', map_location=torch.device('cpu'))):
    #    param.data = torch.tensor(src, device=device)
    #    print(np.mean(np.abs(src)))
    # model = torch.load('model.pt')
    print("Hash", hash(model))
    orthonet = model.orthonet
    model = nn.DataParallel(model)
    # model.eval()
    frames = torch.cat([load_rand() for _ in range(64)], dim=0)
    xx = frames[:, :60].to(device)
    train_set = Dataset(train_indices, input_length + time_range - 1, 30, output_length, train_direc, True)
    valid_set = Dataset(valid_indices, input_length + time_range - 1, 30, 6, test_direc, True)
    # workers causing bugs on m1x, likely due to lack of memory
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

    loss_fun = torch.nn.MSELoss()
    error_fun = BigErrorLoss()
    regularizer = DivergenceLoss(torch.nn.MSELoss())
    hammer = HammerSchedule(lorris=1e-1, lorris_buffer=1e-3, lorris_decay=2e-4,
                            hammer=1e-1, hammer_buffer=1e-3, hammer_decay=2e-4)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    train_mse = []
    train_emse = []

    valid_mse = []
    valid_emse = []

    test_mse = []
    min_mse = 100

    for i in range(1000):
        print("Epoch", i)

        start = time.time()

        torch.cuda.empty_cache()

        model.train()
        prev_error = torch.zeros((64, 8, frames.shape[-2], frames.shape[-1]), device=device)
        im, prev_error = model(xx, prev_error)
        ran_sample(model, im, prev_error,
                   frames[:, 60:62])
        print("hash", hash(model))
        mse, emse, = train_epoch(train_loader, model, orthonet, optimizer, hammer, con_list,
                                 loss_fun, error_fun,
                                 pruning_size,
                                 coef, regularizer)

        train_mse.append(mse)
        train_emse.append(emse)

        model.eval()
        mse, emse, preds, trues = eval_epoch(valid_loader, model, orthonet, hammer,
                                             con_list, loss_fun, error_fun, pruning_size)
        valid_mse.append(mse)
        valid_emse.append(emse)

        if valid_mse[-1] + valid_emse[-1] < min_mse:
            min_mse = valid_mse[-1] + valid_emse[-1]
            best_model = model
            torch.save(best_model, "model.pt")
            torch.save(list(x.cpu().data.numpy().copy() for x in best_model.parameters()), 'model_state.pt')
            # test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, test_direc, True)
            # test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
            # preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun, error_fun)
            #
            # torch.save({"preds": preds,
            #             "trues": trues,
            #             "loss_curve": loss_curve},
            #            "results.pt")

        end = time.time()

        print("train mse", train_mse[-1], "train emse", train_emse[-1],
              "valid mse", valid_mse[-1], "valid emse", valid_emse[-1],
              "minutes", round((end - start) / 60, 5))

        if len(train_mse) > 75 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5]):
            break
