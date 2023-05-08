import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import time
from model import Orthonet, con_list, ran_sample
from penalty import DivergenceLoss, BigErrorLoss
from train import ClusteredDataset, ClippingDataset, train_epoch, train_clipping_epoch, eval_epoch, test_epoch
from util import get_device
from hammer_scheduler import HammerSchedule
import sys
torch.multiprocessing.set_sharing_strategy('file_system')

device = get_device()

train_direc = "../data/ensemble/"
clipping_direc = "../data/clipping/"
test_direc = "../data/ensemble/"

# best_params: kernel_size 3, learning_rate 0.001, dropout_rate 0, batch_size 120, input_length 25, output_length 4
time_range = 1
output_length = 64
input_length = 16
learning_rate = 1e-3
dropout_rate = 0
kernel_size = 3
batch_size = 16
clipping_batch = 128
pruning_size = 16
coef = 0

clipping_indices = list(range(0, int(2 ** 15)))
train_indices = list(range(0, 8 * 256))
valid_indices = list(range(0, 8 * 256))
test_indices = list(range(7700, 9800))

if __name__ == '__main__':
    model = Orthonet(input_channels=input_length * 2,
                     pruning_size=pruning_size,
                     kernel_size=kernel_size,
                     dropout_rate=dropout_rate,
                     time_range=time_range
                     ).to(device)
    if len(sys.argv) > 1:
        for param, src in zip(model.parameters(), torch.load('model_state.pt', map_location=torch.device('cpu'))):
            param.data = torch.tensor(src, device=device)

    model = nn.DataParallel(model)
    base = model._modules['module'].base
    trans = model._modules['module'].transition
    query = model._modules['module'].query
    clipping = model._modules['module'].clipping

    # model.eval()
    # frames = torch.cat([load_rand() for _ in range(64)], dim=0)
    # xx = frames[:, :12].to(device)
    train_set = ClusteredDataset(train_indices, train_direc, input_length, 6, output_length)
    clipping_set = ClippingDataset(clipping_indices, clipping_direc)
    # workers causing bugs on m1x, likely due to lack of memory
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    clipping_loader = data.DataLoader(clipping_set, batch_size=clipping_batch, shuffle=True, num_workers=0)

    loss_fun = torch.nn.MSELoss()
    error_fun = BigErrorLoss()
    regularizer = DivergenceLoss(torch.nn.MSELoss())
    hammer = HammerSchedule(lorris=1, lorris_buffer=1e-3, lorris_decay=2e-4,
                            hammer=1, hammer_buffer=1e-3, hammer_decay=2e-4)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    train_emse = []
    valid_emse = []

    test_mse = []
    min_mse = 100

    if len(sys.argv) > 1 and sys.argv[1] == 'clipping':
        print("Training clipping")
        for i in range(1000):
            print("Epoch", i)
            start = time.time()

            torch.cuda.empty_cache()

            model.train()

            emse = train_clipping_epoch(clipping_loader, clipping, optimizer)

            train_emse.append(emse)

            torch.save(model, "model.pt")
            torch.save(list(x.cpu().detach().numpy().copy() for x in model.parameters()), 'model_state.pt')
            end = time.time()

            print("train emse", train_emse[-1],
                  "minutes", round((end - start) / 60, 5))

    else:
        print("Training main orthonet")

        for i in range(1000):
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
            emse = train_epoch(train_loader, base, trans, query, optimizer, hammer, con_list, error_fun)

            train_emse.append(emse)
            #
            # model.eval()
            # emse = eval_epoch(valid_loader, base, trans, query, hammer, con_list, error_fun)
            # valid_emse.append(emse)
            # valid_emse = [min_mse * 0.5]
            # test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, test_direc, True)

            torch.save(model, "model.pt")
            torch.save(list(x.cpu().detach().numpy().copy() for x in model.parameters()), 'model_state.pt')
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

            print("train emse", train_emse[-1],
                  "minutes", round((end - start) / 60, 5))