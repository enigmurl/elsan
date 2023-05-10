from hyperparameters import *
from model import Orthonet, ran_sample
from pvalue import get_start, single_frame, DATA_OUT_FRAME
from util import get_device, write_parameters_into_model
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import sys
device = get_device()


def bipartite_maximal_match(true_x, true_y, sim_x, sim_p, sim_y):
    matrix = np.zeros((len(true_x), len(sim_x)))
    for i in range(len(true_x)):
        for j in range(len(sim_x)):
            matrix[i][j] = np.sqrt(np.mean(np.square(true_x[i] - sim_x[j]))) + \
                           np.sqrt(np.mean(np.square(true_y[i] - sim_y[j])))

    row_ind, col_ind = linear_sum_assignment(matrix)

    true_x = [true_x[i] for i in row_ind]
    true_y = [true_y[i] for i in row_ind]
    sim_x = [sim_x[i] for i in col_ind]
    sim_p = [sim_p[i] for i in col_ind]
    sim_y = [sim_y[i] for i in col_ind]

    return true_x, true_y, sim_x, sim_p, sim_y


def model():
    model = Orthonet(input_channels=O_INPUT_LENGTH * 2,
                     pruning_size=O_PRUNING_SIZE,
                     kernel_size=O_KERNEL_SIZE,
                     dropout_rate=O_DROPOUT_RATE,
                     time_range=O_TIME_RANGE
                     ).to(device)
    write_parameters_into_model(model, 'model_state.pt')
    return model.to(device)
    

if __name__ == '__main__':
    model = model()
    base = model.base
    trans = model.transition
    query = model.query
    clip = model.clipping
    model.train()

    for e in range(int(sys.argv[1]), CLIPPING_EPOCHS):
        rot_start, b_start, sx, sy = get_start()
        start = torch.unsqueeze(torch.flatten(torch.stack((sx, sy)).transpose(0, 1), 0, 1), 0)
        start = start.tile((CLIPPING_BATCH_SIZE, 1, 1, 1))

        real_x = [[] for _ in range(DATA_OUT_FRAME)]
        real_y = [[] for _ in range(DATA_OUT_FRAME)]
        sim_x = []
        sim_p = []
        sim_y = []

        for _ in range(CLIPPING_BATCH_SIZE):
            sx, sy = single_frame(rot_start, b_start)

            for f, (x, y) in enumerate(zip(sx, sy)):
                real_x[f].append(x.cpu())
                real_y[f].append(y.cpu())

        error = base(start)

        for f in range(DATA_OUT_FRAME):
            res = ran_sample(query, error, None)
            x = res[:, 0].cpu()
            y = res[:, 1].cpu()

            sim_x.append(x)
            sim_p.append(error)
            sim_y.append(y)

            error = trans(error)

        for f, tx, ty, sx, sp, sy in zip(range(len(real_x)), real_x, real_y, sim_x, sim_p, sim_y):
            tx1, ty1, sx1, sp, sy1 = bipartite_maximal_match(
                torch.stack(tx).numpy(),
                torch.stack(ty).numpy(),
                sx.numpy(),
                sp.cpu().detach().numpy(),
                sy.numpy()
            )

            x = torch.tensor(np.array([tx1, ty1]))
            p = torch.tensor(sp)
            y = torch.tensor(np.array([sx1, sy1]))

            x = torch.swapaxes(x, 0, 1)
            y = torch.swapaxes(y, 0, 1)

            for z, (zx, zp, zy) in enumerate(zip(x, p, y)):
                n = str(e * DATA_OUT_FRAME * CLIPPING_BATCH_SIZE + f * CLIPPING_BATCH_SIZE + z)
                torch.save(zx.cpu().float().clone(), '../data/clipping/x_' + n + '.pt')
                torch.save(zp.cpu().float().clone(), '../data/clipping/p_' + n + '.pt')
                torch.save(zy.cpu().float().clone(), '../data/clipping/y_' + n + '.pt')

                print("LOG", "finished frame", e * DATA_OUT_FRAME * CLIPPING_BATCH_SIZE + f * CLIPPING_BATCH_SIZE + z)
