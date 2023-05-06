from model import Orthonet, ran_sample
from pvalue import get_start, single_frame, OUT_FRAME
from util import get_device
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import sys
device = get_device()

# generate a bunch of frames... and then match framewise 
# not terrible, actually!

EPOCHS = 32
BATCH = 1


def bipartite_maximal_match(true_x, true_y, sim_x, sim_y):
    matrix = np.zeros((len(true_x), len(sim_x)))
    for i in range(len(true_x)):
        for j in range(len(sim_x)):
            matrix[i][j] = np.sqrt(np.mean(np.square(true_x[i] - sim_x[j]))) + \
                           np.sqrt(np.mean(np.square(true_y[i] - sim_y[j])))

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    row_ind, col_ind = linear_sum_assignment(matrix)

    true_x = [true_x[i] for i in row_ind]
    true_y = [true_y[i] for i in row_ind]
    sim_x = [sim_x[i] for i in col_ind]
    sim_y = [sim_y[i] for i in col_ind]

    return true_x, true_y, sim_x, sim_y


def model():
    model = Orthonet(input_channels=32,
                     pruning_size=16,
                     kernel_size=3,
                     dropout_rate=0,
                     time_range=1
                     ).to(device)
    for param, src in zip(model.parameters(), torch.load('model_state.pt', map_location=torch.device('mps'))):
        param.data = torch.tensor(src)
    return model.to(device)
    

if __name__ == '__main__':
    model = model()
    base = model.base
    trans = model.transition
    query = model.query
    clip = model.clipping
    model.train()

    for e in range(int(sys.argv[1]), EPOCHS):
        rot_start, b_start, sx, sy = get_start()
        start = torch.unsqueeze(torch.flatten(torch.stack((sx, sy)).transpose(0, 1), 0, 1), 0)
        start = start.tile((BATCH, 1, 1, 1))

        real_x = [[] for _ in range(OUT_FRAME)]
        real_y = [[] for _ in range(OUT_FRAME)]
        sim_x = []
        sim_y = []

        for _ in range(BATCH):
            sx, sy = single_frame(rot_start, b_start)

            for f, (x, y) in enumerate(zip(sx, sy)):
                real_x[f].append(x.cpu())
                real_y[f].append(y.cpu())

        error = base(start)

        for f in range(OUT_FRAME):
            res = ran_sample(query, error, None)
            x = res[:, 0].cpu()
            y = res[:, 1].cpu()
            error = trans(error)

            sim_x.append(x)
            sim_y.append(y)

        for f, tx, ty, sx, sy in zip(range(len(real_x)), real_x, real_y, sim_x, sim_y):
            tx1, ty1, sx1, sy1 = bipartite_maximal_match(
                torch.stack(tx).numpy(),
                torch.stack(ty).numpy(),
                sx.numpy(),
                sy.numpy()
            )

            x = torch.tensor(np.array([tx1, ty1]))
            y = torch.tensor(np.array([sx1, sy1]))

            x = torch.swapaxes(x, 0, 1)
            y = torch.swapaxes(y, 0, 1)

            for z, (zx, zy) in enumerate(zip(x, y)):
                torch.save(zx.cpu().float().clone(), '../data/clipping/x_' + str(e * OUT_FRAME * BATCH + f * BATCH + z) + '.pt')
                torch.save(zy.cpu().float().clone(), '../data/clipping/y_' + str(e * OUT_FRAME * BATCH + f * BATCH + z) + '.pt')

                print("LOG: Finished frame", e * OUT_FRAME * BATCH + f * BATCH + z)

