from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import matplotlib.pyplot as plt

from hyperparameters import *
from penalty import TKE, full_vorticity, full_divergence
from util import orthonet_model, get_device
device = get_device()
# device = torch.device('cpu')


def rmse_most_likely(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.square(y_true - y_pred)
                                 .view(-1, y_true.shape[1] // 2, 2, y_true.shape[2], y_true.shape[3]),
                                 dim=(0, 2, 3, 4))).to(device)


def rmse_linear_assignment(y_true, y_pred):
    loss = []
    for f in range(y_true.shape[1] // 2):
        matrix = torch.zeros((y_true.shape[0], y_pred.shape[0]))
        for i in range(y_true.shape[0]):
            for j in range(y_pred.shape[0]):
                matrix[i][j] = torch.sqrt(torch.mean(torch.square(y_true[i, f*2:(f+1)*2] -
                                                                  y_pred[j, f*2:(f+1)*2])))
        row_ind, col_ind = linear_sum_assignment(matrix)
        loss.append(torch.sqrt(torch.mean(torch.square(y_true[row_ind, f*2:(f+1)*2] - y_pred[col_ind, f*2:(f+1)*2]))))
    return torch.tensor(loss).to(device)


def vort_most_likely(y_true, y_pred):
    return torch.stack([full_vorticity(y_true[:, f*2:f*2+1],
                                       y_pred[:, f*2:f*2+1]) for f in range(y_true.shape[1] // 2)]
                       ).to(device)


def vort_linear_assignment(y_true, y_pred):
    loss = []
    for f in range(y_true.shape[1] // 2):
        matrix = torch.zeros((y_true.shape[0], y_pred.shape[0]))
        for i in range(y_true.shape[0]):
            for j in range(y_pred.shape[0]):
                matrix[i][j] = full_vorticity(torch.unsqueeze(y_true[i, f * 2:(f + 1) * 2], dim=0),
                                              torch.unsqueeze(y_pred[j, f * 2:(f + 1) * 2], dim=0))
        row_ind, col_ind = linear_sum_assignment(matrix)
        loss.append(full_vorticity(y_true[row_ind, f * 2:(f + 1) * 2], y_pred[col_ind, f * 2:(f + 1) * 2]))
    return torch.tensor(loss).to(device)


def div_most_likely(y_true, y_pred):
    return torch.stack([full_divergence(y_true[:, f*2:f*2+1],
                                       y_pred[:, f*2:f*2+1]) for f in range(y_true.shape[1] // 2)]
                       ).to(device)


def div_linear_assignment(y_true, y_pred):
    loss = []
    for f in range(y_true.shape[1] // 2):
        matrix = torch.zeros((y_true.shape[0], y_pred.shape[0]))
        for i in range(y_true.shape[0]):
            for j in range(y_pred.shape[0]):
                matrix[i][j] = full_divergence(torch.unsqueeze(y_true[i, f * 2:(f + 1) * 2], dim=0),
                                              torch.unsqueeze(y_pred[j, f * 2:(f + 1) * 2], dim=0))
        row_ind, col_ind = linear_sum_assignment(matrix)
        loss.append(full_divergence(y_true[row_ind, f * 2:(f + 1) * 2], y_pred[col_ind, f * 2:(f + 1) * 2]))
    return torch.tensor(loss).to(device)


def loss_standard_deviation(y_true, y_pred):
    s1 = torch.std(y_true.view(-1, y_true.shape[1] // 2, 2, y_true.shape[2], y_true.shape[3]),
                                 dim=(0, 2, 3, 4))
    s2 = torch.std(y_pred.view(-1, y_true.shape[1] // 2, 2, y_true.shape[2], y_true.shape[3]),
                   dim=(0, 2, 3, 4))

    return torch.abs(s1 - s2).to(device)


model_losses = {}
models = {}
metrics = {
    'RMSE LSA': rmse_linear_assignment,
    'RMSE RAW': rmse_most_likely,
    'VORT LSA': vort_linear_assignment,
    'VORT RAW': vort_most_likely,
    'DIV RAW': div_most_likely,
    'DIV LSA': div_linear_assignment,
    'LOSS STD': loss_standard_deviation,
}


def init_models():
    global models, model_losses

    # mps error...
    orthonet = orthonet_model().eval()
    orthonet = orthonet.cpu()

    models = {
        'orthonet': orthonet,
        # 'tfnet': None,
    }
    model_losses = {m_name: {m: torch.zeros(DATA_OUT_FRAME, device=device)
                             for m in metrics.keys()}
                    for m_name in models.keys()}


if __name__ == '__main__':
    init_models()

    with torch.no_grad():
        # for each batch
        # sum up all metrics on all the different models
        for bnum in V_INDICES[:1]:
            print("Batch: ", bnum)
            x_true = torch.load(DATA_DIR + V_DIREC + 'seed_' + str(bnum) + '.pt')
            x_true = torch.flatten(x_true, 0, 1)
            y_true = torch.load(DATA_DIR + V_DIREC + 'frames_' + str(bnum) + '.pt')

            x_true = torch.cat([torch.unsqueeze(x_true, 0) for _ in range(V_BATCH_SIZE)], dim=0).float().to(device)
            y_true = y_true.float().to(device)
            for model_name, model in models.items():
                # mps error
                y_pred = model(x_true.cpu(), y_true.shape[1] // 2).to(device)
                for metric_name, metric in metrics.items():
                    delta = metric(y_true, y_pred)
                    model_losses[model_name][metric_name] += delta / len(V_INDICES)

        # plot
        for metric_name, metric in metrics.items():
            plt.figure()
            plt.title(metric_name)
            plt.xlabel('Frame #')
            plt.ylabel(metric_name)
            for model_name, model in models.items():
                plt.plot(model_losses[model_name][metric_name].cpu().detach().numpy(), label=model_name)
            plt.legend()

        plt.show()
