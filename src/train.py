import torch
import numpy as np
from torch.utils import data

from hyperparameters import WARMUP_SLOPE, O_MAX_ENSEMBLE_COUNT
from util import get_device, ternary_decomp, rmse_lsa_unary_frame

device = get_device()


class EnsembleDataset(data.Dataset):
    def __init__(self, index, run_size, direc, input_length):
        super(EnsembleDataset, self).__init__()
        self.map = index
        self.direc = direc
        self.run_size = run_size
        self.input_length = input_length

    def __len__(self):
        return self.run_size

    def __getitem__(self, index):
        # split epochs into multiple epochs
        # index = np.random.randint(0, len(self.map))
        # n = np.random.randint(0, 64 - O_MAX_ENSEMBLE_COUNT)
        seed = torch.load(self.direc + 'seed_' + str(index) + '.pt').to(device)
        frames = torch.load(self.direc + 'frames_' + str(index) + '.pt').to(device)  # [n:n + O_MAX_ENSEMBLE_COUNT, ::8]
        return seed.float(), frames.float()


class ClusteredDataset(data.Dataset):
    def __init__(self, index, direc, input_length):
        super(ClusteredDataset, self).__init__()
        self.map = index
        self.direc = direc
        self.input_length = input_length

    def __len__(self):
        return len(self.map)

    def __getitem__(self, index):
        seed = torch.load(self.direc + 'seed_' + str(index // 8) + '.pt').to(device)
        lower = torch.load(self.direc + 'lowers_' + str(index) + '.pt').to(device)
        upper = torch.load(self.direc + 'uppers_' + str(index) + '.pt').to(device)
        frames = torch.load(self.direc + 'answer_' + str(index) + '.pt').to(device)
        return seed.float(), lower.float(), upper.float(), frames.float()


class ClippingDataset(data.Dataset):
    def __init__(self, index, direc):
        super(ClippingDataset, self).__init__()
        self.map = index
        self.direc = direc

    def __len__(self):
        return len(self.map)

    def __getitem__(self, index):
        xx = torch.load(self.direc + 'x_' + str(index) + '.pt').to(device).float()
        pp = torch.load(self.direc + 'p_' + str(index) + '.pt').to(device).float()
        yy = torch.load(self.direc + 'y_' + str(index) + '.pt').to(device).float()
        return xx, pp, yy


def train_clipping_epoch(train_loader, clipping, optimizer):
    loss_a = []
    for b, (yy, pp, xx) in enumerate(train_loader):
        optimizer.zero_grad()
        zz = torch.cat((yy, pp), dim=1)
        loss = torch.sqrt(torch.mean(torch.square(clipping(zz) - yy)))
        loss.backward()
        optimizer.step()

        loss_a.append(float(loss))

    return np.mean(loss_a)


def train_base_orthonet_epoch(train_loader, base, trans, query, optimizer, e_loss_fun):
    train_emse = []

    for b, (seed, lower, upper, frames) in enumerate(train_loader):
        e_loss = 0

        index = frames.shape[1]  # how many frames to go into future, some variations have a warmup period

        seed = seed.to(device).detach()
        lower = lower.to(device)[:, :index].detach()
        upper = upper.to(device)[:, :index].detach()
        frames = frames.to(device)[:, :index].detach()

        error = base(seed)

        for f, y in enumerate(frames.transpose(0, 1)):
            dloss = e_loss_fun(query, lower[:, f], upper[:, f], error, y)
            e_loss += dloss

            if f < index - 1:
                error = trans(error)

        full_loss = e_loss

        train_emse.append(e_loss.item() / frames.shape[1])

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

    return round(np.mean(train_emse), 5)


def train_orthonet_epoch(train_loader, e_num, model, optimizer):
    train_emse = []

    for b, (seed, frames) in enumerate(train_loader):
        e_loss = 0

        seed = seed.detach()
        frames = torch.flatten(frames.detach(), 0, 1)

        max_index = min((e_num + 1) * WARMUP_SLOPE, frames.shape[1])
        index = torch.randint(0, max_index)

        decomp = ternary_decomp(index)

        running = -1
        seed = torch.repeat_interleave(seed, frames.shape[0] // seed.shape[0], dim=0)
        error = model.base(seed)

        for delta in decomp:
            running += delta
            error = model.trans[delta](error)
            res = torch.normal(0, 1, size=frames.shape).to(device)
            full_vector = torch.cat((res, error), dim=1)
            overall = model.clipping(full_vector)
            post_clip_loss = rmse_lsa_unary_frame(overall, frames[:, running], O_MAX_ENSEMBLE_COUNT)
            e_loss += post_clip_loss

        e_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return round(np.mean(train_emse), 5)


# TODO
def eval_epoch(valid_loader, base, trans, query, hammer, c_fun, e_loss_fun):
    valid_emse = []

    with torch.no_grad():
        for b, (xx, yy) in enumerate(valid_loader):
            e_loss = 0
            xx = xx.to(device).detach()
            yy = yy.to(device).detach()
            error = base(xx)

            for f, y in enumerate(yy.transpose(0, 1)):
                dloss = e_loss_fun(query, error, y, c_fun, hammer, f)
                e_loss += dloss

                if f != yy.shape[1] - 1:
                    error = trans(error)

            valid_emse.append(e_loss.item() / yy.shape[1])

    e_loss = round(np.sqrt(np.mean(valid_emse)), 5)
    return e_loss


def test_epoch(test_loader, model, loss_function, e_loss_fun):
    valid_mse = []
    valid_emse = []
    preds = []
    trues = []
    with torch.no_grad():
        loss_curve = []
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            error = torch.zeros((xx.shape[0], 2, *xx.shape[2:])).float().to(device)

            loss = 0
            e_loss = 0
            ims = []

            for f, y in enumerate(yy.transpose(0, 1)):
                im, error = model(xx, error)
                xx = torch.cat([xx[:, 2:], im], 1)
                mse = loss_function(im, y)
                loss += mse
                loss_curve.append(mse.item())

                e_loss += e_loss_fun(im, error, y, f)

                ims.append(im.cpu().data.numpy())

            ims = np.array(ims).transpose((1, 0, 2, 3, 4))
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())            
            valid_mse.append(loss.item()/yy.shape[1])
            valid_emse.append(e_loss.item()/yy.shape[1])

        preds = np.catenate(preds, axis=0)
        trues = np.catenate(trues, axis=0)

        loss_curve = np.array(loss_curve).reshape(-1, 60)
        loss_curve = np.sqrt(np.mean(loss_curve, axis=0))
    return preds, trues, loss_curve
