import random

import torch
import numpy as np
from torch.utils import data
from util import get_device

device = get_device()


class ClusteredDataset(data.Dataset):
    def __init__(self, index, direc, input_length, mid, output_length):
        super(ClusteredDataset, self).__init__()
        self.map = index
        self.direc = direc
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length

    def __len__(self):
        return len(self.map)

    def __getitem__(self, index):
        seed = torch.flatten(torch.load(self.direc + 'seed_' + str(index // 8) + '.pt').to(device), 0, 1)
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
        yy = torch.load(self.direc + 'y_' + str(index) + '.pt').to(device).float()
        return xx, yy


def train_clipping_epoch(train_loader, clipping, optimizer):
    loss_a = []
    for b, (yy, xx) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = torch.sqrt(torch.mean(torch.square(clipping(xx) - yy)))
        loss.backward()
        optimizer.step()

        loss_a.append(float(loss))

    return np.mean(loss_a)


def train_epoch(train_loader, base, trans, query, optimizer, hammer, c_fun, e_loss_fun):
    train_emse = []

    for b, (seed, lower, upper, frames) in enumerate(train_loader):
        e_loss = 0
        seed = seed.to(device)
        index = max(1, min(frames.shape[1], hammer.step_num))
        lower = lower.to(device)[:, :index].detach()
        upper = upper.to(device)[:, :index].detach()
        frames = frames.to(device)[:, :index].detach()

        error = base(seed)

        for f, y in enumerate(frames.transpose(0, 1)):
            dloss = e_loss_fun(query, lower[:, f], upper[:, f], error, y, c_fun, hammer, f)
            e_loss += dloss

            if f != frames.shape[1] - 1:
                error = trans(error)

        full_loss = e_loss

        train_emse.append(e_loss.item() / frames.shape[1])

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()
        hammer.step()

    e_loss = round(np.sqrt(np.mean(train_emse)), 5)
    return e_loss


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
