import torch
import numpy as np
from torch.utils import data
from util import get_device
from penalty import p_full_in

device = get_device()


class Dataset(data.Dataset):
    def __init__(self, indices, input_length, mid, output_length, direc, stack_x):
        super(Dataset, self).__init__()
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.stack_x = stack_x
        self.direc = direc
        self.list_IDs = indices
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        org = torch.load(self.direc + str(ID) + ".pt")
        y = org[self.mid:(self.mid+self.output_length)]
        if self.stack_x:
            x = org[(self.mid-self.input_length):self.mid].\
                reshape(-1, y.shape[-2], y.shape[-1])
        else:
            x = org[(self.mid-self.input_length):self.mid]
        return x.float(), y.float()


def train_epoch(train_loader, base, trans, query, optimizer, hammer, c_fun, e_loss_fun):
    train_emse = []

    for b, (xx, yy) in enumerate(train_loader):
        e_loss = 0
        xx = xx.to(device).detach()
        yy = yy.to(device).detach()
        error = base(xx)

        for f, y in enumerate(yy.transpose(0, 1)):
            dloss = e_loss_fun(query, error, y, c_fun, hammer, f)
            e_loss += dloss

            if f != yy.shape[1] - 1:
                error = trans(error)

        full_loss = e_loss

        train_emse.append(e_loss.item() / yy.shape[1])

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
