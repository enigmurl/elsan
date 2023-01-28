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
        y = torch.load(self.direc + str(ID) + ".pt")[self.mid:(self.mid+self.output_length)]
        if self.stack_x:
            x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid].\
                reshape(-1, y.shape[-2], y.shape[-1])
        else:
            x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid]
        return x.float(), y.float()


def train_epoch(train_loader, model, orthonet, optimizer, loss_function, e_loss_fun, pruning_size, coef=0, regularizer=None):
    train_mse = []
    train_emse = []
    p = []

    for xx, yy in train_loader:
        loss = 0
        e_loss = 0
        ims = []
        xx = xx.to(device).detach()
        yy = yy.to(device).detach()
        error = torch.zeros((xx.shape[0], pruning_size, *xx.shape[2:])).float().to(device)

        for y in yy.transpose(0, 1):
            im, error = model(xx, error)
            xx = torch.cat([xx[:, 2:], im], 1)
      
            if coef != 0:
                loss += loss_function(im, y) + coef*regularizer(im, y)
            else:
                loss += loss_function(im, y)

            e_loss += e_loss_fun(orthonet, im, error, y)
            p.append(p_full_in(im, error, y))

            error = torch.nn.functional.pad(error, (1, 1, 1, 1))

            # ims.append(im.cpu().data.numpy())

        full_loss = loss + e_loss

        train_emse.append(e_loss.item() / yy.shape[1])
        train_mse.append(loss.item() / yy.shape[1])

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

    train_mse, e_loss = round(np.sqrt(np.mean(train_mse)), 5), round(np.sqrt(np.mean(train_emse)), 5)
    p = np.mean(p, axis=0)

    return train_mse, e_loss, p


def eval_epoch(valid_loader, model, orthonet, loss_function, e_loss_fun, pruning_size):
    valid_mse = []
    valid_emse = []
    p = []

    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
            loss = 0
            e_loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            error = torch.zeros((xx.shape[0], pruning_size, *xx.shape[2:])).float().to(device)
            ims = []

            print("Batch")
            for y in yy.transpose(0, 1):
                im, error = model(xx, error)
                xx = torch.cat([xx[:, 2:], im], 1)
                loss += loss_function(im, y)
                ims.append(im.cpu().data.numpy())

                e_loss += e_loss_fun(orthonet, im, error, y)

                # p.append(p_full_in(im, error, y))
                p.append(0)

            # ims = np.array(ims).transpose((1, 0, 2, 3, 4))
            # preds.append(ims)
            # trues.append(yy.cpu().data.numpy())

            valid_mse.append(loss.item()/yy.shape[1])
            valid_emse.append(e_loss.item()/yy.shape[1])

        # preds = np.concatenate(preds, axis=0)
        # trues = np.concatenate(trues, axis=0)

        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
        valid_emse = round(np.sqrt(np.mean(valid_emse)), 5)
        p = np.mean(p, axis=0)

    return valid_mse, valid_emse, p, preds, trues


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

            for y in yy.transpose(0,1):
                im, error = model(xx, error)
                xx = torch.cat([xx[:, 2:], im], 1)
                mse = loss_function(im, y)
                loss += mse
                loss_curve.append(mse.item())

                e_loss += e_loss_fun(im, error, y)

                ims.append(im.cpu().data.numpy())

            ims = np.array(ims).transpose((1, 0, 2, 3, 4))
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())            
            valid_mse.append(loss.item()/yy.shape[1])
            valid_emse.append(e_loss.item()/yy.shape[1])

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        loss_curve = np.array(loss_curve).reshape(-1, 60)
        loss_curve = np.sqrt(np.mean(loss_curve, axis=0))
    return preds, trues, loss_curve
