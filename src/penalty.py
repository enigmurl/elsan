import torch
import numpy as np
from statistics import NormalDist
from util import get_device, mask_tensor

device = get_device()


def p_full_in(actual_mu, actual_sigma, expected):
    return
    # pos_dif = torch.abs(actual_mu - expected)
    # real_dif = actual_sigma - pos_dif
    #
    # count_in = torch.sum(real_dif > 0, dim=list(range(1, len(real_dif.shape)))).float()
    # full_in = count_in == torch.numel(real_dif[0])
    # p_in = torch.sum(full_in).float() / real_dif.shape[0]
    #
    # print("full", p_in, "channel", torch.sum(real_dif > 0) / torch.numel(real_dif),
    #       "min predicted off", torch.max(actual_sigma),
    #       "min actual off", torch.max(pos_dif),
    #       "average node", torch.mean(torch.abs(actual_mu)),
    #       "batch", len(actual_mu))
    #
    # return p_in.cpu().data.numpy()


def c_sample(max_p=NormalDist().cdf(4)):
    return -1
    in_p = torch.rand(1) * (2 * max_p - 1) + (1 - max_p)
    delta = abs(in_p - 0.5) * 2
    return NormalDist().inv_cdf(delta)


class BigErrorLoss(torch.nn.Module):
    def __init__(self, drift=0.05):
        super(BigErrorLoss, self).__init__()
        self.drift = drift

    def forward(self, ortho_nets, actual_pruning, expected, con_list, hammer):
        prev, mask = mask_tensor(expected.shape[-1])
        loss = 0

        for ortho, c in zip(ortho_nets, con_list):
            batch_masks = (torch.rand(len(expected)) * len(prev)).long()

            real_prev, real_mask = torch.unsqueeze(prev[batch_masks], dim=1), torch.unsqueeze(mask[batch_masks], dim=1)
            # real_prev = torch.rand((len(expected), 1, *expected.shape[2:])) > 0.5
            # real_mask = torch.rand((len(expected), 1, *expected.shape[2:])) > 0.5
            real_prev = torch.tile(real_prev & ~real_mask, (2, 1, 1))
            mask2 = torch.tile(real_mask, (2, 1, 1))
            mask4 = torch.tile(real_mask, (4, 1, 1))

            # compute query
            query = torch.full((len(expected), 4, *expected.shape[2:]), -5.0, device=device)
            query[:, 2:] = 5
            query[:, :2][real_prev] = expected[real_prev]
            query[:, 2:][real_prev] = expected[real_prev]
            query[mask4] = -query[mask4]

            predicted = ortho(actual_pruning, query)

            pred_min = torch.flatten(predicted[:, :2])
            pred_max = torch.flatten(predicted[:, 2:])
            compare = expected[mask2]

            greater = compare >= pred_min
            less = compare <= pred_max

            # standard hammer and lorris
            full_in = torch.logical_and(greater, less)
            count_in = torch.sum(full_in)
            p_in = count_in / torch.numel(full_in)

            widths = torch.mean(torch.square(pred_max - pred_min))

            converted_c = NormalDist().cdf(c)

            loss += self.drift * torch.mean(torch.square((pred_max + pred_min) / 2 - compare))

            if p_in < converted_c:
                loss += hammer.hammer_loss(pred_min, compare, pred_max)
                print(f"Not contained {float(p_in.cpu().data):.4f} "
                      f"compare {converted_c:.4f} "
                      f"greater {float((torch.sum(greater) / torch.numel(greater)).cpu().data):.4f} "
                      f"less {float((torch.sum(less) / torch.numel(less)).cpu().data):.4f} "
                      f"width {float(torch.sqrt(widths).cpu().data):.4f} "
                      f"drift {loss / self.drift:.4f}")
            else:
                loss += hammer.lorris_loss(pred_min, compare, pred_max)
                print(f"Yes contained {float(p_in.cpu().data):.4f} "
                      f"compare {converted_c:.4f} "
                      f"greater {float((torch.sum(greater) / torch.numel(greater)).cpu().data):.4f} "
                      f"less {float((torch.sum(less) / torch.numel(less)).cpu().data):.4f} "
                      f"width {float(torch.sqrt(widths).cpu().data):.4f}")

        return p_in, torch.sqrt(widths), loss


'''
class ErrorLoss(torch.nn.Module):
    def __init__(self):
        super(ErrorLoss, self).__init__()

    def forward(self, ortho_nets, actual_pruning, expected, con_list, hammer):
        loss = 0
        # so new system is to randomly apply the ins and outs
        # and then the relative pivot position for each batch is fixed...
        # [batch, 4, rows, cols]
        batch = expected.shape[0]
        r = expected.shape[-2]
        c = expected.shape[-1]
        for con, ortho in zip(con_list, ortho_nets):
            rows = ortho.grid_rows
            query = torch.tile(expected, dims=(1, 2, 1, 1))

            # shut off random channels
            channel_mask = torch.rand((batch, 1, r, c), device=device) > 0.5
            mask = torch.tile(channel_mask, dims=(1, 2, 1, 1))

            query[:, :2][mask] = -1
            query[:, 2:][mask] = +1

            # input inverted interval
            # TODO we need to see how to do per instance of batch
            pivot_dr = (torch.rand(batch) * rows).long()
            pivot_dc = (torch.rand(batch) * rows).long()

            for k in range(batch):
                query[k, :2, pivot_dr[k]::rows, pivot_dc[k]::rows] = torch.ones(1).to(device)
                query[k, 2:, pivot_dr[k]::rows, pivot_dc[k]::rows] = -torch.ones(1).to(device)

            query = query.detach()

            result = ortho(actual_pruning, query, con)
            # real_dif = (expected - actual_mu).detach()  # do not adjust mu
            real_dif = expected

            # for each node, set the central value to the true difference of the pivot
            reverse_pad = (rows - 1) // 2 * 2
            dif_index = torch.zeros((batch, 2, r - reverse_pad, c - reverse_pad), device=device)
            # TODO, remove for loops (possibly via reshaping + padding)
            for i in range(rows):
                for j in range(rows):
                    for k in range(batch):
                        end_r = (r - reverse_pad - i + rows - 1) // rows * rows + pivot_dr[k]
                        end_c = (c - reverse_pad - j + rows - 1) // rows * rows + pivot_dc[k]
                        dif_index[k, :, i::rows, j::rows] = real_dif[k, :, pivot_dr[k]:end_r:rows, pivot_dc[k]:end_c:rows]

            greater = dif_index >= result[:, :2]
            less = dif_index <= result[:, 2:]

            full_in = torch.logical_and(greater, less)
            count_in = torch.sum(full_in)
            p_in = count_in / torch.numel(full_in)

            widths = torch.mean(torch.square(result[:, 2:] - result[:, :2]))

            converted_c = NormalDist().cdf(con)

            if p_in < converted_c:
                # Hammer on nodes outside
                loss += hammer.hammer_loss(result[:, :2], dif_index, result[:, 2:])
                print(f"Not contained {float(p_in.cpu().data):.4f} "
                      f"compare {converted_c:.4f} "
                      f"greater {float((torch.sum(greater) / torch.numel(greater)).cpu().data):.4f} "
                      f"less {float((torch.sum(less) / torch.numel(less)).cpu().data):.4f} "
                      f"width {float(torch.sqrt(widths).cpu().data):.4f}")
            else:
                loss += hammer.lorris_loss(result[:, :2], dif_index, result[:, 2:])
                print(f"Yes contained {float(p_in.cpu().data):.4f} "
                      f"compare {converted_c:.4f} "
                      f"greater {float((torch.sum(greater) / torch.numel(greater)).cpu().data):.4f} "
                      f"less {float((torch.sum(less) / torch.numel(less)).cpu().data):.4f} "
                      f"width {float(torch.sqrt(widths).cpu().data):.4f}")

        loss /= len(ortho_nets)

        return p_in, torch.sqrt(widths), loss
'''

class MagnitudeLoss(torch.nn.Module):
    def __init__(self, loss):
        super(MagnitudeLoss, self).__init__()
        self.loss = loss

    def forward(self, w):
        return self.loss(w, w.detach() * 0)


class SmoothnessLoss(torch.nn.Module):
    """From Back to Basics:
    Unsupervised Learning of Optical Flow
    via Brightness Constancy and Motion Smoothness"""

    def __init__(self, loss, delta=1):
        super(SmoothnessLoss, self).__init__()
        self.loss = loss
        self.delta = delta

    def forward(self, w):
        ldudx = self.loss((w[:, 0, 1:, :] - w[:, 0, :-1, :]) /
                          self.delta, w[:, 0, 1:, :].detach() * 0)
        ldudy = self.loss((w[:, 0, :, 1:] - w[:, 0, :, :-1]) /
                          self.delta, w[:, 0, :, 1:].detach() * 0)
        ldvdx = self.loss((w[:, 1, 1:, :] - w[:, 1, :-1, :]) /
                          self.delta, w[:, 1, 1:, :].detach() * 0)
        ldvdy = self.loss((w[:, 1, :, 1:] - w[:, 1, :, :-1]) /
                          self.delta, w[:, 1, :, 1:].detach() * 0)
        return ldudx + ldudy + ldvdx + ldvdy


class WeightedSpatialMSELoss(torch.nn.Module):
    def __init__(self, weights):
        super(WeightedSpatialMSELoss, self).__init__()
        self.loss = torch.nn.MSELoss(reduce=False, size_average=False)
        self.weights = weights

    def forward(self, preds, trues):
        print(self.loss(preds, trues).shape, self.weights.shape)
        return self.loss(preds, trues).mean(4).mean(3).mean(2).mean(0) * self.weights


class DivergenceLoss(torch.nn.Module):
    def __init__(self, loss, delta=1):
        super(DivergenceLoss, self).__init__()
        self.delta = delta
        self.loss = loss

    def forward(self, preds):
        # preds: bs*2*H*W

        u = preds[:, :1]
        v = preds[:, -1:]
        u_x = field_grad(u, 0)
        v_y = field_grad(v, 1)
        div = v_y + u_x
        return self.loss(div, div.detach() * 0)


class DivergenceLoss2(torch.nn.Module):
    def __init__(self, loss, delta=1):
        super(DivergenceLoss2, self).__init__()
        self.delta = delta
        self.loss = loss

    def forward(self, preds, trues):
        # preds: bs*steps*2*H*W
        u = preds[:, :1]
        v = preds[:, -1:]
        u_x = field_grad(u, 0)
        v_y = field_grad(v, 1)
        div_pred = v_y + u_x

        u = trues[:, :1]
        v = trues[:, -1:]
        u_x = field_grad(u, 0)
        v_y = field_grad(v, 1)
        div_true = v_y + u_x
        return self.loss(div_pred, div_true)


def vorticity(u, v):
    return field_grad(v, 0) - field_grad(u, 1)


class VorticityLoss(torch.nn.Module):
    def __init__(self, loss):
        super(VorticityLoss, self).__init__()
        self.loss = loss

    def forward(self, preds, trues):
        u, v = trues[:, :1], trues[:, -1:]
        u_pred, v_pred = preds[:, :1], preds[:, -1:]
        return self.loss(vorticity(u, v), vorticity(u_pred, v_pred))


def field_grad(f, dim):
    # dim = 1: derivative to x direction, dim = 2: derivative to y direction
    dx = 1
    dim += 1
    N = len(f.shape)
    out = torch.zeros(f.shape).to(device)
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    # 2nd order interior
    slice1[-dim] = slice(1, -1)
    slice2[-dim] = slice(None, -2)
    slice3[-dim] = slice(1, -1)
    slice4[-dim] = slice(2, None)
    out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2 * dx)

    # 2nd order edges
    slice1[-dim] = 0
    slice2[-dim] = 0
    slice3[-dim] = 1
    slice4[-dim] = 2
    a = -1.5 / dx
    b = 2. / dx
    c = -0.5 / dx
    out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    slice1[-dim] = -1
    slice2[-dim] = -3
    slice3[-dim] = -2
    slice4[-dim] = -1
    a = 0.5 / dx
    b = -2. / dx
    c = 1.5 / dx

    out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    return out


def TKE(preds):
    preds = preds.reshape(preds.shape[0], -1, 2, 64, 64)
    mean_flow = torch.mean(preds, dim=1).unsqueeze(1)
    tur_preds = torch.mean((preds - mean_flow) ** 2, dim=1)
    tke = (tur_preds[:, 0] + tur_preds[:, 1]) / 2
    return tke


class TKELoss(torch.nn.Module):
    def __init__(self, loss):
        super(TKELoss, self).__init__()
        self.loss = loss

    def forward(self, preds, trues):
        return self.loss(TKE(trues), TKE(preds))
