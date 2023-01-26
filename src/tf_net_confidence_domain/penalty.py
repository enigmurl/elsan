import torch
from util import get_device

device = get_device()


def p_full_in(actual_mu, actual_sigma, expected):
    pos_dif = torch.abs(actual_mu - expected)
    real_dif = actual_sigma - pos_dif

    count_in = torch.sum(real_dif > 0, dim=list(range(1, len(real_dif.shape)))).float()
    full_in = count_in == torch.numel(real_dif[0])
    p_in = torch.sum(full_in).float() / real_dif.shape[0]

    print("full", p_in, "channel", torch.sum(real_dif > 0) / torch.numel(real_dif),
          "min predicted off", torch.max(actual_sigma),
          "min actual off", torch.max(pos_dif),
          "average node", torch.mean(torch.abs(actual_mu)),
          "batch", len(actual_mu))

    return p_in.cpu().data.numpy()


class ErrorLoss(torch.nn.Module):
    def __init__(self, c, coef, buffer=0.05, regulizer=0.25):
        super(ErrorLoss, self).__init__()
        self.c = c
        self.coef = coef
        self.buffer = torch.tensor([buffer]).to(device)
        self.regulizer = torch.tensor([regulizer]).to(device)
        self.mse = torch.nn.MSELoss()

    def forward(self, actual_mu, actual_sigma, expected):
        # full penalty in bitmap space
        # if percentage of things fully contained is less than c, penalize the ones outside (outsid channels only)
        # percentage of things not fully contained is greater than c, penalize the ones inside (all channels)
        # so that gives us the underlying boundary, we want to clip that even further
        # CLES also outputs a pruning vector
        # we have a secondary model that given a bunch of ranges and a pruning vector, clips that down to the minimum
        # amount of information necessary (so its output is also a bunch of ranges)
        # we train this by taking in the real errors and then saying that you could've deduced that from limited
        # information (so add some noise to the inputs)
        # i suppose we need to be careful that it also contains the stuff exactly p% of the time though...
        # not perfect, but I think it's on the right track.
        # need to take C into account somewhere on pruning level... i suppose
        # train this separately on vowel data since it's an independent task...

        # OK so train the secondary separately on random data
        # from here, we combine this. Then the loss is similar to what we have now, where we see whats in and out, and
        # adjust everything to move it in if needed (p is less), and out if not. We just see the intervals given all
        # other points, and do normal stuff). So then, each output node is the estimation based on all other points
        pos_dif = torch.abs(actual_mu - expected).detach()
        real_dif = actual_sigma - pos_dif

        # channel wise
        # p_in = torch.sum(real_dif > 0) / torch.numel(real_dif)
        #
        # if p_in > self.c:
        #     loss = torch.mean(torch.square(torch.relu(real_dif)))
        # else:
        #     loss = torch.mean(torch.square(torch.relu(-real_dif)))

        # full
        full_in = torch.sum(real_dif > 0, dim=list(range(1, len(real_dif.shape)))) == torch.numel(real_dif[0])
        p_in = torch.sum(full_in) / real_dif.shape[0]

        loss = torch.tensor([0.]).to(device)
        # if p_in > self.c:
        loss += torch.mean(torch.square(torch.relu(real_dif))) * self.regulizer  # reduce on all channels of nodes in
        if p_in < self.c:
            loss += torch.mean(torch.square(torch.relu(-real_dif) + self.buffer))  # increase on all channels outside

        return loss * self.coef


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
