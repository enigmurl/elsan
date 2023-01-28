import torch
from util import get_device

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


class ErrorLoss(torch.nn.Module):
    def __init__(self, c, coef, lorris=0.25, hammer=1):
        super(ErrorLoss, self).__init__()
        self.c = c
        self.coef = coef
        self.lorris = lorris
        self.hammer = hammer
        self.mse = torch.nn.MSELoss()

    def forward(self, ortho_net, actual_mu, actual_pruning, expected):
        rows = ortho_net.grid_rows
        # batch, x, y, prune
        t_pruning = torch.permute(actual_pruning, (0, 2, 3, 1))

        real_dif = expected - actual_mu
        perm_dif = torch.permute(real_dif, (0, 2, 3, 1))
        '''
        Note: I use bitmasks here to give orthonet the widest possible
        variety of input data
        However, this has the problem that for large kernel sizes, 
        2 ** (n * n - 1) is far too large, making them impractical.
        Instead, one can make the optimization that while querying,
        we generally only go top left to bottom right. Therefore, a lot of
        the bitmask states are useless and can be optimized to a polynomial
        amount of states with respect to the kernel size.
        '''
        batch_size = actual_mu.shape[0]
        # TODO might be a better way (since it's very similar to a convolution)
        # but for now this works, especially for the small kernel size
        mega_batch = torch.zeros((batch_size, t_pruning.shape[1], t_pruning.shape[2], rows, rows, 4),
                                 device=device)
        pivot = torch.zeros((batch_size, t_pruning.shape[1], t_pruning.shape[2], 2),
                            device=device)

        filt = torch.rand((batch_size, t_pruning.shape[1], t_pruning.shape[2]), device=device) > 0.5
        index_choice = torch.rand(mega_batch.shape[0] * mega_batch.shape[1] * mega_batch.shape[2], device=device)
        index_choice = (index_choice * rows * rows).long()

        for i in range(rows):
            for j in range(rows):
                end_x = real_dif.shape[2] + i - (rows - 1)
                end_y = real_dif.shape[3] + j - (rows - 1)
                # hmmm doing [0, 2] isn't broadcasting correctly, which would otherwise require a transpose
                mega_batch[:, :, :, i, j, 0] = real_dif[:, 0, i:end_x, j:end_y]
                mega_batch[:, :, :, i, j, 2] = real_dif[:, 0, i:end_x, j:end_y]
                mega_batch[:, :, :, i, j, 1] = real_dif[:, 1, i:end_x, j:end_y]
                mega_batch[:, :, :, i, j, 3] = real_dif[:, 1, i:end_x, j:end_y]

                mask = index_choice.view(mega_batch.shape[:3]) == rows * i + j
                pivot[mask] = perm_dif[:, i:end_x, j:end_y, :][mask]
        mega_batch[:, :, :, :, :, [0, 1]][filt] = -1  # some special value to indicate that it's not collapsed
        mega_batch[:, :, :, :, :, [2, 3]][filt] = +1

        mega_batch = torch.flatten(mega_batch, 0, 2)
        mega_batch = torch.flatten(mega_batch, 1, 2)
        # final special value to indicate which value to search for
        mega_batch[torch.arange(len(mega_batch)), index_choice.long()] = torch.tensor([1., 1., -1., -1.], device=device)
        mega_batch = torch.flatten(mega_batch, 1, 2)

        pivot = torch.flatten(pivot, 0, 2)

        # [megabatch, 4]
        t_pruning = torch.flatten(t_pruning, 0, 2)
        results = ortho_net(t_pruning, mega_batch)

        greater = (pivot - results[:, :2]) >= 0
        less = (pivot - results[:, 2:]) <= 0

        full_in = torch.logical_and(greater, less)
        count_in = torch.sum( torch.all(full_in, dim=-1))
        p_in = count_in / len(results)

        widths = torch.sum(torch.abs(results[:, 2:] - results[:, :2]))
        lorris = self.lorris * widths  # sum of widths

        loss = lorris  # regularizer
        if p_in < self.c ** (1 / (actual_mu.shape[2] * actual_mu.shape[3])):
            # Hammer on nodes outside
            loss += torch.mean(torch.square(torch.relu(pivot - results[:, 2:]))) * self.hammer
            loss += torch.mean(torch.square(torch.relu(results[:, :2] - pivot))) * self.hammer

        return loss


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
