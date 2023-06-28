import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

from hyperparameters import *
from functools import lru_cache
from util import get_device, mask_tensor, ternary_decomp

device = get_device()


def sample(channels: torch.tensor, t):
    ret = torch.zeros((channels.shape[0], 2, *channels.shape[2:]), device=t.device)

    prev_x, prev_y, prev_p = 0, 0, 0
    for i in range(len(CON_LIST) + 1):
        next_x = 0 if i == len(CON_LIST) else channels[:, 2 * i]
        next_y = 0 if i == len(CON_LIST) else channels[:, 2 * i + 1]
        next_p = 1 if i == len(CON_LIST) else NormalDist().cdf(CON_LIST[i])

        mask = (t >= prev_p) & (t <= next_p)
        t_prime = (t - prev_p) / (next_p - prev_p)

        ret[:, 0][mask] = (t_prime * (next_x - prev_x) + prev_x)[mask]
        ret[:, 1][mask] = (t_prime * (next_y - prev_y) + prev_y)[mask]

        prev_x = next_x
        prev_y = next_y
        prev_p = next_p

    return ret


def ran_sample(model, pruning_error, expected):
    output = torch.ones((pruning_error.shape[0]), 4, pruning_error.shape[-2], pruning_error.shape[-1],
                        device=pruning_error.device)

    masks_ = mask_tensor()
    masks_ = masks_[0].to(pruning_error.device), masks_[1].to(pruning_error.device)
    prevs, masks = torch.tile(torch.unsqueeze(masks_[0], 0), (len(pruning_error), 1, 1, 1)), \
        torch.tile(torch.unsqueeze(masks_[1], 0), (len(pruning_error), 1, 1, 1))

    rmse = []
    for i in range(masks.shape[1]):
        query = torch.full((pruning_error.shape[0], 4, pruning_error.shape[-2], pruning_error.shape[-1]),
                           DATA_UPPER_UNKNOWN_VALUE, device=pruning_error.device)
        query[:, :2] = DATA_LOWER_UNKNOWN_VALUE
        m = masks[:, i]
        real_prev, real_mask = torch.unsqueeze(prevs[:, i], dim=1), torch.unsqueeze(m, dim=1)
        real_prev = torch.tile(real_prev & ~real_mask, (2, 1, 1))
        mask2 = torch.tile(real_mask, (2, 1, 1))
        query[0, :2][real_prev[0]] = output[0, :2][real_prev[0]]
        query[0, 2:][real_prev[0]] = output[0, :2][real_prev[0]]

        # compute query
        query[0:, :2][mask2] = DATA_LOWER_QUERY_VALUE
        query[0:, 2:][mask2] = DATA_UPPER_QUERY_VALUE

        predicted = model(pruning_error, query)
        start = NormalDist().cdf(CON_LIST[0])
        delta = sample(predicted, start + (1 - 2 * start) * torch.rand((predicted.shape[0], *predicted.shape[2:]),
                                                                       device=pruning_error.device))

        output[:, :2][mask2] = delta[mask2]
        output[:, 2:][mask2] = delta[mask2]

        if expected is not None:
            rmse.append(torch.mean(torch.square(output[0, :2][mask2[0]] - expected[0][mask2[0]])))

    if expected is not None:
        print(f"RMSE: {torch.sqrt(torch.mean(torch.tensor(rmse))):4f}")

    return output[:, :2]


def conv(input_channels, output_channels, kernel_size, stride, dropout_rate, pad=True, norm=True):
    if not norm:
        layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2 if pad else 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )
    else:
        layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2 if pad else 0),
            nn.BatchNorm2d(output_channels, track_running_stats=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )
    return layer


def deconv(input_channels, output_channels):
    layer = nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )
    return layer


class Encoder(nn.Module):
    def __init__(self, input_channels, kernel_size, dropout_rate):
        super(Encoder, self).__init__()
        self.conv1 = conv(input_channels, 64, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2 = conv(64, 128, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv3 = conv(128, 256, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv4 = conv(256, 512, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.002 / n)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        return out_conv1, out_conv2, out_conv3, out_conv4


class ClippingLayer(nn.Module):
    def __init__(self, noise, pruning, samples, dropout_rate=0, kernel=O_KERNEL_SIZE):
        super(ClippingLayer, self).__init__()
        self.encoder = Encoder(noise + pruning, kernel, dropout_rate)

        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)

        self.output_layer = nn.Conv2d(32 + noise + pruning, 2 * samples,
                                      kernel_size=3,
                                      padding=(kernel - 1) // 2)

    def forward(self, xx):
        out_conv1_mean, out_conv2_mean, out_conv3_mean, out_conv4_mean = self.encoder(xx)

        out_deconv3 = self.deconv3(out_conv4_mean)
        out_deconv2 = self.deconv2(out_conv3_mean + out_deconv3)
        out_deconv1 = self.deconv1(out_conv2_mean + out_deconv2)
        out_deconv0 = self.deconv0(out_conv1_mean + out_deconv1)

        cat0 = torch.cat((xx, out_deconv0[:, :, :63, :63]), dim=-3)

        return self.output_layer(cat0)


class BasePruner(nn.Module):
    def __init__(self, input_channels, pruning_size, kernel_size, dropout_rate, time_range):
        super(BasePruner, self).__init__()

        self.spatial_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.temporal_filter = nn.Conv2d(time_range, 1, kernel_size=1, padding=0, bias=False)
        self.input_channels = input_channels
        self.time_range = time_range
        self.pruning_size = pruning_size

        self.encoder1 = Encoder(input_channels, kernel_size, dropout_rate)
        self.encoder2 = Encoder(input_channels, kernel_size, dropout_rate)
        self.encoder3 = Encoder(input_channels, kernel_size, dropout_rate)

        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)

        self.output_layer = nn.Conv2d(32 + input_channels, pruning_size,
                                      kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2)

    def forward(self, xx):
        xx_len = xx.shape[1]
        # u = u_mean + u_tilde + u_prime
        u_tilde = self.spatial_filter(xx.reshape(xx.shape[0] * xx.shape[1], 1, 63, 63)) \
            .reshape(xx.shape[0], xx.shape[1], 63, 63)
        # u_prime
        u_prime = (xx - u_tilde)[:, (xx_len - self.input_channels):]
        # u_mean
        u_tilde2 = u_tilde.reshape(u_tilde.shape[0], u_tilde.shape[1] // 2, 2, 63, 63)
        u_mean = []
        for i in range(xx_len // 2 - self.input_channels // 2, xx_len // 2):
            cur_mean = torch.cat(
                [self.temporal_filter(u_tilde2[:, i - self.time_range + 1:i + 1, 0, :, :]).unsqueeze(2),
                 self.temporal_filter(u_tilde2[:, i - self.time_range + 1:i + 1, 1, :, :]).unsqueeze(2)], dim=2
            )
            u_mean.append(cur_mean)
        u_mean = torch.cat(u_mean, dim=1)
        u_mean = u_mean.reshape(u_mean.shape[0], -1, 63, 63)
        # u_tilde
        u_tilde = u_tilde[:, (self.time_range - 1) * 2:] - u_mean
        out_conv1_mean, out_conv2_mean, out_conv3_mean, out_conv4_mean = self.encoder1(u_mean)
        out_conv1_tilde, out_conv2_tilde, out_conv3_tilde, out_conv4_tilde = self.encoder2(u_tilde)
        out_conv1_prime, out_conv2_prime, out_conv3_prime, out_conv4_prime = self.encoder3(u_prime)

        out_deconv3 = self.deconv3(out_conv4_mean + out_conv4_tilde + out_conv4_prime)
        out_deconv2 = self.deconv2(out_conv3_mean + out_conv3_tilde + out_conv3_prime + out_deconv3)
        out_deconv1 = self.deconv1(out_conv2_mean + out_conv2_tilde + out_conv2_prime + out_deconv2)
        out_deconv0 = self.deconv0(out_conv1_mean + out_conv1_tilde + out_conv1_prime + out_deconv1)
        cat0 = torch.cat((xx[:, (xx_len - self.input_channels):], out_deconv0[:, :, :63, :63]), 1)
        out = self.output_layer(cat0)

        return out


class TransitionPruner(nn.Module):
    def __init__(self, pruning_size, kernel=3, dropout=0):
        super(TransitionPruner, self).__init__()

        self.network = Encoder(pruning_size, kernel_size=kernel, dropout_rate=dropout)

        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)

        self.output_layer = nn.Conv2d(32 + pruning_size, pruning_size,
                                      kernel_size=3,
                                      padding=(kernel - 1) // 2)

    def forward(self, prune):
        out_conv1_mean, out_conv2_mean, out_conv3_mean, out_conv4_mean = self.network(prune)

        out_deconv3 = self.deconv3(out_conv4_mean)
        out_deconv2 = self.deconv2(out_conv3_mean + out_deconv3)
        out_deconv1 = self.deconv1(out_conv2_mean + out_deconv2)
        out_deconv0 = self.deconv0(out_conv1_mean + out_deconv1)

        cat0 = torch.cat((prune, out_deconv0[:, :, :63, :63]), dim=-3)

        return self.output_layer(cat0)


class OrthoQuerier(nn.Module):
    def __init__(self, pruning_vector, kernel_size=3, dropout_rate=0):
        super(OrthoQuerier, self).__init__()

        in_channels = 4 + pruning_vector

        self.encoder = Encoder(in_channels, kernel_size=kernel_size, dropout_rate=dropout_rate)

        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)

        self.output_layer = nn.Conv2d(32 + in_channels, 2 * len(CON_LIST), kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2)

    def forward(self, pruning, query):
        # takes in a query and pruning, and outputs the necessary nodes everywhere
        u = torch.cat((pruning, query), dim=-3)
        # u = query
        out_conv1_mean, out_conv2_mean, out_conv3_mean, out_conv4_mean = self.encoder(u)

        out_deconv3 = self.deconv3(out_conv4_mean)
        out_deconv2 = self.deconv2(out_conv3_mean + out_deconv3)
        out_deconv1 = self.deconv1(out_conv2_mean + out_deconv2)
        out_deconv0 = self.deconv0(out_conv1_mean + out_deconv1)

        cat0 = torch.cat((u, out_deconv0[:, :, :63, :63]), dim=-3)
        out = self.output_layer(cat0)

        return out


class Orthonet(nn.Module):
    def __init__(self, input_channels, pruning_size, kernel_size, dropout_rate, time_range):
        super(Orthonet, self).__init__()

        self.base = BasePruner(input_channels, pruning_size,
                               kernel_size=kernel_size,
                               dropout_rate=dropout_rate,
                               time_range=time_range)

        self.query = OrthoQuerier(pruning_size, kernel_size=kernel_size, dropout_rate=dropout_rate)

        self.clipping = ClippingLayer(pruning_size, dropout_rate=dropout_rate, kernel=kernel_size)

        self.transition_1 = TransitionPruner(pruning_size,
                                             kernel=kernel_size,
                                             dropout=dropout_rate)
        self.transition_3 = TransitionPruner(pruning_size,
                                             kernel=kernel_size,
                                             dropout=dropout_rate)
        self.transition_9 = TransitionPruner(pruning_size,
                                             kernel=kernel_size,
                                             dropout=dropout_rate)
        self.transition_27 = TransitionPruner(pruning_size,
                                              kernel=kernel_size,
                                              dropout=dropout_rate)
        self.trans = {
            1: self.transition_1,
            3: self.transition_3,
            9: self.transition_9,
            27: self.transition_27
        }

        # self.remove_batch_norm()

    def frame(self, x, t):
        decomp = ternary_decomp(t)
        error = self.base(x)

        for delta in decomp:
            error = self.trans[delta](error)
        res = torch.normal(0, 1, size=(x.shape[0], 8, 63, 63)).to(device)
        full_vector = torch.cat((res, error), dim=1)
        return self.clipping(full_vector)

        raw = torch.stack(out)
        raw = torch.transpose(raw, 0, 1)
        raw = torch.flatten(raw, 1, 2)

        return raw

    def forward(self, x, t):
        out = []
        for i in range(t):
            # out.append(self.frame(x, 48))
            out.append(self.frame(x, i + 1))

        raw = torch.stack(out)
        raw = torch.transpose(raw, 0, 1)
        raw = torch.flatten(raw, 1, 2)

        return raw

    def remove_batch_norm(self):
        for m in self.modules():
            for c in m.children():
                if isinstance(c, nn.BatchNorm2d):
                    c.track_running_stats = False
                    c.affine = True

        return self

    def eval(self):
        super(Orthonet, self).eval()


def load_seed(indices):
    return torch.cat([torch.load('../data/ensemble/seed_' + str(int(i)) + '.pt').float().unsqueeze(0).to(device)
                      for i in indices]).detach()


@lru_cache(16)
def _load_frame(index):
    return torch.load('../data/ensemble/frames_' + str(int(index)) + '.pt').to(device).detach()


def load_frame(index, ensemble_nums, fnum):
    return _load_frame(index)[ensemble_nums, fnum].float()


def index_decomp(num, base, fracture_rate=0.25):
    num = int(num)
    decomp = [1] * (num // base) + [0] * (num % base)
    np.random.shuffle(decomp)
    return decomp
    """
    decomp = []
    num = int(num)
    while num > 0:
        decomp.append(num % base)
        num //= base

    tmp = []
    for i in range(len(decomp)):
        tmp += [i] * decomp[i]
    
    ret = []
    for x in tmp:
        if x > 0 and np.random.random() < fracture_rate:
            ret += [x - 1] * base
        else:
            ret.append(x)

    np.random.shuffle(ret)
    return ret
    """


class ELSAN(nn.Module):
    def __init__(self,
                 input_channels=16 * 2,
                 pruning_size=16,
                 noise_dim=0,
                 dropout_rate=0,
                 base=8,
                 seeds_in_batch=32,
                 ensembles_per_batch=4,  # must divide ensemble_total_size
                 ensemble_total_size=128,
                 max_out_frame=64,
                 kernel_size=3):
        super().__init__()

        # parameters
        self.k = base
        self.seeds_in_batch = seeds_in_batch
        self.ensembles_per_batch = ensembles_per_batch
        self.ensemble_total_size = ensemble_total_size
        self.max_out_frame = max_out_frame

        self.noise_shape = (noise_dim, 63, 63)

        # layers
        self.base = BasePruner(input_channels, pruning_size,
                               kernel_size=kernel_size,
                               dropout_rate=dropout_rate,
                               time_range=1)  # time range not used

        self.query = OrthoQuerier(pruning_size, kernel_size=kernel_size, dropout_rate=dropout_rate)

        self.clipping = ClippingLayer(noise_dim, pruning_size, self.ensemble_total_size,
                                      dropout_rate=dropout_rate, kernel=kernel_size)
        self.second_clipping = ClippingLayer(noise_dim, pruning_size + 2, self.ensemble_total_size,
                                             dropout_rate=dropout_rate, kernel=kernel_size)

        # trans_required = int(1 + np.floor(np.log(max_out_frame) / np.log(base)))
        trans_required = 2
        self.trans = nn.ModuleList([TransitionPruner(pruning_size,
                                                     kernel=kernel_size,
                                                     dropout=dropout_rate) for _ in range(trans_required)])

    def run_full(self, seed, t, apply_second=False):
        # uses duplicate noise
        res = [self.run_single(seed, i + 1, apply_second=apply_second)[0].to(device) for i in range(t)]
        return torch.cat(res, dim=1)

    def run_single(self, seed, t, apply_second=False):
        decomp = index_decomp(t, self.k)
        running = -1
        error = self.base(torch.unsqueeze(seed, dim=0))
        for delta in decomp:
            running += self.k ** delta
            error = self.trans[delta](error)

        # sample error repeatedly
        # error = torch.repeat_interleave(error, repeats=len(noise), dim=0)
        base = self.clipping(error)
        if apply_second:
            base = base.view(-1, 2, 63, 63)
            base = self.second_clipping(torch.cat((error, base[:1]), dim=1))
        return base, error

    def run_single_true(self, seed, t, true, apply_second=False):
        decomp = index_decomp(t, self.k)
        running = -1
        error = self.base(torch.unsqueeze(seed, dim=0))
        for delta in decomp:
            running += self.k ** delta
            error = self.trans[delta](error)

        # sample error repeatedly
        # error = torch.repeat_interleave(error, repeats=len(noise), dim=0)
        base = self.clipping(error)
        if apply_second:
            base = base.view(-1, 2, 63, 63)
            base = self.second_clipping(torch.cat((error, true[:1]), dim=1))
        return base, error

    # override for max_out_frame provided in case of warmup period
    def train_epoch(self, max_seed_index, optimizer, max_out_frame=None):
        stat_loss_curve = []

        permutation = torch.randperm(max_seed_index)

        for mini_index in range(0, (max_seed_index + self.seeds_in_batch - 1) // self.seeds_in_batch):
            seed_indices = permutation[mini_index * self.seeds_in_batch: (mini_index + 1) * self.seeds_in_batch]
            frame_seeds = load_seed(seed_indices)
            real_max_index = min(self.max_out_frame, max_out_frame) if max_out_frame else self.max_out_frame
            jump_count = 1 + torch.randint(real_max_index, seed_indices.shape)
            jump_count = 0 * jump_count + 30

            # noise = torch.normal(0, 1, size=(seed_indices.shape[0], self.ensemble_total_size, *self.noise_shape)) \
            #     .to(device)
            lsa_row_indices = torch.zeros((seed_indices.shape[0], self.ensemble_total_size), dtype=torch.long) \
                .to(device)
            lsa_col_indices = torch.zeros((seed_indices.shape[0], self.ensemble_total_size), dtype=torch.long) \
                .to(device)
            indices = torch.randint(0, self.ensemble_total_size, (seed_indices.shape[0], )).to(device)
            lsa_row_indices2 = torch.zeros((seed_indices.shape[0], self.ensemble_total_size), dtype=torch.long) \
                .to(device)
            lsa_col_indices2 = torch.zeros((seed_indices.shape[0], self.ensemble_total_size), dtype=torch.long) \
                .to(device)

            with torch.no_grad():
                for j in range(seed_indices.shape[0]):
                    rmse = torch.zeros((self.ensemble_total_size, self.ensemble_total_size))
                    # shape: [ensemble_total_size, 2 (x,y), 63 (u), 63 (v)]
                    y_pred, error = self.run_single(frame_seeds[j], jump_count[j])
                    y_pred = y_pred.view(self.ensemble_total_size, 2, 63, 63)
                    y_true = load_frame(seed_indices[j], list(range(self.ensemble_total_size)), jump_count[j] - 1)
                    for k in range(self.ensemble_total_size):
                        rmse[k] = torch.sqrt(torch.mean(torch.square((y_pred[k] - y_true)), dim=(1, 2, 3)))

                    rr, cc = linear_sum_assignment(rmse.cpu().numpy())
                    lsa_row_indices[j] = torch.tensor(rr).to(device)
                    lsa_col_indices[j] = torch.tensor(cc).to(device)

                    # second clipping (y_true still the same)
                    rmse = torch.zeros((self.ensemble_total_size, self.ensemble_total_size))
                    index = indices[j]
                    y_pred = self.second_clipping(torch.cat((error, y_true[index].unsqueeze(0)), dim=1))
                    y_pred = y_pred.view(self.ensemble_total_size, 2, 63, 63)
                    for k in range(self.ensemble_total_size):
                        rmse[k] = torch.sqrt(torch.mean(torch.square((y_pred[k] - y_true)), dim=(1, 2, 3)))

                    rr, cc = linear_sum_assignment(rmse.cpu().numpy())
                    lsa_row_indices2[j] = torch.tensor(rr).to(device)
                    lsa_col_indices2[j] = torch.tensor(cc).to(device)

            # using indices, actually do grad work through multiple sub-batches
            # space wise more efficient than naive method, but computationally 2x redundant
            # note that it may be the case that after a gradient update, the bipartite
            # matching is no longer optimal, but this is a small effect and should not
            # affect the overall training
            for i in range(self.ensemble_total_size // self.ensembles_per_batch):
                rows = lsa_row_indices[:, i * self.ensembles_per_batch: (i + 1) * self.ensembles_per_batch]
                cols = lsa_col_indices[:, i * self.ensembles_per_batch: (i + 1) * self.ensembles_per_batch]
                rows2 = lsa_row_indices2[:, i * self.ensembles_per_batch: (i + 1) * self.ensembles_per_batch]
                cols2 = lsa_col_indices2[:, i * self.ensembles_per_batch: (i + 1) * self.ensembles_per_batch]
                loss = 0

                for j, (r, c, r2, c2) in enumerate(zip(rows, cols, rows2, cols2)):
                    y_true = load_frame(seed_indices[j], c, jump_count[j] - 1)  # [ensembles_per_batch, 2, 63, 63]
                    y_pred, error = self.run_single(frame_seeds[j], jump_count[j])
                    y_pred = y_pred.view(self.ensemble_total_size, 2, 63, 63)[r]
                    # loss += torch.sqrt(torch.mean(torch.square(y_pred - y_true))) / rows.shape[0]

                    y_seed = load_frame(seed_indices[j], [indices[j]], jump_count[j] - 1)
                    y_true = load_frame(seed_indices[j], c2, jump_count[j] - 1)
                    y_pred = self.second_clipping(torch.cat((error, y_seed), dim=1))
                    y_pred = y_pred.view(self.ensemble_total_size, 2, 63, 63)[r2]
                    loss += (torch.sqrt(torch.mean(torch.square(y_pred - y_true))) +
                             torch.mean(torch.abs(torch.std(y_pred) + torch.std(y_true)))) / rows.shape[0]
                stat_loss_curve.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return stat_loss_curve


class CGAN(nn.Module):
    def __init__(self):
        super().__init__()


class MISELBO(nn.Module):
    def __init__(self):
        super().__init__()
