import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

from hyperparameters import *
from functools import lru_cache
from util import get_device, mask_tensor, ternary_decomp

device = get_device()
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
            nn.BatchNorm2d(output_channels),
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
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        return out_conv1, out_conv2, out_conv3, out_conv4


class ClippingLayer(nn.Module):
    def __init__(self, noise, pruning, out_size=2, dropout_rate=0, kernel=O_KERNEL_SIZE):
        super(ClippingLayer, self).__init__()
        self.encoder = Encoder(noise + pruning, kernel, dropout_rate)

        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 256)
        self.deconv1 = deconv(256, 128)
        self.deconv0 = deconv(128, 128)

        self.output_layer = nn.Conv2d(128 + noise + pruning, out_size,
                                      kernel_size=3,
                                      padding=(kernel - 1) // 2)

    def forward(self, xx):
        out_conv1_mean, out_conv2_mean, out_conv3_mean, out_conv4_mean = self.encoder(xx)

        out_deconv3 = self.deconv3(out_conv4_mean)
        out_deconv2 = self.deconv2(out_conv3_mean + out_deconv3)
        out_deconv1 = self.deconv1(out_deconv2)
        out_deconv0 = self.deconv0(out_deconv1)

        cat0 = torch.cat((xx, out_deconv0[:, :, :63, :63]), dim=-3)

        return self.output_layer(cat0)

class DownPrune(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(DownPrune, self).__init__()

        self.down = ClippingLayer(0, in_channels, out_channels)
        
    def forward(self, x):
        return self.down(x)

class UpPrune(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(UpPrune, self).__init__()

        self.up = ClippingLayer(0, in_channels, out_channels)

    def forward(self, x):
        return self.up(x)


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


class FluidFlowPredictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_full(self, seed, t):
        # uses duplicate noise
        res = [self.run_single(seed, i + 1)[0].to(device) for i in range(t)]
        return torch.cat(res, dim=1)


class ELSAN(FluidFlowPredictor):
    def __init__(self,
                 input_channels=16 * 2,
                 dropout_rate=0,
                 base=6,
                 seeds_in_batch=4,
                 ensembles_per_batch=16,  # must divide ensemble_total_size
                 ensemble_total_size=128,
                 max_out_frame=31,
                 kernel_size=3):
        super().__init__()

        # parameters
        self.k = base
        self.seeds_in_batch = seeds_in_batch
        self.ensembles_per_batch = ensembles_per_batch
        self.ensemble_total_size = ensemble_total_size
        self.max_out_frame = max_out_frame
        self.pruning_channels = 2

        # layers
        self.encoder = DownPrune(in_channels=2, out_channels=self.pruning_channels, dropout_rate=dropout_rate)
        self.decoder = UpPrune(in_channels=self.pruning_channels, out_channels=2, dropout_rate=dropout_rate)

        self.base = BasePruner(input_channels, self.pruning_channels,
                               kernel_size=kernel_size,
                               dropout_rate=dropout_rate,
                               time_range=1)  

        self.base_error = BasePruner(input_channels, self.pruning_channels,
                                     kernel_size=kernel_size,
                                     dropout_rate=dropout_rate,
                                     time_range=1)  

        # trans_required = int(1 + np.floor(np.log(max_out_frame) / np.log(base)))
        trans_required = 2
        self.trans = nn.ModuleList([TransitionPruner(self.pruning_channels,
                                                     kernel=kernel_size,
                                                     dropout=dropout_rate) for _ in range(trans_required)])

        self.trans_clip = DownPrune(in_channels=self.pruning_channels, out_channels=self.pruning_channels, dropout_rate=dropout_rate)


        self.trans_error = nn.ModuleList([TransitionPruner(self.pruning_channels,
                                                           kernel=kernel_size,
                                                           dropout=dropout_rate) for _ in range(trans_required)])

        self.trans_error_clip = DownPrune(in_channels=self.pruning_channels, out_channels=self.pruning_channels * self.ensemble_total_size, dropout_rate=dropout_rate)
        

    def run_single(self, seed, t):
        decomp = index_decomp(t, self.k)

        seed = torch.unsqueeze(seed, dim=0)

        error = self.base(seed)
        error_e = self.base_error(seed)
        for delta in decomp:
            error = self.trans[delta](error)
            error_e = self.trans_error[delta](error_e)

        error = self.trans_clip(error)
        error = torch.repeat_interleave(error, self.ensemble_total_size, dim=0)

        error_e = self.trans_error_clip(error_e) \
            .view(self.ensemble_total_size, self.pruning_channels, *error.shape[2:])

        final = self.decoder(error + error_e)
        return final, error, error_e

    def auto_encode(self, y_true):
        pruning = self.encoder(y_true)
        return self.decoder(pruning), pruning

    def parameters1(self):
        for name, parameter in self.named_parameters():
            if name.startswith("encoder.") or name.startswith("decoder."):
                yield parameter

    def parameters2(self):
        for name, parameter in self.named_parameters():
            if name.startswith("base.") or name.startswith("trans.") or name.startswith("trans_clip."):
                yield parameter

    def parameters3(self):
        for name, parameter in self.named_parameters():
            if name.startswith("base_error.") or name.startswith("trans_error.") or name.startswith("trans_error_clip."):
                yield parameter

    # override for max_out_frame provided in case of warmup period
    def train_epoch(self, max_seed_index, optimizers, epoch_num=0):
        stat_loss_curve = []

        permutation = torch.randperm(max_seed_index)
        step = 0

        max_out_frame = epoch_num // 4 + 1 

        for mini_index in range(0, (max_seed_index + self.seeds_in_batch - 1) // self.seeds_in_batch):
            seed_indices = permutation[mini_index * self.seeds_in_batch: (mini_index + 1) * self.seeds_in_batch]
            frame_seeds = load_seed(seed_indices)
            real_max_index = min(self.max_out_frame, max_out_frame) if max_out_frame else self.max_out_frame
            jump_count = 1 + torch.randint(real_max_index, seed_indices.shape)

            lsa_row_indices = torch.zeros((seed_indices.shape[0], self.ensemble_total_size), dtype=torch.long) \
                .to(device)
            lsa_col_indices = torch.zeros((seed_indices.shape[0], self.ensemble_total_size), dtype=torch.long) \
                .to(device)
            indices = torch.randint(0, self.ensemble_total_size, (seed_indices.shape[0],)).to(device)
            lsa_row_indices2 = torch.zeros((seed_indices.shape[0], self.ensemble_total_size), dtype=torch.long) \
                .to(device)
            lsa_col_indices2 = torch.zeros((seed_indices.shape[0], self.ensemble_total_size), dtype=torch.long) \
                .to(device)

            with torch.no_grad():
                for j in range(seed_indices.shape[0]):
                    rmse = torch.zeros((self.ensemble_total_size, self.ensemble_total_size))
                    # shape: [ensemble_total_size, 2 (x,y), 63 (u), 63 (v)]
                    y_pred, error, error_e = self.run_single(frame_seeds[j], jump_count[j])
                    y_true = load_frame(seed_indices[j], list(range(self.ensemble_total_size)), jump_count[j] - 1)
                    res, pruning = self.auto_encode(y_true)
                    pruning = pruning.view(self.ensemble_total_size, self.pruning_channels, *pruning.shape[2:])
                    y_true_e = pruning - torch.mean(pruning, dim=0)
                    for k in range(self.ensemble_total_size):
                        rmse[k] = torch.sqrt(torch.mean(torch.square((error_e[k] - y_true_e)), dim=(1, 2, 3)))

                    rr, cc = linear_sum_assignment(rmse.cpu().numpy())
                    lsa_row_indices[j] = torch.tensor(rr).to(device)
                    lsa_col_indices[j] = torch.tensor(cc).to(device)

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
                step += 1
                rows = lsa_row_indices[:, i * self.ensembles_per_batch: (i + 1) * self.ensembles_per_batch]
                cols = lsa_col_indices[:, i * self.ensembles_per_batch: (i + 1) * self.ensembles_per_batch]
                rows2 = lsa_row_indices2[:, i * self.ensembles_per_batch: (i + 1) * self.ensembles_per_batch]
                cols2 = lsa_col_indices2[:, i * self.ensembles_per_batch: (i + 1) * self.ensembles_per_batch]
                loss = [0, 0, 0]
                for j, (r, c, r2, c2) in enumerate(zip(rows, cols, rows2, cols2)):
                    y_true = load_frame(seed_indices[j], c, jump_count[j] - 1)  # [ensembles_per_batch, 2, 63, 63]
                    y_pred, error, error_e = self.run_single(frame_seeds[j], jump_count[j])

                    # auto_encoder_loss
                    res, pruning = self.auto_encode(y_true)
                    loss[0] += torch.mean(torch.abs(y_true - res)) / rows.shape[0]
                    # main loss
                    pruning_mean = torch.mean(pruning, dim=0)
                    # error loss
                    error_e = error_e[r]
                    loss[1] += torch.mean(torch.abs(pruning_mean - error)) / rows.shape[0]
                    loss[2] += (torch.mean(torch.abs((pruning - pruning_mean) - error_e)) \
                            + 0.5 * torch.abs(torch.mean(torch.abs(pruning - pruning_mean)) - torch.mean(torch.abs(error_e)))) / rows.shape[0]
                stat_loss_curve.append(loss[step % len(loss)].item())
                for optimizer in optimizers:
                    optimizer.zero_grad()
                loss[step % len(loss)].backward()
                optimizers[step % len(optimizers)].step()

        print(loss)
        return stat_loss_curve

class CGAN(FluidFlowPredictor):
    def __init__(self,
                 seeds_in_batch=16,
                 ensembles_per_batch=8,
                 ensemble_total_size=128,
                 max_out_frame=32,
                 noise_dim=16,
                 input_channels=16 * 2,
                 pruning_size=1,
                 kernel_size=3,
                 dropout_rate=0,
                 mse=0.35):
        super().__init__()

        self.seeds_in_batch = seeds_in_batch
        self.ensembles_per_batch = ensembles_per_batch
        self.ensemble_total_size=ensemble_total_size
        self.max_out_frame = max_out_frame
        self.noise_shape = (noise_dim, 63, 63)

        self.mse = mse

        self.generator_base = BasePruner(input_channels, pruning_size,
                               kernel_size=kernel_size,
                               dropout_rate=dropout_rate,
                               time_range=1)
        self.discriminator_base = BasePruner(input_channels, pruning_size,
                               kernel_size=kernel_size,
                               dropout_rate=dropout_rate,
                               time_range=1)

        self.generator = ClippingLayer(noise_dim, pruning_size,
                                       dropout_rate=dropout_rate, kernel=kernel_size)

        self.discriminator_0 = conv(pruning_size+2, 64, kernel_size=3, stride=2, dropout_rate = dropout_rate)
        self.discriminator_1 = conv(64 , 128, kernel_size=3, stride=2, dropout_rate = dropout_rate)
        self.discriminator_2 = conv(128, 256, kernel_size=3, stride=2, dropout_rate = dropout_rate)
        self.discriminator_3 = conv(256, 512, kernel_size=3, stride=2, dropout_rate = dropout_rate)

        self.discriminator_4 = torch.nn.Linear(512 * 4 * 4, 512)
        self.discriminator_5 = torch.nn.Linear(512, 16)
        self.discriminator_6 = torch.nn.Linear(16, 1)

        self.generator_trans = TransitionPruner(
                pruning_size,
                kernel=kernel_size,
                dropout=dropout_rate
        )

        self.discriminator_trans = TransitionPruner(
                pruning_size,
                kernel=kernel_size,
                dropout=dropout_rate
        ) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.002 / n)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def single_pruning(self, seed, t):
        running = -1
        seed = torch.unsqueeze(seed, dim=0)

        error = self.generator_base(seed)
        for _ in range(t):
            error = self.generator_trans(error)

        return error

    def discriminator_single_pruning(self, seed, t):
        seed = torch.unsqueeze(seed, dim=0)

        error = self.discriminator_base(seed)
        for _ in range(t):
            error = self.discriminator_trans(error)

        return error

    def gparameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("generator"):
                yield parameter

    def dparameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("discriminator"):
                yield parameter

    def run_single(self, seed, t):
        # advanced pruning vector
        pruning = self.single_pruning(seed, t)
        noise = torch.normal(0, 1, size=(1, *self.noise_shape)).to(device)
        return self.generator(torch.cat((noise, pruning), dim=-3)), pruning

    def train_epoch(self, max_seed_index, optimizers, epoch_num):
        stat_loss_curve = []

        max_out_frame = epoch_num // 6 + 1
        if epoch_num == 512:
            self.mse /= 2

        permutation = torch.randperm(max_seed_index)
        step = 0
        for mini_index in range(0, (max_seed_index + self.seeds_in_batch - 1) // self.seeds_in_batch):
            seed_indices = permutation[mini_index * self.seeds_in_batch: (mini_index + 1) * self.seeds_in_batch]
            seeds = load_seed(seed_indices)

            pruning_starts = []
            preds = []
            trues = []
            for j in range(seed_indices.shape[0]):
                jump = 1 + np.random.randint(0, np.minimum(self.max_out_frame, max_out_frame))
                pruning = self.single_pruning(seeds[j], jump)
                pruning_starts.append(self.discriminator_single_pruning(seeds[j], jump))
                noise = torch.normal(0, 1, size=(self.ensembles_per_batch, *self.noise_shape)).to(device)

                preds.append(self.generator(torch.cat((noise, torch.repeat_interleave(pruning, self.ensembles_per_batch, dim=0)), dim=-3)))

                indices = torch.randint(0, self.ensemble_total_size, (self.ensembles_per_batch,)).to(device)
                trues.append(load_frame(seed_indices[j], indices, jump - 1))

            preds = torch.cat(preds, dim=0)
            trues = torch.cat(trues, dim=0)
            pruning_starts = torch.repeat_interleave(torch.cat(pruning_starts + pruning_starts, dim=0),
                                                     self.ensembles_per_batch, dim=0)
            disc_input = torch.cat((preds, trues), dim=0)
            disc_input = torch.cat((disc_input, pruning_starts), dim=1)

           
            disc = self.discriminator_0(disc_input)
            disc = self.discriminator_1(disc)
            disc = self.discriminator_2(disc)
            disc = self.discriminator_3(disc)

            disc = torch.nn.functional.leaky_relu(self.discriminator_4(disc.view(disc.shape[0], -1)))
            disc = torch.nn.functional.leaky_relu(self.discriminator_5(disc))
            disc = torch.flatten(torch.sigmoid(self.discriminator_6(disc)))

            dexpected = torch.tensor([1] * (self.seeds_in_batch * self.ensembles_per_batch)
                                    + [0] * (self.seeds_in_batch * self.ensembles_per_batch), dtype=torch.float32) \
                .to(device)

            gexpected = torch.zeros_like(dexpected)

            gloss_raw = torch.nn.functional.binary_cross_entropy(disc, gexpected) 
            gloss = torch.sqrt(torch.mean(torch.square(trues - preds))) * self.mse + gloss_raw
            dloss = torch.nn.functional.binary_cross_entropy(disc, dexpected)

            # optimizer
            stat_loss_curve.append([gloss.item(), dloss.item(),
                                    torch.sqrt(torch.mean(torch.square(preds - trues))).item(),
                                    ])

            step += 1

            # make sure both receive some attention even in cases of domination
            if (gloss_raw > dloss and mini_index % 8 != 0) or mini_index % 8 == 1:
                optimizers[0].zero_grad()
                gloss.backward()
                optimizers[0].step()
            else:
                optimizers[1].zero_grad()
                dloss.backward()
                optimizers[1].step()

        return stat_loss_curve


class CVAE(nn.Module):
    def __init__(self,
                 seeds_in_batch=16,
                 ensembles_per_batch=8,
                 ensemble_total_size=128,
                 max_out_frame=32,
                 input_channels=16 * 2,
                 pruning_size=1,
                 sigma=0.1,
                 latent=1,
                 kernel_size=3,
                 dropout_rate=0):
        super().__init__()

        self.latent = latent
        self.sigma = sigma
        self.seeds_in_batch = seeds_in_batch
        self.ensembles_per_batch = ensembles_per_batch
        self.ensemble_total_size=ensemble_total_size
        self.max_out_frame = max_out_frame

        self.encoder_base = BasePruner(input_channels, pruning_size,
                               kernel_size=kernel_size,
                               dropout_rate=dropout_rate,
                               time_range=1)
        self.decoder_base = BasePruner(input_channels, pruning_size,
                               kernel_size=kernel_size,
                               dropout_rate=dropout_rate,
                               time_range=1)

        self.encoder_clip = ClippingLayer(0, pruning_size + 2, out_size=2*latent,
                                          dropout_rate=dropout_rate, kernel=kernel_size)
        self.decoder_clip = ClippingLayer(0, pruning_size + latent, out_size=2,
                                          dropout_rate=dropout_rate, kernel=kernel_size)

        self.encoder_trans = TransitionPruner(
                pruning_size,
                kernel=kernel_size,
                dropout=dropout_rate
        )

        self.decoder_trans = TransitionPruner(
                pruning_size,
                kernel=kernel_size,
                dropout=dropout_rate
        ) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.002 / n)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def enc_single_pruning(self, seed, t):
        running = -1
        seed = torch.unsqueeze(seed, dim=0)

        error = self.encoder_base(seed)
        for _ in range(t):
            error = self.encoder_trans(error)

        return error

    def dec_single_pruning(self, seed, t):
        seed = torch.unsqueeze(seed, dim=0)

        error = self.decoder_base(seed)
        for _ in range(t):
            error = self.decoder_trans(error)

        return error

    def run_single(self, seed, t):
        # advanced pruning vector
        pruning = self.dec_single_pruning(seed, t)
        noise = torch.normal(0, 1, size=(1, self.latent, 63, 63)).to(device)
        return self.decoder_clip(torch.cat((pruning, noise), dim=-3)), pruning

    def train_epoch(self, max_seed_index, optimizer, epoch_num):
        stat_loss_curve = []

        max_out_frame = epoch_num // 6 + 1

        permutation = torch.randperm(max_seed_index)
        step = 0
        for mini_index in range(0, (max_seed_index + self.seeds_in_batch - 1) // self.seeds_in_batch):
            seed_indices = permutation[mini_index * self.seeds_in_batch: (mini_index + 1) * self.seeds_in_batch]
            seeds = load_seed(seed_indices)

            pruning_starts = []
            preds = []
            trues = []

            loss = 0
            for j in range(seed_indices.shape[0]):
                jump = 1 + np.random.randint(0, np.minimum(self.max_out_frame, max_out_frame))

                e_pruning = self.enc_single_pruning(seeds[j], jump)
                e_pruning = torch.repeat_interleave(e_pruning, self.ensembles_per_batch, dim=0)

                d_pruning = self.dec_single_pruning(seeds[j], jump)
                d_pruning = torch.repeat_interleave(d_pruning, self.ensembles_per_batch, dim=0)

                indices = torch.randint(0, self.ensemble_total_size, (self.ensembles_per_batch,)).to(device)
                true_vals = load_frame(seed_indices[j], indices, jump - 1)
                
                mu_sigma = self.encoder_clip(torch.cat((e_pruning, true_vals), dim=1))
                mu = mu_sigma[:, :self.latent]
                sigma = torch.abs(mu_sigma[:, self.latent:])

                epsilon = torch.normal(0, 1, size=(self.ensembles_per_batch, self.latent, 63, 63)).to(device)
                z = mu + epsilon * sigma

                pred = self.decoder_clip(torch.cat((d_pruning, z), dim=1))

                # log probability of recovering 
                # since we're in a gaussian
                # sigma * log(e^{-x^{2}} = -sigma * differences in square
                loss += -self.sigma * torch.sqrt(torch.mean(torch.square(pred - true_vals)))
                # equation 7
                kl = 0.5 * (torch.sum(sigma) + torch.sum(mu * mu) - torch.numel(mu) - torch.sum(torch.log(sigma)))
                # normalization constant
                loss += -kl / torch.numel(mu)


            # we want to maximize
            loss = -loss
            # optimizer
            stat_loss_curve.append([loss.item()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return stat_loss_curve
