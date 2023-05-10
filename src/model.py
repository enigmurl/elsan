import torch
import torch.nn as nn

from hyperparameters import *
from util import get_device, mask_tensor
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
    with torch.no_grad():
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
    def __init__(self, pruning, dropout_rate=0, kernel=O_KERNEL_SIZE):
        super(ClippingLayer, self).__init__()
        self.encoder = Encoder(2 + pruning, kernel, dropout_rate)

        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)

        self.output_layer = nn.Conv2d(32 + 2 + pruning, 2,
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

        self.transition = TransitionPruner(pruning_size,
                                           kernel=kernel_size,
                                           dropout=dropout_rate)

        self.query = OrthoQuerier(pruning_size, kernel_size=kernel_size, dropout_rate=dropout_rate)

        self.clipping = ClippingLayer(pruning_size, dropout_rate=dropout_rate, kernel=kernel_size)

    def forward(self, x):
        return x

    def eval(self):
        super(Orthonet, self).eval()

        for m in self.modules():
            for c in m.children():
                if isinstance(c, nn.BatchNorm2d):
                    c.track_running_stats = False
                    c.affine = True

        return self
