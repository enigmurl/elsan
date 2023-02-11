import torch
import torch.nn as nn
from statistics import *
from util import get_device, mask_tensor

device = get_device()

con_list = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]


def sample(channels: torch.tensor, t):
    ret = torch.zeros((channels.shape[0], 2, *channels.shape[2:]), device=t.device)
    # so from each channel, lets see what's the best way to do this
    prev_x, prev_y, prev_p = -5, -5, 0
    for i in range(len(con_list) + 1):
        next_x = 5 if i == len(con_list) else channels[:, 2 * i]
        next_y = 5 if i == len(con_list) else channels[:, 2 * i + 1]
        next_p = 1 if i == len(con_list) else NormalDist().cdf(con_list[i])

        mask = (t >= prev_p) & (t <= next_p)
        t_prime = (t - prev_p) / (next_p - prev_p)

        ret[:, 0][mask] = (t_prime * (next_x - prev_x) + prev_x)[mask]
        ret[:, 1][mask] = (t_prime * (next_y - prev_y) + prev_y)[mask]

        prev_x = next_x
        prev_y = next_y
        prev_p = next_p

    return ret


def ran_sample(model, mu, pruning_error, expected):
    with torch.no_grad():
        output = torch.ones((mu.shape[0]), 4, mu.shape[-2], mu.shape[-1], device=mu.device)

        masks_ = mask_tensor(64)
        masks_ = masks_[0].to(mu.device), masks_[1].to(mu.device)
        prevs, masks = torch.tile(torch.unsqueeze(masks_[0], 0), (64, 1, 1, 1)), \
                       torch.tile(torch.unsqueeze(masks_[1], 0), (64, 1, 1, 1))

        for i in range(64):
            batch = (torch.rand(63, device=mu.device) * 64).long()
            masks[1:, i] = masks_[1][batch]
            prevs[1:, i] = masks_[0][batch]

        rmse = []
        rmse_1 = []
        for i in range(masks.shape[1]):
            query = 5 * torch.ones((mu.shape[0]), 4, mu.shape[-2], mu.shape[-1], device=mu.device)
            query[:, :2] = -query[:, :2]
            m = masks[:, i]
            real_prev, real_mask = torch.unsqueeze(prevs[:, i], dim=1), torch.unsqueeze(m, dim=1)
            real_prev = torch.tile(real_prev & ~real_mask, (2, 1, 1))
            mask2 = torch.tile(real_mask, (2, 1, 1))
            mask4 = torch.tile(real_mask, (4, 1, 1))
            query[:, :2][real_prev] = expected[real_prev]
            query[:, 2:][real_prev] = expected[real_prev]
            query[0, :2][real_prev[0]] = output[0, :2][real_prev[0]]
            query[0, 2:][real_prev[0]] = output[0, :2][real_prev[0]]

            # compute query
            query[mask4] = -query[mask4]

            predicted = model._modules['module'].orthonet(pruning_error, query)
            start = NormalDist().cdf(con_list[0])
            delta = sample(predicted, start + (1 - 2 * start) * torch.rand((predicted.shape[0], *predicted.shape[2:]), device=mu.device))

            query[:, :2][mask2] = delta[mask2]
            query[:, 2:][mask2] = delta[mask2]
            output[:, :2][mask2] = delta[mask2]
            output[:, 2:][mask2] = delta[mask2]

            rmse.append(torch.mean(torch.square(query[0, :2][mask2[0]] - expected[0][mask2[0]])))
            rmse_1.append(torch.mean(torch.square(query[1:, :2][mask2[1:]] - expected[1:][mask2[1:]])))
            print(f"RMSE 0 {i} {torch.sqrt(rmse[-1]):4f}")
            print(f"RMSE 1 {i} {torch.sqrt(rmse_1[-1]):4f}")

        print(f"RMSE main {torch.sqrt(torch.mean(torch.tensor(rmse))):4f}")
        print(f"RMSE full {torch.sqrt(torch.mean(torch.tensor(rmse_1))):4f}")

        return output[:1, :2]


def find_p(channels: torch.tensor, target):
    ret = torch.zeros((channels.shape[0], 2, *channels.shape[2:]), device=target.device)
    # so from each channel, lets see what's the best way to do this
    prev_v, prev_p = -5, 0
    for i in range(len(con_list) + 1):
        next_v = 5 if i == len(con_list) else channels[:, 2 * i: 2 * i + 2]
        next_p = 1 if i == len(con_list) else NormalDist().cdf(con_list[i])

        mask = (target >= prev_v) & (target <= next_v)
        t_prime = (target - prev_v) / (next_v - prev_v) * (next_p - prev_p) + prev_p

        ret[mask] = t_prime[mask]

        prev_v = next_v
        prev_p = next_p

    ret[ret < 0.5] = 2 * ret[ret < 0.5]
    ret[ret >= 0.5] = 2 * (1 - ret[ret >= 0.5])
    ret = torch.max(ret, dim=1).values

    return ret


def p_value(model, mu, pruning_error, y_true):
    def reduce(lst):
        return min(lst)

    with torch.no_grad():
        output = torch.ones((mu.shape[0]), 4, mu.shape[-2], mu.shape[-1], device=mu.device)

        masks_ = mask_tensor(64)
        masks_ = masks_[0].to(mu.device), masks_[1].to(mu.device)
        prevs, masks = torch.tile(torch.unsqueeze(masks_[0], 0), (64, 1, 1, 1)), \
                       torch.tile(torch.unsqueeze(masks_[1], 0), (64, 1, 1, 1))

        for i in range(64):
            batch = (torch.rand(63, device=mu.device) * 64).long()
            masks[1:, i] = masks_[1][batch]
            prevs[1:, i] = masks_[0][batch]

        ps = []
        for i in range(masks.shape[1]):
            query = 5 * torch.ones((mu.shape[0]), 4, mu.shape[-2], mu.shape[-1], device=mu.device)
            query[:, :2] = -query[:, :2]
            m = masks[:, i]
            real_prev, real_mask = torch.unsqueeze(prevs[:, i], dim=1), torch.unsqueeze(m, dim=1)
            real_prev = torch.tile(real_prev & ~real_mask, (2, 1, 1))
            mask2 = torch.tile(real_mask, (2, 1, 1))
            mask4 = torch.tile(real_mask, (4, 1, 1))
            query[:, :2][real_prev] = y_true[real_prev]
            query[:, 2:][real_prev] = y_true[real_prev]
            query[0, :2][real_prev[0]] = output[0, :2][real_prev[0]]
            query[0, 2:][real_prev[0]] = output[0, :2][real_prev[0]]

            # compute query
            query[mask4] = -query[mask4]

            predicted = model._modules['module'].orthonet(pruning_error, query)
            start = NormalDist().cdf(con_list[0])
            delta = sample(predicted, start + (1 - 2 * start) * torch.rand((predicted.shape[0], *predicted.shape[2:]),
                                                                           device=mu.device))

            p = find_p(predicted, y_true)[torch.squeeze(m)]
            ps.append(reduce(p))
            sort = sorted(list(p.cpu().data.numpy()))
            print("P-value", i, "Min", ps[-1], "Mean", torch.mean(p), sort[len(sort) // 3], sort[2 * len(sort) // 3])

            query[:, :2][mask2] = y_true[mask2]
            query[:, 2:][mask2] = y_true[mask2]
            output[:, :2][mask2] = delta[mask2]
            output[:, 2:][mask2] = delta[mask2]

        print("Overall p", reduce(ps))

        return reduce(ps)


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
        u_tilde = self.spatial_filter(xx.reshape(xx.shape[0] * xx.shape[1], 1, 64, 64)) \
            .reshape(xx.shape[0], xx.shape[1], 64, 64)
        # u_prime
        u_prime = (xx - u_tilde)[:, (xx_len - self.input_channels):]
        # u_mean
        u_tilde2 = u_tilde.reshape(u_tilde.shape[0], u_tilde.shape[1] // 2, 2, 64, 64)
        u_mean = []
        for i in range(xx_len // 2 - self.input_channels // 2, xx_len // 2):
            cur_mean = torch.cat(
                [self.temporal_filter(u_tilde2[:, i - self.time_range + 1:i + 1, 0, :, :]).unsqueeze(2),
                 self.temporal_filter(u_tilde2[:, i - self.time_range + 1:i + 1, 1, :, :]).unsqueeze(2)], dim=2
            )
            u_mean.append(cur_mean)
        u_mean = torch.cat(u_mean, dim=1)
        u_mean = u_mean.reshape(u_mean.shape[0], -1, 64, 64)
        # u_tilde
        u_tilde = u_tilde[:, (self.time_range - 1) * 2:] - u_mean
        out_conv1_mean, out_conv2_mean, out_conv3_mean, out_conv4_mean = self.encoder1(u_mean)
        out_conv1_tilde, out_conv2_tilde, out_conv3_tilde, out_conv4_tilde = self.encoder2(u_tilde)
        out_conv1_prime, out_conv2_prime, out_conv3_prime, out_conv4_prime = self.encoder3(u_prime)

        out_deconv3 = self.deconv3(out_conv4_mean + out_conv4_tilde + out_conv4_prime)
        out_deconv2 = self.deconv2(out_conv3_mean + out_conv3_tilde + out_conv3_prime + out_deconv3)
        out_deconv1 = self.deconv1(out_conv2_mean + out_conv2_tilde + out_conv2_prime + out_deconv2)
        out_deconv0 = self.deconv0(out_conv1_mean + out_conv1_tilde + out_conv1_prime + out_deconv1)
        cat0 = torch.cat((xx[:, (xx_len - self.input_channels):], out_deconv0), 1)
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

        cat0 = torch.cat((prune, out_deconv0), 1)
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

        self.output_layer = nn.Conv2d(32 + in_channels, 2 * len(con_list), kernel_size=kernel_size,
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

        cat0 = torch.cat((u, out_deconv0), dim=-3)
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

    def forward(self, x):
        # might implement everything here instead?
        return x
