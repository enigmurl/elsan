import torch
import torch.nn as nn
from statistics import *
from util import get_device, mask_tensor

device = get_device()

con_list = [-1,
            0,
            1,  # we'll add the other ones later
            ]
e_kernel = 11


def _orthocon_sample_t(model, prune, query, rand):
    rights = []
    lefts = []
    for ortho, c in zip(model.ortho_cons, con_list):
        data = torch.flatten(ortho(prune, query), 1)
        cprime = NormalDist().cdf(c)
        rights.append((0.5 + cprime / 2, data[:, 2:]))
        lefts.append((0.5 - cprime / 2, data[:, :2]))

    last_c = NormalDist().cdf(con_list[-1])

    total = lefts[::-1] + rights
    ret = []
    for k in range(len(prune)):
        arr = []
        for j in range(2):
            p = rand[k][j]
            norm_p = (p * last_c) + (1 - last_c) / 2
            res = None
            for i in range(len(total) - 1):
                l = total[i][0]
                r = total[i + 1][0]
                d = (norm_p - l) / (r - l)
                if l <= norm_p <= r:
                    res = (total[i + 1][1][k][j] - total[i][1][k][j]) * d + total[i][1][k][j]
                    break

            arr.append(float(res))
        ret.append(arr)

    return torch.tensor(ret)


def ran_sample(model, mu, pruning_error):
    rand = torch.rand((mu.shape[0], 2, mu.shape[-2], mu.shape[-1]), device=mu.device)
    query = 5 * torch.ones((mu.shape[0]), 4, mu.shape[-2], mu.shape[-1], device=mu.device)
    query[:, :2] = -query[:, :2]

    masks = mask_tensor(64)
    masks = torch.unsqueeze(masks[0], 0), torch.unsqueeze(masks[1], 0)

    for i in range(masks[0].shape[1]):
        m = masks[1][:, i]
        real_mask = torch.unsqueeze(m, 0)
        mask2 = torch.tile(real_mask, (2, 1, 1))
        mask4 = torch.tile(real_mask, (4, 1, 1))

        # compute query
        query[mask4] = -query[mask4]

        predicted = model.ortho_cons[-1](pruning_error, query)
        delta = (predicted[:, 2:] - predicted[:, :2]) * torch.rand((mu.shape[0], 2, 64, 64), device=mu.device) \
            + predicted[:, :2]
        query[:, :2][mask2] = delta[mask2]
        query[:, 2:][mask2] = delta[mask2]

    return query[:, :2]

    # # seed column
    # for c in range(mu.shape[-1]):
    #     for r in range(e_kernel):
    #         query[:, :2, r, c] = 1
    #         query[:, 2:, r, c] = -1
    #         cp = max(0, c - e_kernel + 1)
    #         converted_t = _orthocon_sample_t(model,
    #                                          pruning_error[:, :, r:r+1, cp:cp+1],
    #                                          query[:, :, :e_kernel, cp:cp + e_kernel],
    #                                          rand[:, :, r, c])
    #         query[:, :2, r, c] = converted_t
    #         query[:, 2:, r, c] = converted_t
    #
    # for r in range(e_kernel, mu.shape[-2]):
    #     for c in range(mu.shape[-1]):
    #         query[:, :2, r, c] = 1
    #         query[:, 2:, r, c] = -1
    #         cp = max(0, c - e_kernel + 1)
    #         rp = max(0, r - e_kernel + 1)
    #
    #         converted_t = _orthocon_sample_t(model,
    #                                          pruning_error[:, :, rp:rp+1, cp:cp+1],
    #                                          query[:, :, rp:rp + e_kernel, cp:cp + e_kernel],
    #                                          rand[:, :, r, c])
    #         query[:, :2, r, c] = converted_t
    #         query[:, 2:, r, c] = converted_t
    #
    # return query[:, :2]


def contains_sample(model, mu, pruning_error, y_true):
    return False

    query = torch.ones((mu.shape[0]), 4, mu.shape[-2], mu.shape[-1], device=device)
    query[:, :2] = -query[:, :2]

    delta = y_true

    count = 0

    ret = torch.full((mu.shape[0],), 1, dtype=torch.uint8, device=device)
    for c in range(mu.shape[-1]):
        for r in range(model.e_kernel):
            query[:, :2, r, c] = 1
            query[:, 2:, r, c] = -1
            cp = max(0, c - e_kernel + 1)
            rp = max(0, r - e_kernel + 1)

            ranges = model.ortho_cons[-1](pruning_error[:, :, rp:rp + 1, cp:cp + 1],
                                          query[:, :, :e_kernel, cp:cp + e_kernel],
                                          con_list[-1])

            x = torch.squeeze(torch.logical_and(delta[:, 0, r, c] >= ranges[:, 0],
                                                delta[:, 0, r, c] <= ranges[:, 2]))  # x
            y = torch.squeeze(torch.logical_and(delta[:, 1, r, c] >= ranges[:, 1],
                                                delta[:, 1, r, c] <= ranges[:, 3]))  # y
            if x and y:
                count += 1

            ret &= x
            ret &= y

            query[:, 0::2, r, c] = delta[:, 0, r, c]
            query[:, 1::2, r, c] = delta[:, 1, r, c]

    for r in range(e_kernel, mu.shape[-2]):
        for c in range(mu.shape[-1]):
            query[:, :2, r, c] = 1
            query[:, 2:, r, c] = -1
            cp = max(0, c - e_kernel + 1)
            rp = max(0, r - e_kernel + 1)

            ranges = model.ortho_cons[-1](pruning_error[:, :, rp:rp + 1, cp:cp + 1],
                                          query[:, :, rp:rp + e_kernel, cp:cp + e_kernel],
                                          con_list[-1])

            x = torch.squeeze(torch.logical_and(delta[:, 0, r, c] >= ranges[:, 0],
                                                delta[:, 0, r, c] <= ranges[:, 2]))  # x
            y = torch.squeeze(torch.logical_and(delta[:, 1, r, c] >= ranges[:, 1],
                                                delta[:, 1, r, c] <= ranges[:, 3]))  # y
            if x and y:
                count += 1

            ret &= x
            ret &= y

            query[:, 0::2, r, c] = delta[:, 0, r, c]
            query[:, 1::2, r, c] = delta[:, 1, r, c]

    print("Count", count)
    return ret


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


class Orthonet(nn.Module):
    def __init__(self, pruning_vector, kernel_size=3, dropout_rate=0):
        super(Orthonet, self).__init__()

        in_channels = pruning_vector + 4

        self.encoder = Encoder(in_channels, kernel_size=kernel_size, dropout_rate=dropout_rate)

        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)

        self.output_layer = nn.Conv2d(32 + in_channels, 4, kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2)

    def forward(self, pruning, query):
        # takes in a query and pruning, and outputs the necessary nodes everywhere
        u = torch.cat((pruning, query), dim=-3)
        out_conv1_mean, out_conv2_mean, out_conv3_mean, out_conv4_mean = self.encoder(u)

        out_deconv3 = self.deconv3(out_conv4_mean)
        out_deconv2 = self.deconv2(out_conv3_mean + out_deconv3)
        out_deconv1 = self.deconv1(out_conv2_mean + out_deconv2)
        out_deconv0 = self.deconv0(out_conv1_mean + out_deconv1)

        cat0 = torch.cat((u, out_deconv0), dim=-3)
        out = self.output_layer(cat0)

        return out


class Orthocon(nn.Module):
    def __init__(self, grid_elems, pruning_elems, dropout_rate):
        super(Orthocon, self).__init__()
        self.grid_rows = int(grid_elems ** 0.5)

        self.layer0 = conv(4, 12,
                           kernel_size=5, stride=1, dropout_rate=dropout_rate, pad=False, norm=False)
        self.layer01 = conv(12, 12,
                            kernel_size=5, stride=1, dropout_rate=dropout_rate, pad=False, norm=False)
        self.layer02 = conv(12, 12,
                            kernel_size=3, stride=1, dropout_rate=dropout_rate, pad=False, norm=False)
        self.layer2 = conv(12 + pruning_elems + 1, 16,
                           kernel_size=1, stride=1, dropout_rate=dropout_rate, norm=False)
        self.layer3 = conv(16, 12,
                           kernel_size=1, stride=1, dropout_rate=dropout_rate, norm=False)
        self.layer4 = conv(12, 8,
                           kernel_size=1, stride=1, dropout_rate=dropout_rate, norm=False)
        self.layer5 = nn.Conv2d(8, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, prune, query, c):
        d0 = self.layer0(query)
        d1 = self.layer01(d0)
        d1 = self.layer02(d1)
        d1 = torch.cat((prune, d1), dim=1)
        d1 = torch.nn.functional.pad(d1, (0, 0, 0, 0, 1, 0), value=c)  # add a layer of c

        d2 = self.layer2(d1)
        d3 = self.layer3(d2)
        d4 = self.layer4(d3)
        d5 = self.layer5(d4)

        return d5


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


class LES(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate, time_range):
        super(LES, self).__init__()
        self.spatial_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.temporal_filter = nn.Conv2d(time_range, 1, kernel_size=1, padding=0, bias=False)
        self.input_channels = input_channels
        self.time_range = time_range

        self.encoder1 = Encoder(input_channels, kernel_size, dropout_rate)
        self.encoder2 = Encoder(input_channels, kernel_size, dropout_rate)
        self.encoder3 = Encoder(input_channels, kernel_size, dropout_rate)

        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)

        self.output_layer = nn.Conv2d(32 + input_channels, output_channels, kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2)

    def forward(self, xx, error):
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
                 self.temporal_filter(u_tilde2[:, i - self.time_range + 1:i + 1, 1, :, :]).unsqueeze(2)], dim=2)
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
        return out, error


class CLES(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate, time_range, pruning_size, orthos):
        super(CLES, self).__init__()
        self.spatial_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.temporal_filter = nn.Conv2d(time_range, 1, kernel_size=1, padding=0, bias=False)
        self.input_channels = input_channels
        self.time_range = time_range
        self.pruning_size = pruning_size

        self.encoder1 = Encoder(input_channels, kernel_size, dropout_rate)
        self.encoder2 = Encoder(input_channels, kernel_size, dropout_rate)
        self.encoder3 = Encoder(input_channels, kernel_size, dropout_rate)

        self.encoder4 = Encoder(pruning_size, kernel_size, dropout_rate)

        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)

        self.output_layer = nn.Conv2d(32 + input_channels, output_channels,
                                      kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2)

        # error layers
        # takes in error and previous frame computes new error
        self.e_deconv3 = deconv(512, 256)
        self.e_deconv2 = deconv(256, 128)
        self.e_deconv1 = deconv(128, 64)
        self.e_deconv0 = deconv(64, 32)
        self.e_kernel = e_kernel
        self.e_output_layer = nn.Conv2d(32 + input_channels, pruning_size,
                                        kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2)
        self.dropout = nn.Dropout(p=0.75)

        seed = torch.seed()
        ortho_list = []
        for _ in range(orthos):
            torch.manual_seed(seed)
            # ortho_list.append(Orthocon(grid_elems=self.e_kernel * self.e_kernel,
            #                            pruning_elems=pruning_size,
            #                            dropout_rate=dropout_rate))
            ortho_list.append(Orthonet(pruning_size))

        self.ortho_cons = nn.ModuleList(ortho_list)

    def forward(self, xx, prev_error):
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
        out_conv1_error, out_conv2_error, out_conv3_error, out_conv4_error = self.encoder4(prev_error)

        out_deconv3 = self.deconv3(out_conv4_mean + out_conv4_tilde + out_conv4_prime + out_conv4_error)
        out_deconv2 = self.deconv2(out_conv3_mean + out_conv3_tilde + out_conv3_prime + out_conv3_error + out_deconv3)
        out_deconv1 = self.deconv1(out_conv2_mean + out_conv2_tilde + out_conv2_prime + out_conv2_error + out_deconv2)
        out_deconv0 = self.deconv0(out_conv1_mean + out_conv1_tilde + out_conv1_prime + out_conv1_error + out_deconv1)
        cat0 = torch.cat((xx[:, (xx_len - self.input_channels):], out_deconv0), 1)
        out = self.output_layer(cat0)

        out_deconv3 = self.e_deconv3(out_conv4_mean + out_conv4_tilde + out_conv4_prime + out_conv4_error)
        out_deconv2 = self.e_deconv2(out_conv3_mean + out_conv3_tilde + out_conv3_prime + out_conv3_error + out_deconv3)
        out_deconv1 = self.e_deconv1(out_conv2_mean + out_conv2_tilde + out_conv2_prime + out_conv2_error + out_deconv2)
        out_deconv0 = self.e_deconv0(out_conv1_mean + out_conv1_tilde + out_conv1_prime + out_conv1_error + out_deconv1)
        cat0 = torch.cat((xx[:, (xx_len - self.input_channels):], out_deconv0), 1)
        error_out = self.dropout(self.e_output_layer(cat0))

        return out, error_out
