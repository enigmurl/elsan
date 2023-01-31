import torch
import torch.nn as nn


def conv(input_channels, output_channels, kernel_size, stride, dropout_rate, pad=True):
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


class Orthocon(nn.Module):
    def __init__(self, grid_elems, pruning_elems, dropout_rate):
        super(Orthocon, self).__init__()
        self.grid_rows = int(grid_elems ** 0.5)

        self.layer1 = conv(4, 12,
                           kernel_size=self.grid_rows, stride=1, dropout_rate=dropout_rate, pad=False)
        self.layer2 = conv(12 + pruning_elems + 1, 16,
                           kernel_size=1, stride=1, dropout_rate=dropout_rate)
        self.layer3 = conv(16, 12,
                           kernel_size=1, stride=1, dropout_rate=dropout_rate)
        self.layer4 = conv(12, 8,
                           kernel_size=1, stride=1, dropout_rate=dropout_rate)
        self.layer5 = nn.Conv2d(8, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, prune, query, c):
        d1 = self.layer1(query)
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
        self.e_kernel = 3
        self.e_output_layer = nn.Conv2d(32 + input_channels, pruning_size,
                                        kernel_size=self.e_kernel,
                                        padding=0)

        seed = torch.seed()
        ortho_list = []
        for _ in range(orthos):
            torch.manual_seed(seed)
            ortho_list.append(Orthocon(grid_elems=self.e_kernel * self.e_kernel,
                                       pruning_elems=pruning_size,
                                       dropout_rate=dropout_rate))

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
        error_out = self.e_output_layer(cat0)

        return out, error_out
