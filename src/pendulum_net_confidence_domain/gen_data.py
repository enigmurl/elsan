"""
Generates data according to a set of differential equations corresponding to a double pendulum
with both wires being 1 meter long and having zero weight, attached to 1kg bobs
"""

import os
import torch

DT = 1 / 60
FRAMES = 128
INSTANCES = 2048

L1 = 1
M1 = 1
L2 = 1
M2 = 1

G = 9.806

SAVE = "data/double_pendulum.pt"


def double_pendulum(theta1, theta2):
    """
    Due to the simple nature of the problem (and the fact that the training data is static)
    the tensor is computed 'manually'

    Differential equations taken from
    https://web.mit.edu/jorloff/www/chaosTalk/double-pendulum/double-pendulum-en.html

    :param theta1 defined with respect to the positive x axis
    :param theta2 defined with respect to the positive x axis
    """

    w1, w2 = torch.zeros(len(theta1)), torch.zeros(len(theta2))  # angular velocity

    history = torch.zeros((len(theta1), FRAMES, 2))

    ft1 = torch.unsqueeze(theta1, 1)
    ft2 = torch.unsqueeze(theta2, 1)
    history[:, 0] = torch.concat([ft1, ft2], dim=1)

    for i in range(1, FRAMES):
        theta1 = history[:, i - 1, 0]
        theta2 = history[:, i - 1, 1]

        d_w1 = ((
                    -G * (2 * M1 + M2) * torch.sin(theta1)
                    - M2 * G * torch.sin(theta1 - 2 * theta2)
                    - 2 * torch.sin(theta1 - theta2) * M2 *
                    (w2 ** 2 * L1 + w1 ** 2 * L2 * torch.cos(theta1 - theta2))
                )
                /
                (L1 * (2 * M1 + M2 - M2 * torch.cos(2 * theta1 - 2 * theta2)))
                )
        d_w2 = ((
                    2 * torch.sin(theta1 - theta2) *
                    (
                        w1 ** 2 * L1 * (M1 + M2)
                        + G * (M1 + M2) * torch.cos(theta1)
                        + w2 ** 2 * L2 * M2 * torch.cos(theta1 - theta2)
                    )
                ) /
                (L2 * (2 * M1 + M2 - M2 * torch.cos(2 * theta1 - 2 * theta2)))
                )

        w1 += d_w1 * DT
        w2 += d_w2 * DT

        fw1 = torch.unsqueeze(w1, 1)
        fw2 = torch.unsqueeze(w2, 1)

        history[:, i] = history[:, i - 1] + torch.concat([fw1, fw2], dim=1) * DT

    return history


def load_data():
    if os.path.exists(SAVE):
        ret = torch.load(SAVE)
    else:
        theta1 = torch.rand(INSTANCES) * 2 * torch.pi
        theta2 = torch.rand(INSTANCES) * 2 * torch.pi

        ret = double_pendulum(theta1, theta2)

        torch.save(ret, SAVE)

    return ret


if __name__ == '__main__':
    load_data()
