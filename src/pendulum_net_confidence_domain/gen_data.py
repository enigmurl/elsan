"""
Generates data according to a set of differential equations corresponding to a double pendulum
with both wires being 1 meter long and having zero weight, attached to 1kg bobs
"""

import os
import torch
import scipy

DT = 1 / 60
FRAMES = 256
INSTANCES = 2048

L1 = 1
M1 = 1
L2 = 1
M2 = 1

G = 9.806

SAVE = "../../data/double_pendulum.pt"


def double_pendulum(theta1, theta2):
    """
    Differential equations taken from
    https://web.mit.edu/jorloff/www/chaosTalk/double-pendulum/double-pendulum-en.html
    """

    w1, w2 = torch.zeros(len(theta1)), torch.zeros(len(theta2))  # angular velocity

    history = torch.zeros((len(theta1), FRAMES, 6))

    ft1 = torch.unsqueeze(theta1, 1)
    ft2 = torch.unsqueeze(theta2, 1)
    history[:, 0] = torch.concat([ft1, ft2,
                                  torch.unsqueeze(w1, 1), torch.unsqueeze(w2, 1),
                                  torch.zeros_like(ft1), torch.zeros_like(ft2)], dim=1)

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

        fw1 = torch.unsqueeze(w1, 1) * DT
        fw2 = torch.unsqueeze(w2, 1) * DT
        fdw1 = torch.unsqueeze(d_w1, 1) * DT
        fdw2 = torch.unsqueeze(d_w2, 1) * DT

        delta = torch.concat([fw1, fw2, fdw1, fdw2], dim=1)
        delta[torch.isnan(delta)] = 0

        history[:, i, :4] = history[:, i - 1, :4] + delta
        history[:, i - 1, 4] = d_w1
        history[:, i - 1, 5] = d_w2

    # clip history to [-pi, pi)
    history = history % (2 * torch.pi)
    history = (history + 2 * torch.pi) % (2 * torch.pi)
    history[history > torch.pi] -= 2 * torch.pi

    return history


def load_data():
    # if os.path.exists(SAVE):
    #    ret = torch.load(SAVE)
    # else:
    theta1 = torch.rand(INSTANCES) * 2 * torch.pi
    theta2 = torch.rand(INSTANCES) * 2 * torch.pi

    ret = double_pendulum(theta1, theta2)

    torch.save(ret, SAVE)

    return ret


if __name__ == '__main__':
    load_data()
