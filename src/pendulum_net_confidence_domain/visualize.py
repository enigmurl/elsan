"""
Ensures gen_data.py is creating proper results.
Requires manim to work
"""
import random

from manimlib import *
import torch

DT = 1 / 60


# run using `manimgl pendulum_net_confidence_domain/gen_data.py ValidateTruth` from the src directory
class ValidateTruth(Scene):

    def construct(self):
        self.wait(1)

        root = Dot()

        wire1 = Line(start=ORIGIN)
        bob1 = Dot()

        wire2 = Line(start=ORIGIN)
        bob2 = Dot()

        self.add(root, wire1, bob1, wire2, bob2)

        np.random.seed(None)
        random.seed(None)

        data = torch.load("data/double_pendulum.pt")
        print(data.shape)
        data = data[int(random.random() * len(data))]
        t = 0

        p1 = ORIGIN
        p2 = ORIGIN

        def inc_t(m, dt):
            nonlocal t, p1, p2

            t += dt

            if dt == 0 or int(t / DT) >= len(data):
                return

            frame = int(t / DT)

            theta1, theta2 = data[frame]
            p1 = np.array([np.sin(theta1), -np.cos(theta1), 0])
            p2 = p1 + np.array([np.sin(theta2), -np.cos(theta2), 0])


        root.add_updater(inc_t)

        wire1.add_updater(lambda m, dt: m.become(Line(ORIGIN, p1)))
        bob1.add_updater(lambda m, dt: m.become(Dot(p1)))

        wire2.add_updater(lambda m, dt: m.become(Line(p1, p2)))
        bob2.add_updater(lambda m, dt: m.become(Dot(p2)))


