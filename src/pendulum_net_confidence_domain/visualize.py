"""
Ensures gen_data.py is creating proper results.
Requires manim to work
"""
import random

from manimlib import *
from model import Model, model_from_parameters, CONVEX_SPACE_DIMENSION, C_BUFFER
from tensorflow_test import step_from_parameters, PFRAMES
from gen_data import *
import torch

RUN_RATE = 0.05


# run using `manimgl visualize.py TrainingData`
class TrainingData(Scene):
    t_step = 0
    p1 = ORIGIN
    p2 = ORIGIN
    data = None
        
    def construct(self):
        np.random.seed(None)
        random.seed(None)
        data = load_data()

        ind = int(random.random() * len(data))
        print("INDEX", ind)
        self.data = data[ind]

        self.wait(1)
        self.add_mobjects()
        self.main()

        self.wait(4 / RUN_RATE)

    def add_mobjects(self):
        wire1 = Line(start=ORIGIN)
        bob1 = Dot()

        wire2 = Line(start=ORIGIN)
        bob2 = Dot()

        wire1.add_updater(lambda m, dt: m.become(Line(ORIGIN, self.p1)))
        bob1.add_updater(lambda m, dt: m.become(Dot(self.p1)))

        wire2.add_updater(lambda m, dt: m.become(Line(self.p1, self.p2)))
        bob2.add_updater(lambda m, dt: m.become(Dot(self.p2)))

        self.add(wire1, bob1, wire2, bob2)

    def main(self):
        root = Dot()
        root.add_updater(self.inc_t)
        self.add(root)

    def inc_t(self, m, dt):
        self.t_step += dt * RUN_RATE

        if dt == 0 or int(self.t_step / DT) >= len(self.data):
            return

        frame = int(self.t_step / DT)
        theta1, theta2 = self.data[frame, :2]
        self.p1 = np.array([np.sin(theta1), -np.cos(theta1), 0])
        self.p2 = self.p1 + np.array([np.sin(theta2), -np.cos(theta2), 0])


class ValidateModel(TrainingData):
    samples = 1

    frame = None
    frames = []
    model = None
    scale = None
    recorded_steps = 0

    def add_mobjects(self):
        self.model = step_from_parameters(torch.load("../../models/temp_net_parameters.pt"))
        self.frames = [*self.data[:PFRAMES]]
        self.frame = self.frames[0]

        curr = self.data[PFRAMES - 1]
        finished = []
        for i in range(self.data.shape[0]):
            print(self.data.shape, curr.shape, i)
            d = self.model(
                torch.concat([torch.flatten(self.data[i:PFRAMES]), *(finished[max(0, len(finished) - PFRAMES):])])
            )
            d = torch.flatten(d)
            v = curr[2:4] + DT * d
            curr = torch.concat([curr[0:2] + v * DT, v, d])
            self.frames.append(curr.clone())
            finished.append(curr.clone())
        #
        # for i in range(PFRAMES, self.data.shape[0]):
        #
        #     flat_prev = torch.concat((convex, convex), dim=-1)
        #     flat_prev = torch.flatten(flat_prev)
        #     in_vec = torch.concat((flat_prev, torch.tensor([1])), dim=0)
        #     frame = self.model.step(in_vec)
        #
        #     shift = convex[1: PFRAMES]
        #     convex = torch.concat((shift, torch.unsqueeze(frame[:CONVEX_SPACE_DIMENSION], dim=0)), dim=0)
        #
        #     self.frames.append(frame[:CONVEX_SPACE_DIMENSION])

        self.scale = torch.rand((self.samples, CONVEX_SPACE_DIMENSION)) * 0

        groups = VGroup()
        for _ in range(self.samples):
            wire1 = Line(start=ORIGIN)
            bob1 = Dot()

            wire2 = Line(start=ORIGIN)
            bob2 = Dot()

            group = VGroup(wire1, bob1, wire2, bob2)
            groups += group

        def update_group(mob, dt):
            # print("Frame\n", self.model.from_convex(self.frame[:CONVEX_SPACE_DIMENSION].data).data,
            #       "Width\n",self.frame[CONVEX_SPACE_DIMENSION:].data,
            #       "\nFull code", self.model.from_convex(self.model.to_convex(self.data[int(self.t / DT)])).data,
            #       "\n t1", self.data[int(self.t / DT)])
            for i, m in enumerate(mob):
                # data = self.model.from_convex((self.frame[:CONVEX_SPACE_DIMENSION]) +
                #                               self.scale[i] * (self.frame[CONVEX_SPACE_DIMENSION:])
                #                               )
                data = self.frame[:CONVEX_SPACE_DIMENSION]

                theta1, theta2 = data[:2]
                theta1, theta2 = float(theta1), float(theta2)
                p1 = np.array([np.sin(theta1), -np.cos(theta1), 0])
                p2 = p1 + np.array([np.sin(theta2), -np.cos(theta2), 0])

                opacity = 1
                m[0].become(Line(ORIGIN, p1).set_color(BLUE).set_opacity(opacity))
                m[1].become(Dot(p1).set_color(BLUE_A).set_opacity(opacity))
                m[2].become(Line(p1, p2).set_color(GREEN).set_opacity(opacity))
                m[3].become(Dot(p2).set_color(GREEN_A).set_opacity(opacity))

        groups.add_updater(update_group)

        self.add(groups)

        tex = Text("0")
        tex.to_edge(UP)
        tex.add_updater(lambda m, dt: tex.become(Text(str(int(self.t_step / DT))).to_edge(UP)))
        self.add(tex)

        super(ValidateModel, self).add_mobjects()

    def inc_t(self, m, dt):
        super(ValidateModel, self).inc_t(m, dt)
        if int(self.t_step / DT) < len(self.frames):
            self.frame = self.frames[int(self.t_step / DT)]
