import numpy as np
from manimlib import *
import torch
import sys
import os
sys.path.append(os.getcwd())
from model import *
from util import *

device = get_device(no_mps=False )

DIR = "../data/data_64/sample_"
INDICES = range(6000, 7700)
TOFFSET = 6

COLOR_MAP = "3b1b_colormap"
FRAME_DT = 1 / 30  # amount of seconds to progress one real frame

SAMPLES = 2
ROW = 4

def vector_frame(axis: Axes, vx: torch.tensor, vy: torch.tensor):
    r = vx.shape[0]
    c = vx.shape[1]

    return VectorField(lambda x, y: np.array([vx[int((y + 0.8) / 1.6 * r), int((x + 0.8) / 1.6 * c)],
                                              vy[int((y + 0.8) / 1.6 * r), int((x + 0.8) / 1.6 * c)]]),
                       axis,
                       step_multiple=0.1,
                       length_func=lambda norm: 0.125 * sigmoid(norm)
                       )


def frame(label: str, tensor: torch.tensor, org: np.ndarray, w=0.0125, res=1):
    color = get_rgb_gradient_function(-3, 3, COLOR_MAP)

    rects = [
        Square(w * res)
            .set_fill(Color(rgb=color(tensor[r * res, c * res])), opacity=1)
            .set_stroke(opacity=0)
            .move_to(org + w * np.array([c * res- tensor.shape[1] / 2, r * res - tensor.shape[0] / 2, 0]))

        for c in range(tensor.shape[1] // res)
        for r in range(tensor.shape[0] // res)
    ]
    rects = VGroup(*rects)

    # label = TexText(label).next_to(org + (w * tensor.shape[0] / 2) * UP, UP)

    return VGroup(rects)  # , label)


# not exactly sure how manim interals work, but seems like construct is called twice?
# only reason I can think of is to get total time on the first run, but that seems like such a waste
render_count = 0

class VisualizeSigma(Scene):

    def load_rand(self):
        # index = np.random.random_integers(0, 256)
        data = torch.load('../data/ensemble/vis_seed.pt', map_location="cpu").to(device)
        frames = torch.load('../data/ensemble/vis_frames.pt', map_location="cpu").to(device)
        # seed = torch.load("../data/ensemble/seed_" + str(index) + ".pt").float()

        # ret = torch.load("../data/ensemble/answer_" + str(8 * index) + ".pt").float()

        return torch.flatten(data, 0, 1), frames
        #
        # huge = []
        # for i in range(64):
        #     index = np.random.random_integers(0, 550)
        #     ret = torch.load("../data/ensemble/" + str(index) + ".pt")[:, i]
        #     ret = torch.unsqueeze(ret.reshape(-1, ret.shape[-2], ret.shape[-1]), dim=0).to(device).float()
        #     xs = ret[0, :18].clone()
        #     ys = ret[0, 18:].clone()
        #     ret[0, ::2] = xs
        #     ret[0, 1::2] = ys
        #     huge.append(ret)
        # return torch.cat(huge)

    def model(self):
        model = Orthonet(input_channels=32,
                         pruning_size=16,
                         kernel_size=3,
                         dropout_rate=0,
                         time_range=1
                         ).to(device)
        for param, src in zip(model.parameters(), torch.load('model_state.pt', map_location=torch.device('mps'))):
            param.data = torch.tensor(src)
        return model.to(device)

    def construct(self) -> None:
        global render_count
        render_count += 1

        seed, frames = self.load_rand()
        seed = torch.cat([torch.unsqueeze(seed, 0) for _ in range(64)], dim=0)
        model = self.model()
        base = model.base
        trans = model.transition
        query = model.query
        model.train()

        root = VGroup()

        t_label = Dot().set_fill(RED, opacity=1).shift(4 * LEFT + 2.5 * UP)
        xt_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        yt_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        t_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        t_axis = NumberPlane([-0.8, 0.8, 1], [-0.8, 0.8, 1]).scale(0.5)

        s_label = Dot().set_fill(GREEN, opacity=1).shift(4 * LEFT + 2.5 * DOWN)
        xs_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        ys_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        s_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        s_axis = NumberPlane([-0.8, 0.8, 1], [-0.8, 0.8, 1]).scale(0.5)

        root.add(xt_frame, yt_frame, t_frame)
        root.add(xs_frame, ys_frame, s_frame)

        t = 0
        fnum = 0
        raw_frame = 0
        xx = seed
        # xx = frames[:, :12].to(device)
        error = base(xx)


        def update(m, dt):
            nonlocal t, fnum, xx, error, raw_frame
            if render_count == 1:
                # print("Early return!")
                return
            t += dt
            prev = fnum
            raw_frame = int(t / (1 / 30))
            fnum = int(t / FRAME_DT)
            mod = min(2 * fnum, frames.shape[1] - 2)

            # if fnum * 2 + 1 + TOFFSET * 2 >= 64:
            #     return

            if mod // 2 > prev:
                error = trans(error)

            # total_matching = 0
            prev, mask = mask_tensor()
            for r in range(SAMPLES):
                samp = ran_sample(query, error, frames[:, mod: mod + 2])

                # maximal_matching = 100000
                # best_i = 0
                # for i in range(64):
                #     if i in taken_indices:
                #         continue
                #     if np.sqrt(np.mean(np.square(samp[0] - frames[i, mod: mod + 2].cpu().data.numpy()))) < \
                #             maximal_matching:
                #         maximal_matching = np.sqrt(np.mean(np.square(samp[0] - frames[i, mod: mod + 2].cpu().data.numpy())))
                #         best_i = i
                # taken_indices.add(best_i)
                # total_matching += maximal_matching
                tx = frames[mod, mod].cpu().data.numpy()
                ty = frames[mod, mod + 1].cpu().data.numpy()

                sx = samp[0, 0]
                sy = samp[0, 1]
                #
                # sx[torch.logical_not(torch.sum(mask[:fnum], dim=0))] = 0
                # sy[torch.logical_not(torch.sum(mask[:fnum], dim=0))] = 0
                # sx[(torch.sum(mask[:fnum], dim=0)).bool()] = 3
                # sy[(torch.sum(mask[:fnum], dim=0)).bool()] = 3

                print(torch.sum((torch.sum(mask[:fnum], dim=0)).bool()))

                # print("mine rmse", fnum, "sample num", r,
                      # np.sqrt(np.mean(np.square(samp[0] - frames[0, mod: mod + 2].cpu().data.numpy()))))

                real_r = r // ROW
                real_c = r % ROW
                d_w = 2.65
                d_h = 0.85
                shift = RIGHT * d_w * real_c + DOWN * d_h * real_r

                xt_frame[r].become(frame("x true", tx, ORIGIN)).shift(4.00 * LEFT + 3.5 * UP + shift)
                yt_frame[r].become(frame("y true", ty, ORIGIN)).shift(3.15 * LEFT + 3.5 * UP + shift)
                # t_frame[r].become(vector_frame(t_axis, tx, ty)).shift(2.30 * LEFT + 3.5 * UP + shift)

                xs_frame[r].become(frame("x samp", sx, ORIGIN)).shift(4.00 * LEFT - 0.5 * UP + shift)
                ys_frame[r].become(frame("y samp", sy, ORIGIN)).shift(3.15 * LEFT - 0.5 * UP + shift)
                # s_frame[r].become(vector_frame(s_axis, sx, sy)).shift(2.30 * LEFT - 0.5 * UP + shift)

            # print("TOTAL MATCHING COST: ", fnum, total_matching / SAMPLES)
        self.add(root)

        root.add_updater(update)
        print("Wait", (frames.shape[1]) / 2 * FRAME_DT)
        # self.wait(1)
        self.wait((frames.shape[1]) / 2 * FRAME_DT)
        # self.wait(0.06)
