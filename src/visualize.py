import numpy as np
from manimlib import *
import torch
import sys
import os

sys.path.append(os.getcwd())

from model import *
from util import *

device = get_device()

COLOR_MAP = "3b1b_colormap"
FRAME_DT = 1 / 30  # amount of seconds to progress one real frame
SAMPLES = 1


def vector_frame(axis: Axes, vx: torch.tensor, vy: torch.tensor):
    r = vx.shape[0]
    c = vx.shape[1]

    return VectorField(lambda x, y: np.array([vx[int((y + 0.8) / 1.6 * r), int((x + 0.8) / 1.6 * c)],
                                              vy[int((y + 0.8) / 1.6 * r), int((x + 0.8) / 1.6 * c)]]),
                       axis,
                       step_multiple=0.2,
                       length_func=lambda norm: 0.125 * sigmoid(norm)
                       )


def frame(label: str, tensor: torch.tensor, org: np.ndarray, w=0.022, res=1):
    color = get_rgb_gradient_function(-3, 3, COLOR_MAP)

    rects = [
        Square(w * res, stroke_width=0)
        .set_fill(Color(rgb=color(tensor[r * res, c * res])), opacity=1)
        .set_stroke(color=WHITE, opacity=0)
        .move_to(org + w * np.array([c * res - tensor.shape[1] / 2, r * res - tensor.shape[0] / 2, 0]))

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
        index = np.random.randint(0, DATA_VALIDATION_ENSEMBLES + 3)
        index = 0

        data = torch.load('../data/validate/seed_' + str(index) + '.pt', map_location="cpu").to(device).float()
        frames = torch.load('../data/validate/frames_' + str(index) + '.pt', map_location="cpu").to(device).float()
        return data, torch.flatten(frames, 1, 2)

    def model(self):
        # model = ELSAN()
        # write_parameters_into_model(model, 'model_state.pt')
        # model = CGAN()
        # write_parameters_into_model(model, 'cgan_state.pt')
        model = CVAE()
        write_parameters_into_model(model, 'cvae_state.pt')
        return model.to(device)

    def row(self, label, frames, X, use_diff=False):
        label = Tex(label).set_fill(BLACK, opacity=1).set_width(1)
        ux = torch.mean(frames[:, 0], dim=0).cpu().data.numpy()
        uy = torch.mean(frames[:, 1], dim=0).cpu().data.numpy()

        mobs = [label]
        for f in frames[:X]:
            u = 10 * (f[0].cpu().data.numpy() - ux) if use_diff else f[0].cpu().data.numpy()
            v = 10 * (f[1].cpu().data.numpy() - uy) if use_diff else f[1].cpu().data.numpy()
            print(np.mean(u), np.std(u))
            panels = [frame("", u, ORIGIN), frame("", v, ORIGIN)]
            mobs.append(VGroup(*panels).arrange(RIGHT, buff=0.1))

        group = VGroup(*mobs)
        group.arrange(RIGHT)
        return group

    def construct(self) -> None:
        global render_count
        render_count += 1

        TARG = 30
        X = 4

        cvae = CVAE()
        write_parameters_into_model(cvae, 'cvae_state.pt')

        elsan = ELSAN()
        write_parameters_into_model(elsan, 'model_state.pt')

        cgan = CGAN()
        write_parameters_into_model(cgan, 'cgan_state.pt')

        seed, frames = self.load_rand()

        ground_frames = frames[:, TARG * 2: TARG * 2 + 2]
        cgan_frames = cgan.run_many(seed, TARG, 128)
        cvae_frames = cvae.run_many(seed, TARG, 128)
        elsan_frames = elsan.run_many(seed, TARG, 128)

        row_1 = self.row("\\text{Truth}", ground_frames, X, use_diff=True)
        row_2 = self.row("\\text{ELSAN}", elsan_frames, X, use_diff=True)
        row_3 = self.row("\\text{CGAN}", cgan_frames, X, use_diff=True)
        row_4 = self.row("\\text{CVAE}", cvae_frames, X, use_diff=True)

        rows = VGroup(row_1, row_2, row_3, row_4)
        rows.arrange(DOWN)

        rect = Rectangle().set_width(10).set_height(10).set_fill(WHITE, opacity=1)
        self.add(rect)

        self.add(rows)
        self.wait(0.05)
        return

        root = VGroup()
        model = self.eval()
        xt_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        yt_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        # t_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        # t_axis = NumberPlane([-0.8, 0.8, 1], [-0.8, 0.8, 1]).scale(0.5)

        s_label = Text("Predicted").set_fill(GREEN, opacity=1).shift(2.5 * DOWN)
        t_label = Text("Ground Truth").set_fill(RED, opacity=1).shift(2.5 * UP)
        xs_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        ys_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        # s_frame = VGroup(*[Dot() for _ in range(SAMPLES)])
        # s_axis = NumberPlane([-0.8, 0.8, 1], [-0.8, 0.8, 1]).scale(0.5)

        root.add(xt_frame, yt_frame, t_label)
        root.add(xs_frame, ys_frame, s_label)

        t = 0
        fnum = 0
        xx = seed
        # error = base(xx)

        if render_count == 1:
            full = None
        else:
            full = model.run_many(xx, TARG, 32)

        def update(m, dt):
            nonlocal t, fnum, xx  # , error
            if render_count == 1:
                return
            t += dt
            prev = fnum
            fnum = int(t / FRAME_DT)
            mod = min(2 * fnum, full.shape[1] - 2)

            # if mod // 2 > prev:
            # error = trans(error)

            d = np.random.randint(0, len(frames))
            # samp = ran_sample(query, error, frames[:, mod: mod + 2])
            # samp = torch.normal(0, 1, (xx.shape[0], 2, 63, 63)).to(device)
            # samp = torch.cat([samp, error], dim=-3)
            # samp0 = clipping(samp)
            # samp = samp0
            # samp = full[:, mod: mod + 2]
            samp = full[np.random.randint(0, len(full))]
            # print(full.shape, frames.shape, mod)

            tx = frames[d, TARG * 2].cpu().data.numpy()
            ty = frames[d, TARG * 2 + 1].cpu().data.numpy()

            sx = samp[0].data.numpy()
            sy = samp[1].data.numpy()

            # print(torch.sqrt(torch.mean(torch.square(samp - frames[0, mod: mod + 2]))))

            xt_frame[0].become(frame("x true", tx, ORIGIN)).shift(0.8 * LEFT + 0.9 * UP)
            yt_frame[0].become(frame("y true", ty, ORIGIN)).shift(0.8 * RIGHT + 0.9 * UP)
            # t_frame[0].become(vector_frame(t_axis, tx, ty)).shift(2.30 * LEFT + 3.5 * UP + shift)

            xs_frame[0].become(frame("x samp", sx, ORIGIN)).shift(0.8 * LEFT - 0.9 * UP)
            ys_frame[0].become(frame("y samp", sy, ORIGIN)).shift(0.8 * RIGHT - 0.9 * UP)
            # s_frame[0].become(vector_frame(s_axis, sx, sy).shift(2.30 * LEFT - 0.5 * UP + shift)

        root.add_updater(update)
        self.add(root)
        self.wait((32) / 2 * FRAME_DT)
