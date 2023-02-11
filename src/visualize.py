import numpy as np
from manimlib import *
import torch
import sys
import os
sys.path.append(os.getcwd())
from model import *
from util import *

device = get_device(no_mps=False)

DIR = "../data/data_64/sample_"
INDICES = range(9800, 9850)
TOFFSET = 30

COLOR_MAP = "3b1b_colormap"
FRAME_DT = 1 / 30  # amount of seconds to progress one real frame


def vector_frame(axis: Axes, vx: torch.tensor, vy: torch.tensor):
    r = vx.shape[0]
    c = vx.shape[1]

    return VectorField(lambda x, y: np.array([vx[int((y + 0.8) / 1.6 * r), int((x + 0.8) / 1.6 * c)],
                                              vy[int((y + 0.8) / 1.6 * r), int((x + 0.8) / 1.6 * c)]]),
                       axis,
                       step_multiple=0.1,
                       length_func=lambda norm: 0.125 * sigmoid(norm)
                       )


def frame(label: str, tensor: torch.tensor, org: np.ndarray, w=0.025, res=1):
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


class VisualizeSigma(Scene):

    def load_rand(self):
        index = np.random.random_integers(INDICES.start, INDICES.stop)
        ret = torch.load(DIR + str(index) + ".pt")
        return torch.unsqueeze(ret.reshape(-1, ret.shape[-2], ret.shape[-1]), dim=0).to(device)

    def model(self):
        model = CLES(input_channels=25 * 2, output_channels=2, kernel_size=3,
                     dropout_rate=0, time_range=6, pruning_size=8,
                     orthos=len(con_list))
        for param, src in zip(model.parameters(), torch.load('model_state.pt', map_location=torch.device('cpu'))):
            param.data = torch.tensor(src)
        return model.to(device)

    def construct(self) -> None:
        self.wait(0.01)

        frames = torch.cat([self.load_rand() for _ in range(64)], dim=0)
        model = self.model()
        model = nn.DataParallel(model)
        # model.eval()

        with torch.no_grad():

            root = VGroup()

            t_label = Dot().set_fill(RED, opacity=1).shift(4 * LEFT + 2.5 * UP)
            xt_frame = Dot()
            yt_frame = Dot()
            t_frame = Dot()
            t_axis = NumberPlane([-0.8, 0.8, 1], [-0.8, 0.8, 1])

            m_label = Dot().set_fill(BLUE, opacity=1).shift(4 * LEFT)
            xm_frame = Dot()
            ym_frame = Dot()
            m_frame = Dot()
            m_axis = NumberPlane([-0.8, 0.8, 1], [-0.8, 0.8, 1])

            s_label = Dot().set_fill(GREEN, opacity=1).shift(4 * LEFT + 2.5 * DOWN)
            xs_frame = Dot()
            ys_frame = Dot()
            s_frame = Dot()
            s_axis = NumberPlane([-0.8, 0.8, 1], [-0.8, 0.8, 1])

            root.add(xt_frame, yt_frame, t_frame, t_label)
            root.add(xm_frame, ym_frame, m_frame, m_label)
            root.add(xs_frame, ys_frame, s_frame, s_label)

            t = 0
            fnum = -1
            raw_frame = 0
            xx = frames[:, :TOFFSET * 2].to(device)
            prev_error = torch.zeros((64, 8, frames.shape[-2], frames.shape[-1]), device=device)
            im, prev_error = model(xx, prev_error)
            samp = ran_sample(model, im, prev_error,
                       frames[:, 60:62]).cpu().data.numpy()
            # p_value(model, im, prev_error, frames[:, 60:62])
            im = im.cpu().data.numpy()

            pm, mask = mask_tensor(64)

            def update(m, dt):
                nonlocal t, fnum, xx, im, prev_error, samp, raw_frame
                t += dt
                raw_frame = int(t / (1 / 30))
                prev = fnum
                fnum = int(t / FRAME_DT)

                print("FNUM", fnum, fnum * 2 + 1 + TOFFSET * 2, frames.shape[1], fnum * 2 + 1 + TOFFSET * 2 >= frames.shape[1])
                if fnum * 2 + 1 + TOFFSET * 2 >= frames.shape[1]:
                    return

                if fnum > prev:
                    pad = (xx.shape[-1] - prev_error.shape[-1]) // 2
                    prev_error = torch.nn.functional.pad(prev_error, (pad, pad, pad, pad))
                    im, prev_error = model(xx, prev_error)
                    xx = torch.cat([xx[:, 2:], im], 1)
                    samp = ran_sample(model, im, prev_error, frames[:, 2 * fnum + TOFFSET * 2: 2 * (fnum + 1) + TOFFSET * 2]).cpu().data.numpy()
                    # p_value(model, im, prev_error,frames[:, 2 * fnum + TOFFSET * 2: 2 * (fnum + 1) + TOFFSET * 2])
                else:
                    samp = ran_sample(model, im, prev_error, frames[:, 2 * fnum + TOFFSET * 2: 2 * (fnum + 1) + TOFFSET * 2]).cpu().data.numpy()

                    sx = samp[0, 0]
                    sy = samp[0, 1]

                    xs_frame.become(frame("x samp", sx, ORIGIN)).shift(2 * LEFT + 2.5 * DOWN)
                    ys_frame.become(frame("y samp", sy, ORIGIN)).shift(2.5 * DOWN)
                    s_frame.become(vector_frame(s_axis, sx, sy)).shift(2 * RIGHT + 2.5 * DOWN)
                    return

                tx = frames[0, 2 * fnum + 0 + TOFFSET * 2].cpu().data.numpy()
                ty = frames[0, 2 * fnum + 1 + TOFFSET * 2].cpu().data.numpy()

                real_im = im.cpu().data.numpy()
                mx = real_im[0, 0]
                my = real_im[0, 1]

                sx = samp[0, 0]
                sy = samp[0, 1]

                print("mine rmse", fnum,
                      np.sqrt(np.mean(np.square(samp[0] - frames[0, 2 * fnum + 0 + TOFFSET * 2: 2 * fnum + 2 + TOFFSET * 2].cpu().data.numpy()))))
                print("tfnt rmse", fnum,
                      np.sqrt(np.mean(np.square(
                          real_im[0] - frames[0, 2 * fnum + 0 + TOFFSET * 2: 2 * fnum + 2 + TOFFSET * 2].cpu().data.numpy()))))

                xt_frame.become(frame("x true", tx, ORIGIN)).shift(2 * LEFT + 2.5 * UP)
                yt_frame.become(frame("y true", ty, ORIGIN)).shift(2.5 * UP)
                t_frame.become(vector_frame(t_axis, tx, ty)).shift(2 * RIGHT + 2.5 * UP)

                xm_frame.become(frame("x mean", mx, ORIGIN)).shift(2 * LEFT)
                ym_frame.become(frame("y mean", my, ORIGIN))
                m_frame.become(vector_frame(m_axis, mx, my)).shift(2 * RIGHT)
                #
                xs_frame.become(frame("x samp", sx, ORIGIN)).shift(2 * LEFT + 2.5 * DOWN)
                ys_frame.become(frame("y samp", sy, ORIGIN)).shift(2.5 * DOWN)
                s_frame.become(vector_frame(s_axis, sx, sy)).shift(2 * RIGHT + 2.5 * DOWN)

                # m.become(Tex("Sigma Visualizer \\text{frame}=", str(fnum)).shift(3 * UP))

            def root_decomp(m, dt):
                nonlocal t, fnum, xx, im, prev_error, samp, raw_frame
                t += dt
                raw_frame = int(t / (1 / 30))
                prev = fnum
                fnum = int(t / FRAME_DT)

                sx = samp[0, 0].copy()
                sy = samp[0, 1].copy()

                if raw_frame < 64:
                    sx[~pm[raw_frame]] = -5
                    sy[~pm[raw_frame]] = -5

                xs_frame.become(frame("x samp", sx, ORIGIN)).shift(2 * LEFT + 2.5 * DOWN)
                ys_frame.become(frame("y samp", sy, ORIGIN)).shift(2.5 * DOWN)
                s_frame.become(vector_frame(s_axis, sx, sy)).shift(2 * RIGHT + 2.5 * DOWN)

                # m.become(Tex("Sigma Visualizer \\text{frame}=", str(fnum)).shift(3 * UP))

            self.add(root)

            root.add_updater(update)
            self.wait((frames.shape[1] - TOFFSET * 2) / 2 * FRAME_DT + 0.2)

            # root.add_updater(root_decomp)
            # self.wait(2.5)
