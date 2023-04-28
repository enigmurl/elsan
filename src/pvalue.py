import torch
import numpy as np
import os 
import sys
import shutil
from statistics import *
from functools import *
from fluiddyn.util.mpi import rank
from fluidsim.solvers.ns2d.bouss.solver import Simul
device = torch.device("mps")

"""
Go in runs of 8 for a single seed with ensemble of 512
Then, have 8 lowers and 8 uppers for the frames in the future and we should be fine, except now we'll have extremely long frames (64?) 
Aim for 256 * 8 total data points then
"""

start_ensemble = int(sys.argv[1]) if len(sys.argv) > 1 else 0

ENSEMBLE_COUNT = 64
ENSEMBLE_SAMPLES = 8
NUM_ENSEMBLES = 256
SKIP = 8
IN_FRAME = 16
OUT_FRAME = 32
DATA = "../data/"

NOISE = 0.05
T_STEP = 0.2
NU = 3e-4

P_VALUES = [NormalDist().cdf(c) for c in [-2, -1, -0.5, 0, 0.5, 1, 2]]


@cache
def mask_tensor():
    n = 63

    frm = int(np.ceil(np.log2(n)) - 1) * 2
    mask = torch.zeros((frm, n, n), dtype=torch.uint8).bool()
    prev = torch.zeros((frm, n, n), dtype=torch.uint8).bool()

    for i in range(frm):
        if i >= 1:
            prev[i] = torch.logical_or(mask[i - 1], prev[i - 1])

        reverse_index = frm // 2 - i // 2
        step = int(2 ** reverse_index)
        r, c = step // 2 - 1, step // 2 - 1
        if i % 2 == 1:
            step //= 2
        mask[i, r::step, c::step] = ~prev[i, r::step, c::step]

    return prev.to(device), mask.to(device)

def rand_prev_mask():
    prev, mask = mask_tensor()
    index = int(np.random.random() * len(prev))
    return prev[index], mask[index]

def get_start():
    params = Simul.create_default_params()
    params.oper.nx = 64
    params.oper.ny = 64
    params.oper.Lx = lx = 2
    params.oper.Ly = lx
    params.oper.coef_dealiasing = 0.7
    
    params.nu_2 = NU
    params.time_stepping.t_end = (SKIP + IN_FRAME) * T_STEP
    params.init_fields.type = "in_script"
    params.output.periods_save.phys_fields = 0.1
    params.output.periods_save.spatial_means = 0.1
    
    sim = Simul(params)

    rot = np.random.normal(size=sim.oper.create_arrayX_random().shape)
    b = np.random.normal(size=sim.oper.create_arrayX_random().shape)
    sim.state.init_from_rotb(rot, b)
    sim.time_stepping.start()
    
    rot, _  = sim.output.phys_fields.get_field_to_plot(key="rot")
    b, _  = sim.output.phys_fields.get_field_to_plot(key="b")
    
    sx = []
    sy = []

    for k in range(IN_FRAME):
        t = (k + SKIP) * T_STEP
        sim.output.phys_fields.set_of_phys_files.update_times()
        
        ux, _  = sim.output.phys_fields.get_field_to_plot(key="ux", time=t, interpolate_time=True)
        uy, _  = sim.output.phys_fields.get_field_to_plot(key="uy", time=t, interpolate_time=True)

        sx.append(ux[:63, :63])
        sy.append(uy[:63, :63])
    return rot, b, 6 * torch.tensor(np.array(sx)).float().to(device), 6 * torch.tensor(np.array(sy)).float().to(device)

def single_frame(rot_s, b_s):
    shutil.rmtree(os.environ['FLUIDSIM_PATH'])
    
    params = Simul.create_default_params()
    params.oper.nx = 64
    params.oper.ny = 64
    params.oper.Lx = lx = 2
    params.oper.Ly = lx
    params.oper.coef_dealiasing = 0.7

    params.nu_2 = NU
    params.time_stepping.t_end = (OUT_FRAME + 1) * T_STEP
    params.init_fields.type = "in_script"
    params.output.periods_save.phys_fields = 0.1 
    params.output.periods_save.spatial_means = 0.1 
    
    sim = Simul(params)

    rot = rot_s + np.random.normal(size=rot_s.shape) * NOISE 
    b = b_s + np.random.normal(size=b_s.shape) * NOISE
    sim.state.init_from_rotb(rot, b)
    sim.time_stepping.start()

    sx = []
    sy = []

    for k in range(OUT_FRAME):
        t = (k + 1) * T_STEP
        sim.output.phys_fields.set_of_phys_files.update_times()

        ux, _  = sim.output.phys_fields.get_field_to_plot(key="ux", time=t, interpolate_time=True)
        uy, _  = sim.output.phys_fields.get_field_to_plot(key="uy", time=t, interpolate_time=True)

        sx.append(ux[:63, :63])
        sy.append(uy[:63, :63])

    # sim.output.phys_fields.plot()

    return 6 * torch.tensor(np.array(sx)).float().to(device), 6 * torch.tensor(np.array(sy)).float().to(device)


def simulate_ensemble(seed, rot_start, b_start, ls, us, prevs):
    # a * operation would cause duplicate references to the same empty list
    answers_x = [[[] for y in range(OUT_FRAME)] for x in range(ENSEMBLE_SAMPLES)]
    answers_y = [[[] for y in range(OUT_FRAME)] for x in range(ENSEMBLE_SAMPLES)]
    answers_p = [[[] for y in range(OUT_FRAME)] for x in range(ENSEMBLE_SAMPLES)]

    for i in range(ENSEMBLE_COUNT):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        
        sx, sy = single_frame(rot_start, b_start)
        
        for e, lz, uz in zip(range(len(ls)), ls, us):
            for f, l, u in zip(range(len(lz)), lz, uz):
                p = prevs[e][f]
                answers_x[e][f].append(sx[f])
                answers_y[e][f].append(sy[f])
                msk = torch.logical_not(torch.isclose(torch.abs(l[0]), torch.tensor([5.0]).to(device)))
                dst = torch.sqrt(torch.mean(torch.square(sx[f][msk] - l[0][msk]) + torch.square(sy[f][msk] - l[1][msk])))
                answers_p[e][f].append(dst)

    ensemble_ps = []
    for e in range(ENSEMBLE_SAMPLES):
        p_values = []
        for f, a_x, a_y, a_p in zip(range(len(answers_x[e])), answers_x[e], answers_y[e], answers_p[e]):
            a_x = torch.stack(a_x)
            a_y = torch.stack(a_y)
            a_p = torch.stack(a_p)
            a_p = torch.max(a_p) - a_p

            a_x, inds_x = torch.sort(a_x, dim=0)
            a_y, inds_y  = torch.sort(a_y, dim=0)
            sm = torch.sum(a_p)
            frame_p = []
            for p in P_VALUES:
                rn = 0
                cn = 0
                res_x = a_x[0].clone()
                res_y = a_y[0].clone()
                for x, y, j, k in zip(a_x, a_y, inds_x, inds_y):
                    rn += a_p[j]
                    cn += a_p[k]
                    mskx = [((rn - a_p[j]) / sm <= p).bool() & ((rn / sm) <= p).bool()]
                    msky = [((cn - a_p[k]) / sm <= p).bool() & ((cn / sm) <= p).bool()]
                    res_x[mskx] = x[mskx]
                    res_y[msky] = y[msky]
                frame_p.append(res_x)
                frame_p.append(res_y)
            # print("Ensemble finish mean", torch.mean(frame_p[-2] - frame_p[0]))
            p_values.append(torch.stack(frame_p))

        ensemble_ps.append(torch.stack(p_values))

    return ensemble_ps

def get_queries(seed, rot_start, b_start):
    ls = []
    us = []
    ps = [[] for _ in range(ENSEMBLE_SAMPLES)]

    for i in range(ENSEMBLE_SAMPLES):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        
        sx, sy = single_frame(rot_start, b_start)
        
        lz = []
        uz = []

        for f in range(OUT_FRAME):
            mid_x = sx[f]
            mid_y = sy[f]

            prev, mask = rand_prev_mask()

            l = torch.zeros((2, 63, 63)).to(device)
            u = torch.zeros((2, 63, 63)).to(device)


            l[0][prev] = (mid_x)[prev]
            l[1][prev] = (mid_y)[prev]
            u[0][prev] = (mid_x)[prev]
            u[1][prev] = (mid_y)[prev]

            l[0][~prev] = -5
            l[1][~prev] = -5
            u[0][~prev] = +5
            u[1][~prev] = +5

            l[0][mask] = +5
            l[1][mask] = +5
            u[0][mask] = -5
            u[1][mask] = -5
           
            lz.append(l)
            uz.append(u)

            ps[i].append(prev)

        ls.append(torch.stack(lz))
        us.append(torch.stack(uz))

    return ls, us, ps


def vis_seed(seed, rot_start, b_start):
    ret = []

    for i in range(64):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        
        sx, sy = single_frame(rot_start, b_start)
        curr = []
        for x,y in zip(sx, sy):
            curr.append(np.array(x.cpu().data))
            curr.append(np.array(y.cpu().data))

        ret.append(curr)

    return torch.tensor(np.array(ret))


if __name__ == '__main__':
    for i in range(start_ensemble, NUM_ENSEMBLES):
        rot_start, b_start, sx, sy = get_start()  # sx has form of [frames, 64, 64]

        start = torch.stack((sx, sy)).transpose(0, 1)
        if i == 0:
            vis = vis_seed(i, rot_start, b_start).float() 
            torch.save(start.float(), DATA + "ensemble/vis_seed.pt")
            torch.save(vis, DATA + "ensemble/vis_frames.pt")
        lowers, uppers, masks = get_queries(i, rot_start, b_start) # lowers has form [ENSEMBLE_SAMPLES, frames, 2, 64, 64]
        torch.save(start.cpu(), DATA + "ensemble/seed_" + str(i) + ".pt")
        answers = simulate_ensemble(i, rot_start, b_start, lowers, uppers, masks) # answers has form of [frames, 2 * len(P_VALUES), 64, 64]
        print("Ensemble finish", i)
        for j, (l, u, a) in enumerate(zip(lowers, uppers, answers)):
            torch.save(l.cpu(), DATA + "ensemble/lowers_" + str(i * ENSEMBLE_SAMPLES + j) + ".pt")
            torch.save(u.cpu(), DATA + "ensemble/uppers_" + str(i * ENSEMBLE_SAMPLES + j) + ".pt")
            torch.save(a.cpu(), DATA + "ensemble/answer_" + str(i * ENSEMBLE_SAMPLES + j) + ".pt")
            print("Ensemble finish batch:", i, j, start.shape, l.shape, u.shape, a.shape)

