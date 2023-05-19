import torch
import numpy as np
import os 
import sys
import shutil
from hyperparameters import *
from fluidsim.solvers.ns2d.bouss.solver import Simul

from util import mask_tensor, get_device
device = get_device()


def rand_prev_mask():
    prev, mask = mask_tensor()
    index = int(np.random.random() * len(prev))
    return prev[index], mask[index]


def get_start():
    params = Simul.create_default_params()
    params.oper.nx = DATA_FRAME_SIZE + 1
    params.oper.ny = DATA_FRAME_SIZE + 1
    params.oper.Lx = lx = 2
    params.oper.Ly = lx
    params.oper.coef_dealiasing = 0.7
    
    params.nu_2 = DATA_NU
    params.time_stepping.t_end = (DATA_SKIP_FRAME + O_INPUT_LENGTH) * DATA_T_STEP
    params.init_fields.type = "in_script"
    params.output.periods_save.phys_fields = 0.1
    params.output.periods_save.spatial_means = 0.1
    
    sim = Simul(params)

    rot = np.random.normal(size=sim.oper.create_arrayX_random().shape)
    b = np.random.normal(size=sim.oper.create_arrayX_random().shape)
    sim.state.init_from_rotb(rot, b)
    sim.time_stepping.start()
    
    rot, _ = sim.output.phys_fields.get_field_to_plot(key="rot")
    b, _ = sim.output.phys_fields.get_field_to_plot(key="b")
    
    sx = []
    sy = []

    for k in range(O_INPUT_LENGTH):
        t = (k + DATA_SKIP_FRAME) * DATA_T_STEP
        sim.output.phys_fields.set_of_phys_files.update_times()
        
        ux, _ = sim.output.phys_fields.get_field_to_plot(key="ux", time=t, interpolate_time=True)
        uy, _ = sim.output.phys_fields.get_field_to_plot(key="uy", time=t, interpolate_time=True)

        sx.append(ux[:DATA_FRAME_SIZE, :DATA_FRAME_SIZE])
        sy.append(uy[:DATA_FRAME_SIZE, :DATA_FRAME_SIZE])

    return rot, b, \
           DATA_STD_SCALE * torch.tensor(np.array(sx)).float().to(device), \
           DATA_STD_SCALE * torch.tensor(np.array(sy)).float().to(device)


def single_frame(rot_s, b_s):
    shutil.rmtree(os.environ['FLUIDSIM_PATH'])
    
    params = Simul.create_default_params()
    params.oper.nx = DATA_FRAME_SIZE + 1
    params.oper.ny = DATA_FRAME_SIZE + 1
    params.oper.Lx = lx = 2
    params.oper.Ly = lx
    params.oper.coef_dealiasing = 0.7

    params.nu_2 = DATA_NU
    params.time_stepping.t_end = (DATA_OUT_FRAME + 1) * DATA_T_STEP
    params.init_fields.type = "in_script"
    params.output.periods_save.phys_fields = 0.1
    params.output.periods_save.spatial_means = 0.1 

    sim = Simul(params)

    rot = rot_s + np.random.normal(size=rot_s.shape) * DATA_NOISE
    b = b_s + np.random.normal(size=b_s.shape) * DATA_NOISE
    sim.state.init_from_rotb(rot, b)
    sim.time_stepping.start()

    sx = []
    sy = []

    for k in range(DATA_OUT_FRAME):
        t = (k + 1) * DATA_T_STEP
        sim.output.phys_fields.set_of_phys_files.update_times()

        ux, _ = sim.output.phys_fields.get_field_to_plot(key="ux", time=t, interpolate_time=True)
        uy, _ = sim.output.phys_fields.get_field_to_plot(key="uy", time=t, interpolate_time=True)

        sx.append(ux[:DATA_FRAME_SIZE, :DATA_FRAME_SIZE])
        sy.append(uy[:DATA_FRAME_SIZE, :DATA_FRAME_SIZE])

    return DATA_STD_SCALE * torch.tensor(np.array(sx)).float().to(device), \
           DATA_STD_SCALE * torch.tensor(np.array(sy)).float().to(device)


def inverse_contribution_measure(x_targ, y_targ, x_pred, y_pred):
    return torch.mean(torch.square(x_targ - x_pred) + torch.square(y_targ - y_pred)) ** CONFIDENCE_SEPARATOR_POWER


def simulate_ensemble(seed, rot_start, b_start, ls, us):
    # a * operation would cause duplicate references to the same empty list
    answers_x = [[[] for _ in range(DATA_OUT_FRAME)] for _ in range(DATA_QUERIES_PER_ENSEMBLE)]
    answers_y = [[[] for _ in range(DATA_OUT_FRAME)] for _ in range(DATA_QUERIES_PER_ENSEMBLE)]
    answers_p = [[[] for _ in range(DATA_OUT_FRAME)] for _ in range(DATA_QUERIES_PER_ENSEMBLE)]

    for i in range(DATA_COUNT_IN_ENSEMBLE):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        
        sx, sy = single_frame(rot_start, b_start)
        
        for e, lz, uz in zip(range(len(ls)), ls, us):
            for f, l, u in zip(range(len(lz)), lz, uz):
                answers_x[e][f].append(sx[f])
                answers_y[e][f].append(sy[f])
                msk = torch.logical_not(torch.isclose(torch.abs(l[0]), torch.tensor([5.0]).to(device)))
                dst = inverse_contribution_measure(sx[f][msk], sy[f][msk], l[0][msk], l[1][msk])
                answers_p[e][f].append(dst)

    ensemble_ps = []
    for e in range(DATA_QUERIES_PER_ENSEMBLE):
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
            p_values.append(torch.stack(frame_p))

        ensemble_ps.append(torch.stack(p_values))

    return ensemble_ps


def get_queries(seed, rot_start, b_start):
    ls = []
    us = []
    ps = [[] for _ in range(DATA_QUERIES_PER_ENSEMBLE)]

    for i in range(DATA_QUERIES_PER_ENSEMBLE):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        
        sx, sy = single_frame(rot_start, b_start)
        
        lz = []
        uz = []

        for f in range(DATA_OUT_FRAME):
            mid_x = sx[f]
            mid_y = sy[f]

            prev, mask = rand_prev_mask()

            l = torch.zeros((2, DATA_FRAME_SIZE, DATA_FRAME_SIZE)).to(device)
            u = torch.zeros((2, DATA_FRAME_SIZE, DATA_FRAME_SIZE)).to(device)

            l[0][prev] = mid_x[prev]
            l[1][prev] = mid_y[prev]
            u[0][prev] = mid_x[prev]
            u[1][prev] = mid_y[prev]

            l[0][~prev] = DATA_LOWER_UNKNOWN_VALUE
            l[1][~prev] = DATA_LOWER_UNKNOWN_VALUE
            u[0][~prev] = DATA_UPPER_UNKNOWN_VALUE
            u[1][~prev] = DATA_UPPER_UNKNOWN_VALUE

            l[0][mask] = DATA_LOWER_QUERY_VALUE
            l[1][mask] = DATA_LOWER_QUERY_VALUE
            u[0][mask] = DATA_UPPER_QUERY_VALUE
            u[1][mask] = DATA_UPPER_QUERY_VALUE
           
            lz.append(l)
            uz.append(u)

            ps[i].append(prev)

        ls.append(torch.stack(lz))
        us.append(torch.stack(uz))

    return ls, us, ps


def vis_seed(bs, rot_start, b_start):
    ret = []

    for i in range(bs):
        sx, sy = single_frame(rot_start, b_start)
        curr = []
        for x, y in zip(sx, sy):
            curr.append([np.array(x.cpu().data), np.array(y.cpu().data)])

        ret.append(curr)

    return torch.tensor(np.array(ret))


if __name__ == '__main__':
    with torch.no_grad():
        if len(sys.argv) < 3:
            print("LOG", "invalid usage")
            exit(1)

        if sys.argv[1] == 'validate':
            # for e in range(int(sys.argv[2]), DATA_VALIDATION_ENSEMBLES):
            e = int(sys.argv[2])

            rot_start, b_start, sx, sy = get_start()

            start = torch.stack((sx, sy)).transpose(0, 1)
            vis = vis_seed(V_BATCH_SIZE, rot_start, b_start).float()
            torch.save(start.cpu().half(), DATA_DIR + "validate/seed_" + str(e) + ".pt")
            torch.save(vis.cpu().half(), DATA_DIR + "validate/frames_" + str(e) + ".pt")

            print("LOG", "finish", e)

        elif sys.argv[1] == 'training':
            # for e in range(int(sys.argv[2]), DATA_NUM_ENSEMBLES):
            e = int(sys.argv[2])
            rot_start, b_start, sx, sy = get_start()

            start = torch.stack((sx, sy)).transpose(0, 1)
            start = torch.flatten(start, 0, 1)
            vis = vis_seed(DATA_COUNT_IN_ENSEMBLE, rot_start, b_start).float()
            torch.save(start.cpu().half(), DATA_DIR + "ensemble/seed_" + str(e) + ".pt")
            torch.save(vis.cpu().half(), DATA_DIR + "ensemble/frames_" + str(e) + ".pt")

            print("LOG", "finish", e)
        else:
            print("LOG", "invalid mode")

