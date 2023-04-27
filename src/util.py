import torch
import numpy as np
from functools import cache
import os


def get_device(no_mps=False):
    if no_mps:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mps seems to have a memory leak
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def mask_indices(batch, elems):
    return (torch.rand(batch) * elems).long()


def _recurse(mask, level, rmin, rmax, cmin, cmax):
    if rmin > rmax or cmin > cmax:
        return -1

    r = (rmin + rmax) // 2
    c = (cmin + cmax) // 2
    mask[level, r, c] = True

    m = level
    m = max(m, _recurse(mask, level + 1, rmin, r, cmin, c - 1))
    m = max(m, _recurse(mask, level + 2, r + 1, rmax, cmin, c))
    m = max(m, _recurse(mask, level + 1, r, rmax, c + 1, cmax))
    m = max(m, _recurse(mask, level + 2, rmin, r - 1, c, cmax))

    return m


def _mod(a, b):
    ret = a % b
    return ret + b if ret < 0 else ret


def _max_index(tensor):
    best = np.random.randint(0, tensor.shape[0]), np.random.randint(0, tensor.shape[1])
    wd = -1
    for r in range(tensor.shape[0]):
        for c in range(tensor.shape[1]):
            if tensor[r][c]:
                pass
            bd = tensor.shape[0] + tensor.shape[1]
            for r1 in range(-tensor.shape[0] // 2, tensor.shape[0]):
                for c1 in range(-tensor.shape[1] // 2, tensor.shape[1]):
                    if tensor[_mod(r1 + r, tensor.shape[0])][_mod(c1 + c, tensor.shape[1])]:
                        if abs(r1) + abs(c1) < bd:
                            bd = abs(r1) + abs(c1)
            if bd > wd:
                wd = bd
                best = r, c

    return best


@cache
def mask_tensor():
    n = 63

    frm = int(np.ceil(np.log2(n)) - 1) * 2
    mask = torch.zeros((frm, n, n), dtype=torch.uint8).bool()
    prev = torch.zeros((frm, n, n), dtype=torch.uint8).bool()

    for i in range(frm):
        if i >= 1:
            prev[i] = torch.logical_or(mask[i - 1], prev[i - 1])

        # r, c = _max_index(prev[i, :block, :block])
        reverse_index = frm // 2 - i // 2
        step = int(2 ** reverse_index)
        r, c = step // 2 - 1, step // 2 - 1
        if i % 2 == 1:
            step //= 2
        mask[i, r::step, c::step] = ~prev[i, r::step, c::step]

    return prev.to(get_device()), mask.to(get_device())


if __name__ == '__main__':
    print(mask_tensor(16))