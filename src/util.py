import torch
import numpy as np
from functools import cache


def get_device(no_mps=True):
    if no_mps:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mps seems to have a memory leak
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


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
    wd = 0
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


def mask_tensor(n):
    device = get_device()

    mask = torch.zeros((n, n, n), dtype=torch.uint8, device=device).bool()
    prev = torch.zeros((n, n, n), dtype=torch.uint8, device=device).bool()
    # 4 then 16 then 64, then ...

    block = int(n ** 0.5)

    for i in range(n):
        if i >= 1:
            prev[i] = torch.logical_or(mask[i - 1], prev[i - 1])

        r, c = _max_index(prev[i, :block, :block])
        mask[i, r::block, c::block] = 1

    # _recurse(mask, 0, 0, n // 2 - 1, 0, n // 2 - 1)
    # levels = _recurse(mask, 1, n // 2, n - 1, 0, n // 2 - 1)
    # _recurse(mask, 1, 0, n // 2 - 1, n // 2, n - 1)
    # _recurse(mask, 0, n // 2, n - 1, n // 2, n - 1)

    return prev.detach(), mask.detach()


if __name__ == '__main__':
    print(mask_tensor(16))