import torch
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
    m = max(m, _recurse(mask, level + 1, r + 1, rmax, cmin, c))
    m = max(m, _recurse(mask, level + 1, r, rmax, c + 1, cmax))
    m = max(m, _recurse(mask, level + 1, rmin, r - 1, c, cmax))

    return m


@cache
def mask_tensor(n):
    device = get_device()

    channels = int(torch.log2(torch.tensor([n * n]))) + 1
    mask = torch.zeros((channels, n, n), dtype=torch.uint8, device=device).bool()
    prev = torch.zeros((channels, n, n), dtype=torch.uint8, device=device).bool()
    # 4 then 16 then 64, then ...

    levels = _recurse(mask, 0, 0, n // 2 - 1, 0, n // 2 - 1)
    _recurse(mask, 0, n // 2, n - 1, 0, n // 2 - 1)
    _recurse(mask, 0, 0, n // 2 - 1, n // 2, n - 1)
    _recurse(mask, 0, n // 2, n - 1, n // 2, n - 1)

    for i in range(1, levels + 1):
        prev[i] = torch.logical_or(mask[i - 1], prev[i - 1])

    return prev[:levels + 1].detach(), mask[:levels + 1].detach()
