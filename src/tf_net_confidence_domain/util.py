import torch


def get_device():
    # return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mps seems to have a memory leak
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
