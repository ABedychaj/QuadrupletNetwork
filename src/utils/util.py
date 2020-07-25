import torch


def find_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device
