import torch
import numpy as np
import numpy.typing as npt
import random
from einops import rearrange

def dataloader(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if (dataset.size > batch_size * context_length):
        valid_end = dataset.size - 1 - batch_size * context_length
        a = random.randint(0, valid_end - 1)
        b = a + batch_size * context_length
        x = dataset[a:b]
        y = dataset[a + 1:b + 1]
    else:
        x = dataset[:-1]
        y = dataset[1:]

    X = torch.from_numpy(x).to(device).reshape(batch_size, context_length)
    Y = torch.from_numpy(y).to(device).reshape(batch_size, context_length)
    return X, Y



