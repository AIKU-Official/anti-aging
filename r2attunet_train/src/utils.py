import os
import random
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image as pil_image
from PIL.Image import Resampling as resampling
from torch import Tensor
from torchvision import transforms as T
from torchvision.transforms import functional as TF


def downscale(image: pil_image.Image, factor: int, resample: str) -> pil_image.Image:
    assert resample in {"nearest", "bilinear", "bicubic"}
    width, height = image.size
    resample = getattr(resampling, resample.upper())
    image = image.resize((width // factor, height // factor), resample=resample)
    image = image.resize((width, height), resample=resample)
    return image


def Downscale(factor: int, resample: str) -> Callable:
    return T.Lambda(lambda image: downscale(image, factor, resample))


def random_blur(
    image: pil_image.Image, kernel_size: Sequence[int], sigma: tuple[float, float]
) -> pil_image.Image:
    assert all(k % 2 == 1 for k in kernel_size)
    assert len(sigma) == 2 and sigma[0] <= sigma[1]
    kernel_size = kernel_size[torch.randint(len(kernel_size), size=(1,))]
    min_sigma, max_sigma = sigma
    sigma = (max_sigma - min_sigma) * torch.rand(size=(1,)) + min_sigma
    image = TF.gaussian_blur(image, kernel_size, sigma.item())
    return image


def RandomBlur(kernel_size: Sequence[int], sigma: tuple[float, float]) -> Callable:
    return T.Lambda(lambda image: random_blur(image, kernel_size, sigma))


def GaussianBlur(kernel_size: int, sigma: float) -> Callable:
    return T.Lambda(lambda image: TF.gaussian_blur(image, kernel_size, sigma))


def seed_everything(seed: int):
    # From https://dacon.io/codeshare/2363
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def eval_collate(batch: list[tuple[Tensor, Tensor]]):
    inputs, targets = zip(*batch)
    return torch.cat(inputs), torch.cat(targets)


def iterate_dataloader(dataloader):
    dataiter = iter(dataloader)
    while True:
        try:
            batch = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            batch = next(dataiter)
        yield batch
