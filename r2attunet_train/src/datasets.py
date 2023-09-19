import os
from glob import glob
from typing import Callable, Literal, Optional

import torch
from PIL import Image as pil_image
from torch.utils.data import Dataset


class Websites(Dataset):
    DIRNAME = "Websites"

    def __init__(
        self,
        root: str,
        split: Literal["train", "valid", "test"],
        downscaler: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super(Websites, self).__init__()
        self.filepaths = glob(os.path.join(root, self.DIRNAME, split, "*.jpg"))
        self.downscaler = downscaler
        self.transform = transform

    def __getitem__(self, index: int) -> None:
        image = pil_image.open(self.filepaths[index])

        lowres, highres = image, image
        if self.downscaler is not None:
            lowres = self.downscaler(lowres)

        if self.transform is not None:
            state = torch.get_rng_state()
            lowres = self.transform(lowres)
            torch.set_rng_state(state)
            highres = self.transform(highres)

        return lowres, highres

    def __len__(self) -> None:
        return len(self.filepaths)
