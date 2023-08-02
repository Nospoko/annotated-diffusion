import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

class Transforms:
    def __init__(self, img_size: int):
        self.i2t = T.Compose(
            [
                T.ToTensor(),
                T.Resize((img_size, img_size)),
                T.Lambda(lambda t: 2 * t -1)
            ]
        )
        self.t2i = T.Compose(
            [
                T.Lambda(lambda t: (t + 1) / 2),
                T.Lambda(lambda t: t.permute(1, 2, 0)),
                T.Lambda(lambda t: t * 255),
                T.Lambda(lambda t: t.numpy().astype(np.uint8)),
                T.ToPILImage()
            ]
        )

    def img2torch(self, x: np.ndarray | Image.Image) -> torch.Tensor:
        return self.i2t(x)

    def torch2img(self, x: torch.Tensor) -> Image.Image:
        return self.t2i(x)