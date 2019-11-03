import numpy as np
from numpy import ndarray
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Callable, Tuple
from urllib import request


class Dsprites(Dataset):
    URL = (
        "https://github.com/deepmind/dsprites-dataset/raw/master/"
        "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    )

    def __init__(
        self,
        file_name: str = "/tmp/dsprites.npz",
        transform: Callable[[ndarray], Any] = lambda x: x,
    ) -> None:
        path = Path(file_name)
        assert path.suffix == ".npz", "Dsprites file name muse be end with .npz"
        if not path.exists():
            self._download(path)
        self.rawdata = np.load(file_name)
        self.path = path
        self.images = list(map(transform, self.rawdata["imgs"]))
        self.latent_values = self.rawdata["latents_values"]
        self.latent_classes = self.rawdata["latents_classes"]

    def __getitem__(self, i) -> Tuple[Any, Tuple[ndarray, ndarray]]:
        return self.images[i], (self.latent_values[i], self.latent_classes[i])

    def __len__(self) -> int:
        return len(self.images)

    def __repr__(self) -> str:
        return "Dsprites dataset in{}".format(self.path)

    def _download(self, path: Path) -> None:
        req = request.Request(self.URL)
        print("Downloading dsprites_ndarray_*.npz...")
        with request.urlopen(req) as res:
            path.write_bytes(res.read())
