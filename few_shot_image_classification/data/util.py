"""Base Dataset class."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch
import torchvision

SequenceOrTensor = Union[Sequence, torch.Tensor]


class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset class that simply processes data and targets through optional transforms.

    Read more: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset

    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by transforms.

        Parameters
        ----------
        index

        Returns
        -------
        (datum, target)
        """

        x_support, y_support = self.data[index][0]
        x_query, y_query = self.data[index][1]

        xs_tensor = []
        xq_tensor = []

        for img_path in x_support:
            img = _read_transform_img(img_path, self.transform)
            xs_tensor.append(torch.unsqueeze(img, 0))

        for img_path in x_query:
            img = _read_transform_img(img_path, self.transform)
            xq_tensor.append(torch.unsqueeze(img, 0))

        return (
            (torch.cat(xs_tensor), torch.tensor(y_support)),
            (torch.cat(xq_tensor), torch.tensor(y_query)),
        )


def _read_transform_img(path: str, transform: Callable = None):
    img = torchvision.io.read_image(path)
    if transform:
        img = transform(img)
    return img
