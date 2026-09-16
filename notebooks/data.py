"""Dataset loading utilities.

Wraps torchvision's MNIST / FashionMNIST datasets behind a small, uniform
interface so the rest of the code (and the notebooks) doesn't need to care
which one is in use -- FashionMNIST is a drop-in, slightly harder
replacement for MNIST, exactly as noted in the original course notebook.
"""

from typing import Literal

import torchvision
from torch.utils.data import DataLoader, Dataset

DatasetName = Literal["mnist", "fashion_mnist"]

_DATASET_CLASSES = {
    "mnist": torchvision.datasets.MNIST,
    "fashion_mnist": torchvision.datasets.FashionMNIST,
}


def get_dataset(
    name: DatasetName = "mnist",
    root: str = "data",
    train: bool = True,
    download: bool = True,
) -> Dataset:
    """Load MNIST or FashionMNIST as a greyscale, [0, 1]-normalized dataset.

    Args:
        name: Either ``"mnist"`` or ``"fashion_mnist"``.
        root: Directory to store/look for the downloaded dataset files.
        train: Whether to load the training split (``True``) or the test
            split (``False``).
        download: Whether to download the dataset if it isn't found under
            ``root``. Requires network access to the torchvision dataset
            mirrors; set to ``False`` if you've already downloaded it, or
            if you're running somewhere without that access (in which case,
            see ``dms.data.get_dataloader``'s docstring for a offline
            fallback for quick smoke tests).

    Returns:
        A ``torch.utils.data.Dataset`` yielding ``(image, label)`` pairs,
        where ``image`` is a ``(1, 28, 28)`` float tensor in ``[0, 1]``.
    """
    if name not in _DATASET_CLASSES:
        raise ValueError(f"Unknown dataset {name!r}. Expected one of {list(_DATASET_CLASSES)}.")

    transform = torchvision.transforms.ToTensor()
    dataset_cls = _DATASET_CLASSES[name]
    return dataset_cls(root=root, train=train, download=download, transform=transform)


def get_dataloader(
    name: DatasetName = "mnist",
    root: str = "data",
    batch_size: int = 128,
    train: bool = True,
    download: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a ``DataLoader`` for MNIST or FashionMNIST.

    This is a thin convenience wrapper around ``get_dataset`` +
    ``DataLoader``, matching the setup used in the original notebook:

        dataset = torchvision.datasets.MNIST(root="mnist/", train=True,
                                              download=True, transform=ToTensor())
        train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    Note:
        This requires network access to download the dataset on first use.
        If you're working somewhere offline (e.g. a sandboxed environment),
        use ``torchvision.datasets.FakeData`` directly to get correctly
        shaped random data for smoke-testing the rest of the pipeline
        without a real download.
    """
    dataset = get_dataset(name=name, root=root, train=train, download=download)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
