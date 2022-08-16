"""Base DataModule class."""
import argparse
import pytorch_lightning as pl

from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union
from torch.utils.data import ConcatDataset, DataLoader


from few_shot_image_classification import util
from few_shot_image_classification.data.util import BaseDataset


BATCH_SIZE = 32
NUM_WORKERS = 6
K = 5
N = 5
H = 84
W = 84
query_samples = 2


class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size: int = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers: int = self.args.get("num_workers", NUM_WORKERS)
        self.episode_classes: int = self.args.get("episode_classes", N)
        self.support_samples: int = self.args.get("support_samples", K)
        self.query_samples: int = self.args.get("query_samples", query_samples)
        self.image_height: int = self.args.get("image_height", H)
        self.image_width: int = self.args.get("image_width", W)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.mapping: Collection
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data"

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help="Number of examples to operate on per forward step.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=NUM_WORKERS,
            help="Number of additional processes to load data.",
        )
        parser.add_argument(
            "--episode_classes",
            type=int,
            default=N,
            help="Number of classes per episode",
        )
        parser.add_argument(
            "--support_samples",
            type=int,
            default=K,
            help="Number of samples per episode per class for support set",
        )
        parser.add_argument(
            "--query_samples",
            type=int,
            default=query_samples,
            help="Number of samples per episode per class for query set",
        )
        parser.add_argument(
            "--image_height",
            type=int,
            default=H,
            help="Height of image",
        )
        parser.add_argument(
            "--image_width",
            type=int,
            default=W,
            help="Width of image",
        )
        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {
            "mapping": self.mapping,
            "episode_classes": self.episode_classes,
            "support_samples": self.support_samples,
            "query_samples": self.query_samples,
            "image_height": self.image_height,
            "image_width": self.image_width,
        }

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )


def load_and_print_info(data_module_class) -> None:
    """Load and print dataset info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


def _download_raw_dataset(
    metadata: Dict, dl_dirname: Path, gdrive_url: bool = False
) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return filename
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    if gdrive_url:
        util.download_gdrive_url(metadata["url"], str(filename))
    else:
        util.download_url(metadata["url"], filename)
    print("Computing SHA-256...")
    sha256 = util.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError(
            "Downloaded data file SHA-256 does not match that listed in metadata document."
        )
    return filename
