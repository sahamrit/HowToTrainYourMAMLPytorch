"""
EMNIST dataset. Downloads from NIST website and saves as .npz file if not already present.
"""
from concurrent.futures import process
import json
import os
import shutil
import zipfile
import h5py
import numpy as np
import pandas as pd
import toml
import tarfile

from pathlib import Path
from typing import *
from torchvision import transforms


from few_shot_image_classification.data.base_data_module import (
    _download_raw_dataset,
    BaseDataModule,
    load_and_print_info,
)
from few_shot_image_classification.data.util import BaseDataset

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "mini_imagenet"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "mini_imagenet"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "mini_imagenet"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "processed.csv"
ESSENTIALS_FILENAME = (
    Path(__file__).parents[0].resolve() / "mini_imagenet_essentials.json"
)


class MiniImagenet(BaseDataModule):
    """
    mini-Imagenet is proposed by Matching Networks for One Shot Learning .
    In NeurIPS, 2016. This dataset consists of 50000 training images and 10000 testing images, evenly distributed across 100 classes.
    https://paperswithcode.com/dataset/miniimagenet-1
    """

    def __init__(self, args=None):
        super().__init__(args)

        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_mini_imagenet()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)
        self.mapping: dict = essentials["mapping"]
        self.inverse_mapping = {v: k for (k, v) in self.mapping.items()}
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.image_height, self.image_width)),
            ]
        )

    def prepare_data(self, *args, **kwargs) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_mini_imagenet()
        with open(ESSENTIALS_FILENAME) as f:
            _essentials = json.load(f)

    def setup(self, stage: str = None) -> None:
        df = pd.read_csv(PROCESSED_DATA_FILENAME, encoding="utf-8")

        if stage == "fit" or stage is None:
            train_df = df[df["data_split"] == "train"]
            val_df = df[df["data_split"] == "val"]

            data_train = _transform_to_episodes(
                train_df,
                self.episode_classes,
                self.support_samples,
                self.query_samples,
            )

            data_val = _transform_to_episodes(
                val_df,
                self.episode_classes,
                self.support_samples,
                self.query_samples,
            )
            self.data_train = BaseDataset(data_train, self.transform)
            self.data_val = BaseDataset(data_val, self.transform)

        if stage == "test" or stage is None:
            test_df = df[df["data_split"] == "test"]

            data_test = _transform_to_episodes(
                test_df,
                self.episode_classes,
                self.support_samples,
                self.query_samples,
            )
            self.data_test = BaseDataset(data_test, self.transform)

    def __repr__(self):
        basic = (
            f"Mini ImageNet Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\n"
            f"episode_classes: {self.episode_classes}\nsupport_samples: {self.support_samples}\n"
            f"query_samples: {self.query_samples}\n"
        )

        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        ((xs, ys), (xq, yq)) = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x support stats: {(xs.shape, xs.dtype, xs.min(), xs.mean(), xs.std(), xs.max())}\n"
            f"Batch y support stats: {(ys.shape, ys.dtype, ys.min(), ys.max())}\n"
            f"Batch x query stats: {(xq.shape, xq.dtype, xq.min(), xq.mean(), xq.std(), xq.max())}\n"
            f"Batch y query stats: {(yq.shape, yq.dtype, yq.min(), yq.max())}\n"
        )
        return basic + data


def _download_and_process_mini_imagenet() -> None:
    metadata = toml.load(METADATA_FILENAME)
    _download_raw_dataset(metadata, DL_DATA_DIRNAME, gdrive_url=True)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)


def _process_raw_dataset(filename: str, dirname: Path) -> None:
    print("Unzipping mini imagenet dataset...")
    curdir = os.getcwd()
    os.chdir(dirname)

    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    extract_dir = PROCESSED_DATA_DIRNAME / "extracted"

    if not os.path.exists(extract_dir):
        tar_file = tarfile.open(filename)
        tar_file.extractall(extract_dir)
    else:
        print("Extracted mini imagenet dataset already exists.")

    root_dir = extract_dir / "mini_imagenet_full_size"

    if not PROCESSED_DATA_FILENAME.exists():
        train_df = _imagenet_style_dataset_to_df(root_dir / "train")
        val_df = _imagenet_style_dataset_to_df(root_dir / "val")
        test_df = _imagenet_style_dataset_to_df(root_dir / "test")
        processed_df = pd.concat([train_df, val_df, test_df])

        print(
            "Saving essential dataset parameters to few_shot_image_classification/data..."
        )

        unique_classes = processed_df["image_class"].unique()
        mapping = {str(v): int(k) for k, v in enumerate(unique_classes)}
        essentials = {
            "mapping": mapping,
            "classes": len(unique_classes),
            "train_images": len(train_df),
            "test_images": len(test_df),
            "val_images": len(val_df),
            "train_classes": len(train_df["image_class"].unique()),
            "test_classes": len(test_df["image_class"].unique()),
            "val_classes": len(val_df["image_class"].unique()),
        }
        with open(ESSENTIALS_FILENAME, "w") as f:
            json.dump(essentials, f)

        processed_df["label"] = processed_df["image_class"].map(mapping)

        print("Saving processed file")
        processed_df.to_csv(PROCESSED_DATA_FILENAME, index=False, encoding="utf-8")
    else:
        print("Processed mini imagenet dataset already exists.")

    os.chdir(curdir)


def _transform_to_episodes(
    df: pd.DataFrame, episode_classes: int, support_samples: int, query_samples: int
):
    """
    df: pd.DataFrame
        columns = ["image_path", "image_class", "data_split", "label"]

    Returns
    -------
    List[Tuple[Tuple[List[str],List[int]],Tuple[List[str],List[int]]]]
        list of episodes, where each episode consists of tuple of support set
        and query set. Each set consists of list of image paths and their labels
    """
    labels: List[int] = list(df["label"].unique())
    img_paths: List[List[str]] = []
    for label in labels:
        img_paths.append(df[df["label"] == label]["image_path"].unique())

    episodes = []

    # classes which have samples >= (support_samples + query_samples) .It updates during dequeue operations.
    active_classes = set()

    for i, img_paths_per_class in enumerate(img_paths):
        if len(img_paths_per_class) >= (support_samples + query_samples):
            active_classes.add(i)

    # in each iteration sample episode_classes classes from active classes and pick support_samples
    # samples per class for support set and query_samples for query set
    while len(active_classes) >= episode_classes:
        chosen_classes = list(
            np.random.choice(list(active_classes), size=episode_classes, replace=False)
        )
        xs = []
        ys = []
        xq = []
        yq = []

        for idx in chosen_classes:
            xs.append(img_paths[idx][:support_samples])
            ys.append(labels[idx])
            xq.append(img_paths[idx][support_samples : support_samples + query_samples])
            yq.append(labels[idx])
            img_paths[idx] = img_paths[idx][support_samples + query_samples :]

            # check which classes still have >=support_samples elements
            if len(img_paths[idx]) < support_samples + query_samples:
                active_classes.remove(idx)

        xs = np.array(xs).flatten().tolist()
        xq = np.array(xq).flatten().tolist()
        ys = np.repeat(ys, support_samples).tolist()
        yq = np.repeat(yq, query_samples).tolist()

        episodes.append(
            (
                (xs, ys),
                (xq, yq),
            )
        )

    return episodes


def _imagenet_style_dataset_to_df(
    data_dir: Path,
) -> pd.DataFrame:
    """Converts ImageNet style data directory structure to `pd.DataFrame` with
    columns=["image_path", "image_class", "data_split"]

    Parameters
    ----------
    data_dir: Path
        path of the data directory which contains data in ImageNet style

    Returns
    -------
    pd.DataFrame

    Additonal Information
    ---------------------
    ImageNet Directory Structure:
        root_dir
        |--train
        |  |--cls1
        |  |  |--a.png
        |  |--cls2
        |  |  |--x.png
        |--test
        |  |--cls3
        |  |  |--b.png
        |  |--cls4
        |  |  |--y.png
        |--val
        |  |--cls5
        |  |  |--c.png
        |  |--cls6
        |  |  |--z.png
    """
    classes = set(os.listdir(data_dir))

    data_split = data_dir.name
    df = pd.DataFrame(columns=["image_path", "image_class", "data_split"])

    print(f"No of classes in {data_split} directory are {len(classes)}")
    for cls in classes:
        img_dir_per_class = os.path.join(data_dir, cls)
        imgs = list(os.listdir(img_dir_per_class))
        for img in imgs:
            img_path = os.path.join(img_dir_per_class, img)
            df = df.append(
                {
                    "image_path": str(img_path),
                    "image_class": str(cls),
                    "data_split": str(data_split),
                },
                ignore_index=True,
            )

    return df


if __name__ == "__main__":
    load_and_print_info(MiniImagenet)
