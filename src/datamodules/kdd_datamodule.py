from optparse import Option
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


def to_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)
    elif isinstance(x, pd.DataFrame):
        return torch.from_numpy(np.array(x)).to(dtype)
    elif isinstance(x, pd.Series):
        return torch.from_numpy(np.array(x)).to(dtype)
    else:
        raise TypeError("Unknown type: {}".format(type(x)))


class KDDDataModule(LightningDataModule):
    def __init__(
        self,
        data_file: str = "data/initial_kddcup.csv",
        train_val_split: Tuple[float, float] = (0.8, 0.2),
        window_size: int = 168,
        target_size: int = 168,
        step: int = 1,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def preprocess(self, features, targets):
        target_size = self.hparams.target_size
        window_size = self.hparams.window_size
        step = self.hparams.step

        features = features[:, :-target_size].unfold(1, window_size, step)
        targets = targets[:, window_size:].unfold(1, target_size, step)

        features = rearrange(features, "n b f t -> b n t f")
        targets = rearrange(targets, "n b t -> b n t")
        return features, targets

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val:
            data = pd.read_csv(self.hparams.data_file)
            data.sort_values(by=["TurbID", "Day", "Tmstamp"], inplace=True)
            n_turb = data["TurbID"].nunique()
            n_tmstamp = data["Day"].nunique() * data["Tmstamp"].nunique()

            features = to_tensor(data.iloc[:, 3:])
            targets = to_tensor(data.iloc[:, -1])

            features = features.reshape(n_turb, n_tmstamp, -1)
            targets = targets.reshape(n_turb, n_tmstamp)

            # TODO: add train_val split
            feat_train, targ_train = self.preprocess(features, targets)
            feat_val, targ_val = self.preprocess(features, targets)

            self.data_train = TensorDataset(feat_train, targ_train)
            self.data_val = TensorDataset(feat_val, targ_val)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
