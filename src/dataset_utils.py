import os
from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional

import pandas as pd
from morphic import Registry, Typed


class Dataset(Typed, Registry, ABC):
    _allow_subclass_override = True

    dataset_name: ClassVar[str]
    train_size: ClassVar[int]
    test_size: ClassVar[int]
    input_cols: ClassVar[List[str]]
    gt_cols: ClassVar[List[str]]
    seed: ClassVar[int] = 42

    data_dir: str

    @classmethod
    @abstractmethod
    def setup(cls, base_dir: str):
        pass

    def train_path(self, base_dir: Optional[str] = None) -> str:
        base_dir = base_dir or self.data_dir
        return os.path.join(
            base_dir, self.dataset_name, f"{self.dataset_name}-train.parquet"
        )

    def test_path(self, base_dir: Optional[str] = None) -> str:
        base_dir = base_dir or self.data_dir
        return os.path.join(
            base_dir, self.dataset_name, f"{self.dataset_name}-test.parquet"
        )

    def train(self) -> pd.DataFrame:
        return (
            pd.read_parquet(self.train_path())
            .sample(frac=1, random_state=self.seed)
            .reset_index(drop=True)
            .head(self.train_size)
        )

    def test(self) -> pd.DataFrame:
        return (
            pd.read_parquet(self.test_path())
            .sample(frac=1, random_state=self.seed)
            .reset_index(drop=True)
            .head(self.test_size)
        )