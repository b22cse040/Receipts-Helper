import os
from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional
from datasets import load_dataset
from src.dataset_utils import Dataset
from morphic import Registry

class SROIE(Dataset):
  aliases = ["SROIE", "sroie"]

  dataset_name: ClassVar[str] = "sroie"
  train_size : ClassVar[int] = 300
  test_size : ClassVar[int] = 200
  input_cols: ClassVar[List[str]] = ["bboxes", "image_path"]
  gt_cols: ClassVar[List[str]] = ["words", "ner_tags"]

  def setup(cls, base_dir: str):
    ds = load_dataset("darentang/sroie")

    dataset_dir = os.path.join(base_dir, cls.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    for split in ["train", "test"]:
      if split in ds:
        df = ds[split].to_pandas()

        # Ensure expected columns exist
        expected_cols = set(cls.input_cols + cls.gt_cols)
        missing_cols = expected_cols - set(df.columns)
        if missing_cols:
          raise ValueError(f"Missing columns: {missing_cols}")

        # Optionally trim to only the needed columns
        df = df[list(expected_cols)]

        # Save to parquet
        parquet_path = os.path.join(dataset_dir,
                                    f"{cls.dataset_name}-{split}.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"Saved {cls.dataset_name}-{split} to {parquet_path}")

    print(f"SROIE dataset successfully set up at {dataset_dir}")




if __name__ == "__main__":
  sroie_dataset = Dataset.of("SROIE", data_dir="data")
  sroie_dataset.setup("data")