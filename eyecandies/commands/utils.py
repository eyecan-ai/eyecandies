import typing as t
from pydantic import BaseModel, Field

if t.TYPE_CHECKING:
    import torch
    import numpy as np


class DataLoaderOptions(BaseModel):
    batch_size: int = Field(32, description="The batch size.")
    shuffle: bool = Field(False, description="Whether to shuffle the dataset.")
    num_workers: int = Field(
        4, description="How many subprocesses to use for data loading."
    )
    drop_last: bool = Field(
        False, description="Whether to drop the last incomplete batch."
    )
    prefecth: int = Field(
        2, description="Number of batches loaded in advance by each worker."
    )

    def make_dataloader(self, samples_sequence):
        from torch.utils.data import DataLoader

        return DataLoader(
            samples_sequence.torch_dataset(),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            prefetch_factor=self.prefecth,
        )


def image_tensor_to_numpy(x: "torch.Tensor") -> "np.ndarray":
    import numpy as np

    return (x.detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
