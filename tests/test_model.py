"""
Tests for S2S2Net.

Based loosely on Pytorch Lightning's testing method described at
https://github.com/PyTorchLightning/pytorch-lightning/blob/1.5.10/.github/CONTRIBUTING.md#how-to-add-new-tests
"""
import pytorch_lightning as pl
import torch

import s2s2net.model


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 2

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": torch.randn(4, 512, 512),
            "mask": torch.randn(1, 2560, 2560),
            "hres": torch.randn(4, 2560, 2560),
        }


def test_s2s2net():
    """
    Run a full train, val, test and prediction loop using 1 batch.
    """
    # Get some random data
    dataloader = torch.utils.data.DataLoader(dataset=RandomDataset())

    # Initialize Model
    model: pl.LightningModule = s2s2net.model.S2S2Net()

    # Training
    trainer: pl.Trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model=model, train_dataloaders=dataloader)

    # Test inference
    predictions = trainer.predict(model=model, dataloaders=dataloader)
    segmmask, superres = predictions[0]
    assert segmmask.shape == (1, 1, 2560, 2560)
    assert superres.shape == (1, 4, 2560, 2560)
