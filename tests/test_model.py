"""
Tests for S2S2Net.

Based loosely on Pytorch Lightning's testing method described at
https://github.com/PyTorchLightning/pytorch-lightning/blob/1.5.10/.github/CONTRIBUTING.md#how-to-add-new-tests
"""
import pytorch_lightning as pl
import torch

import s2s2net.model

# %%
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 2

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": torch.randn(6, 512, 512),
            "mask": torch.randn(1, 2560, 2560),
            "hres": torch.randn(4, 2560, 2560),
        }


# %%
def test_s2s2net():
    """
    Run a full train, val, test and prediction loop using 1 batch.
    """
    # Get some random data
    dataloader = torch.utils.data.DataLoader(dataset=RandomDataset())

    # Initialize Model
    model: pl.LightningModule = s2s2net.model.S2S2Net()

    # Training
    trainer: pl.Trainer = pl.Trainer(accelerator="auto", devices=1, fast_dev_run=True)
    trainer.fit(model=model, train_dataloaders=dataloader)

    # Inference/Prediction
    predictions: list = trainer.predict(model=model, dataloaders=dataloader)
    segmmask = predictions[0]
    assert segmmask.shape == (1, 1, 2560, 2560)

    # Test/Evaluation
    scores: list[dict] = trainer.test(model=model, dataloaders=dataloader)
    assert len(scores[0].keys()) == 2
    assert scores[0]["test_f1"] >= 0.0
    assert scores[0]["test_iou"] > 0
