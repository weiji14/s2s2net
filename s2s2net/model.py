"""
The S2S2Net model architecture and data loading modules.

Code structure adapted from Pytorch Lightning project seed at
https://github.com/PyTorchLightning/deep-learning-project-template
"""

import os
import typing

import numpy as np
import pytorch_lightning as pl
import rasterio.crs
import torch
import torchgeo
import torchgeo.datasets
import torchgeo.samplers
from torch.nn import functional as F

# %%
class S2S2Net(pl.LightningModule):
    """
    Neural network for performing Super-Resolution Semantic Segmentation on
    Sentinel 2 optical satellite imagery.

    Implemented using Pytorch Lightning.
    """

    def __init__(self):
        """
        Define layers of the Convolutional Neural Network.

        TODO
        """
        super().__init__()

        self.conv_in = torch.nn.Conv2d(
            in_channels=4, out_channels=42, kernel_size=[3, 3], padding=[1, 1]
        )
        self.upsample = torch.nn.Upsample(scale_factor=5, mode="nearest")
        self.conv_out = torch.nn.Conv2d(
            in_channels=42, out_channels=1, kernel_size=[1, 1], padding=[0, 0]
        )

    def forward(self, x: torch.Tensor) -> typing.Tuple:
        """
        Forward pass (Inference/Prediction).

        TODO
        """
        _x = self.conv_in(x.float())
        _x = self.upsample(_x)
        _x = self.conv_out(_x)

        return _x

    def training_step(
        self, batch: typing.Dict[str, torch.Tensor], batch_idx: int
    ) -> dict:
        """
        Logic for the neural network's training loop.
        """
        x = batch["image"]
        y = batch["mask"]

        y_hat: torch.Tensor = self(x.float())

        loss: float = F.l1_loss(input=y_hat, target=y)
        # loss: float = F.nll_loss(input=y_hat, target=y

        return loss

    def configure_optimizers(self):
        """
        Optimizing function used to reduce the loss, so that the predicted
        label gets as close as possible to the groundtruth label.

        Using the Adam optimizer with a learning rate of 0.01. See:

        - Kingma, D. P., & Ba, J. (2017). Adam: A Method for Stochastic
          Optimization. ArXiv:1412.6980 [Cs]. http://arxiv.org/abs/1412.6980

        Documentation at:
        https://pytorch-lightning.readthedocs.io/en/1.4.5/common/optimizers.html
        """
        return torch.optim.Adam(params=self.parameters(), lr=0.01, weight_decay=0.0005)


# %%
class S2S2Dataset(torchgeo.datasets.VisionDataset):
    """
    Training dataset for the Sentinel-2 Super Resolution Segmentation model.

    There are 3 image triples:
    1. image - Sentinel-2 RGB-NIR image at 10m resolution (4, 512, 512)
    2. mask - Binary segmentation mask at 2m resolution (1, 2560, 2560)
    3. hres - High resolution RGB-NIR image at 2m resolution (4, 2560, 2560)
    """

    def __init__(
        self,
        root: str = "SuperResolution/chips/npy",
        transforms: typing.Optional[
            typing.Callable[
                [typing.Dict[str, torch.Tensor]], typing.Dict[str, torch.Tensor]
            ]
        ] = None,
    ):
        self.root = root
        self.transforms = transforms
        self.ids: list = [
            int(file[5:9])
            for file in sorted(os.listdir(os.path.join(self.root, "image")))
        ]

    def __getitem__(self, index: int = 0) -> typing.Dict[str, torch.Tensor]:
        """
        Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index
        """

        image: torch.Tensor = torch.from_numpy(
            np.load(
                file=os.path.join(self.root, "image", f"SEN2_{index:04d}.npy")
            ).astype(np.int16)
        )
        mask: torch.Tensor = torch.from_numpy(
            np.load(file=os.path.join(self.root, "mask", f"MASK_{index:04d}.npy"))
        )
        hres: torch.Tensor = torch.from_numpy(
            np.load(
                file=os.path.join(self.root, "hres", f"HRES_{index:04d}.npy")
            ).astype(np.int16)
        )

        sample: dict = {"image": image, "mask": mask, "hres": hres}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """
        Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)


# %%
class S2S2DataModule(pl.LightningDataModule):
    """
    Data preparation code to load Sentinel2 image data and the high resolution
    image mask into a Pytorch DataLoader module (a fancy for-loop generator).

    TODO
    """

    def __init__(self):
        """
        TODO
        """
        super().__init__()

    def prepare_data(self):
        """
        Data operations to perform on a single CPU.
        Load image data and labels from folders, do preprocessing, etc.
        """

    def setup(self, stage: typing.Optional[str] = None) -> torch.utils.data.Dataset:
        """
        Data operations to perform on every GPU.
        Split data into training and test sets, etc.
        """
        # Combine Sentinel2 and Worldview datasets into one!
        self.dataset: torch.utils.data.Dataset = S2S2Dataset()

        # Training/Validation split (80%/20%)
        train_length: int = int(len(self.dataset) * 0.8)
        val_length: int = len(self.dataset) - train_length
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            dataset=self.dataset, lengths=(train_length, val_length)
        )

        return self.dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the training loop.
        Set the training batch size here too.
        """
        return torch.utils.data.DataLoader(
            dataset=self.dataset_train, batch_size=8, num_workers=4
        )

        # for batch in torch.utils.data.DataLoader(
        #     dataset=self.dataset_train, batch_size=8
        # ):
        #     break


# %%
# @track_emissions(project_name="s2s2net")
def cli_main():
    """
    Command line interface to run the S2S2Net model. Based on
    https://github.com/PyTorchLightning/deep-learning-project-template

    The script can be executed in the terminal using:

        python s2s2net/model.py

    This will 1) load the data, 2) initialize the model, 3) train the model

    More options can be found by using `python s2s2net/model.py --help`.
    Happy training!
    """
    # Set a seed to control for randomness
    pl.seed_everything(seed=42)

    # Load Data
    datamodule: pl.LightningDataModule = S2S2DataModule()

    # Initialize Model
    model: pl.LightningModule = S2S2Net()

    # Training
    # TODO contribute to pytorch lightning so that deterministic="warn" works
    # Only works in Pytorch 1.11 or 1.12 I think
    # https://github.com/facebookresearch/ReAgent/pull/582/files
    # torch.use_deterministic_algorithms(True, warn_only=True)
    trainer: pl.Trainer = pl.Trainer(
        # deterministic=True,
        gpus=1,
        max_epochs=3,
        precision=16,
    )

    trainer.fit(model=model, datamodule=datamodule)

    # Export Model
    trainer.save_checkpoint(filepath="s2s2net.ckpt")

    print("Done!")


# %%
if __name__ == "__main__":
    cli_main()
