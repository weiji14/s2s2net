"""
The S2S2Net model architecture and data loading modules.

Code structure adapted from Pytorch Lightning project seed at
https://github.com/PyTorchLightning/deep-learning-project-template
"""

import os
import typing

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
            in_channels=12, out_channels=42, kernel_size=[3, 3], padding=[1, 1]
        )
        # self.upsample = torch.nn.Upsample(scale_factor=5, mode="nearest")
        self.conv_out = torch.nn.Conv2d(
            in_channels=42, out_channels=1, kernel_size=[1, 1], padding=[0, 0]
        )

    def forward(self, x: torch.Tensor) -> typing.Tuple:
        """
        Forward pass (Inference/Prediction).

        TODO
        """
        _x = self.conv_in(x)
        # _x = self.upsample(_x)
        _x = self.conv_out(_x)

        return _x

    def training_step(
        self,
        batch: typing.Tuple[typing.List[torch.Tensor], typing.List[typing.Dict]],
        batch_idx: int,
    ) -> dict:
        """
        Logic for the neural network's training loop.
        """
        image = batch["image"]
        x = image[:, :-1, :, :]
        y = image[:, -1:, :, :]
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
class Sentinel2_L2A(torchgeo.datasets.Sentinel2):
    filename_glob = "S*_*_B02.tif"
    filename_regex = """
        ^S2[AB]_MSIL2A
        _(?P<date>\d{8}T\d{6})
        _N\d{4}
        _R\d{3}
        _T(?P<tile>\d{2}[A-Z]{3})
        _(?P<date1>\d{8}T\d{6})
        _(?P<band>B[018][\dA])\.tif$
    """

    def __init__(
        self,
        root: str = "data",
        crs: typing.Optional[rasterio.crs.CRS] = None,
        res: typing.Optional[float] = None,
        bands: typing.Sequence[str] = [],
        transforms: typing.Optional[
            typing.Callable[
                [typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]
            ]
        ] = None,
        cache: bool = True,
    ) -> None:
        super().__init__(root, crs, res, bands, transforms, cache)

        # self.filename_regex = self.filename_regex.replace(
        #     "T(?P<tile>\d{2}[A-Z]{3})", "T?????"
        # )

        self.bands = bands if bands else self.all_bands
        # self.crs = crs if crs else None


class WorldviewMask(torchgeo.datasets.RasterDataset):
    filename_glob = "GE01_20200518213247_105001001D498400_*_mask_pp.tif"


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

        def del_bbox_transform(sample: dict):
            """
            Workaround for FrozenInstanceError: cannot assign to field 'minx'
            error. Basically to handle the (frozen=True) in
            torchgeo.datasets.utils.BoundingBox dataclass.

            References:
            - https://github.com/microsoft/torchgeo/pull/329#discussion_r775576118
            - https://github.com/microsoft/torchgeo/pull/329/commits/4e2f05552073ec7f747ba3fabb7042d591323418
            """
            del sample["bbox"]
            return sample

        # Pre-load Sentinel-2 data
        self.s2_dataset = Sentinel2_L2A(
            root="by_date/sentinel2/17/downloads",
            # crs=rasterio.crs.CRS.from_epsg(32606),
            bands=[
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                # "B10",
                "B11",
                "B12",
            ],
            transforms=del_bbox_transform,
        )

        ## Pre-load Worldview mask data
        self.wv_dataset = WorldviewMask(root="Nov_2021/", transforms=del_bbox_transform)
        # Reproject Worldview image from EPSG:3413 to EPSG:32606, because
        # torchgeo doesn't do it properly with IntersectionDataset
        self.wv_dataset.crs = rasterio.crs.CRS.from_epsg(32606)

        assert self.s2_dataset.crs == self.wv_dataset.crs

    def setup(self, stage: typing.Optional[str] = None) -> torch.utils.data.Dataset:
        """
        Data operations to perform on every GPU.
        Split data into training and test sets, etc.
        """
        # Combine Sentinel2 and Worldview datasets into one!
        self.dataset = torchgeo.datasets.IntersectionDataset(
            dataset1=self.s2_dataset, dataset2=self.wv_dataset
        )
        return self.dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the training loop.
        Set the training batch size here too.
        """
        batch_sampler = torchgeo.samplers.RandomBatchGeoSampler(
            dataset=self.dataset,
            size=5120,  # width/height in metres
            batch_size=32,  # mini-batch size
            length=256,  # no. of samples per epoch
        )
        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_sampler=batch_sampler,
            collate_fn=torchgeo.datasets.stack_samples,
        )

        # for batch in torch.utils.data.DataLoader(
        #     dataset=self.dataset,
        #     batch_sampler=sampler,
        #     collate_fn=torchgeo.datasets.stack_samples,
        # ):
        #     break
        # image = batch["image"]


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
