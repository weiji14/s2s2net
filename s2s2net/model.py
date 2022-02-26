"""
The S2S2Net model architecture and data loading modules.

Code structure adapted from Pytorch Lightning project seed at
https://github.com/PyTorchLightning/deep-learning-project-template
"""

import os
import typing

import mmseg.models
import numpy as np
import pytorch_lightning as pl
import torch
import torchgeo.datasets
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
        Define layers of the Vision Tranformer Network.

        Using the Segformer MiT-B0 model coupled with UPerNet.
        Using Pytorch implementation from mmsegmentation library. Details at
        https://github.com/open-mmlab/mmsegmentation/tree/v0.21.1/configs/segformer

        |        Backbone        |            'Neck'           |    Head     |
        |------------------------|-----------------------------|-------------|
        |  MixVisionTransformer  |   SegFormer Head + Upsample |   UPerNet   |

        Reference:
        - Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P.
          (2021). SegFormer: Simple and Efficient Design for Semantic
          Segmentation with Transformers. ArXiv:2105.15203 [Cs].
          http://arxiv.org/abs/2105.15203
        """
        super().__init__()

        ## Input Module (Encoder/Backbone). Mix Vision Tranformer config from
        # https://github.com/open-mmlab/mmsegmentation/blob/v0.21.1/configs/_base_/models/segformer_mit-b0.py#L6-L20
        self.segformer_backbone = mmseg.models.backbones.MixVisionTransformer(
            in_channels=4,  # RGB+NIR
            embed_dims=32,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
        )

        ## Middle Module (Decoder). SegFormer's All-MLP Head config from
        # https://github.com/open-mmlab/mmsegmentation/blob/v0.21.1/configs/_base_/models/segformer_mit-b0.py#L21-L29
        self.segformer_head = mmseg.models.decode_heads.SegformerHead(
            in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=8,  # output eight channels
            # norm_cfg=dict(type="SyncBN", requires_grad=True), # for multi-GPU
            align_corners=False,
        )

        ## Upsampling layers (Neck). First one to get back original image size
        # Second upsample is to get a super-resolution result.
        self.upsample_1 = torch.nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )
        self.upsample_2 = torch.nn.Upsample(
            scale_factor=5, mode="bilinear", align_corners=False
        )

        ## Output Module (Head). UPerHead config adapted from
        # https://github.com/open-mmlab/mmsegmentation/blob/a39f5856ce514ca09a161119041fb8490354c18f/configs/_base_/models/upernet_swin.py#L27-L38
        self.uper_head = mmseg.models.decode_heads.UPerHead(
            in_channels=[8, 8],
            in_index=[1, 0],
            pool_scales=[1, 2],
            channels=1,  # one channel for final output
            num_classes=1,
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Inference/Prediction).

        TODO
        """
        ## Step 1. Pass through SegFormer backbone (Mix Transformer x 4)
        # to get multi-level features F1, F2, F3, F4
        mit_output_tensors: typing.List[torch.Tensor] = self.segformer_backbone(
            x.float()
        )
        assert len(mit_output_tensors) == 4
        # f1, f2, f3, f4 = mit_output_tensors
        # print("f1.shape:", f1.shape)  # (8, 32, 128, 128)
        # print("f2.shape:", f2.shape)  # (8, 64, 64, 64)
        # print("f3.shape:", f3.shape)  # (8, 160, 32, 32)
        # print("f4.shape:", f4.shape)  # (8, 256, 16, 16)

        ## Step 2. Pass through SegFormer head (All-MLP Decoder)
        # to get a single tensor of size (N, C, H//4, W//4)
        segformer_output: torch.Tensor = self.segformer_head(mit_output_tensors)
        # print("segformer_output:", segformer_output.shape) # (8, 8, 128, 128)

        ## Step 3. Do a series of bilinear interpolation upsampling
        up1_output: torch.Tensor = self.upsample_1(segformer_output)
        up2_output: torch.Tensor = self.upsample_2(up1_output)
        # print("up1_output.shape:", up1_output.shape)  # (8, 8, 512, 512)
        # print("up2_output.shape:", up2_output.shape)  # (8, 8, 2560, 2560)

        ## Step 4. Pass into UPerNet (Pyramid Pooling Module)
        upernet_output: torch.Tensor = self.uper_head([up1_output, up2_output])
        # print("upernet_output:", uperhead_output.shape)  # (8, 1, 2560, 2560)

        return upernet_output

    def training_step(
        self, batch: typing.Dict[str, torch.Tensor], batch_idx: int
    ) -> dict:
        """
        Logic for the neural network's training loop.
        """
        x = batch["image"]
        y = batch["mask"]

        y_hat: torch.Tensor = self(x)

        # Calculate loss value to minimize
        loss: float = F.binary_cross_entropy_with_logits(input=y_hat, target=y)

        # Calculate metrics to determine how good results are
        metrics = mmseg.core.eval_metrics(
            results=y_hat.detach().cpu().numpy(),
            gt_seg_maps=y.detach().cpu().numpy(),
            num_classes=2,  # Not present and present
            ignore_index=255,  # Bad pixel value to ignore
            metrics=["mIoU", "mDice"],  # , "mFscore"
        )
        self.log_dict(
            dictionary={
                key: torch.as_tensor(np.mean(val)) for key, val in metrics.items()
            },
            prog_bar=True,
        )

        return {"loss": loss, **metrics}

    def configure_optimizers(self):
        """
        Optimizing function used to reduce the loss, so that the predicted
        mask gets as close as possible to the groundtruth mask.

        Using the AdamW optimizer with a learning rate of 0.00006. See:

        - Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay
          Regularization. ArXiv:1711.05101 [Cs, Math].
          http://arxiv.org/abs/1711.05101

        Documentation at:
        https://pytorch-lightning.readthedocs.io/en/1.5.10/common/optimizers.html
        """
        return torch.optim.AdamW(
            params=self.parameters(), lr=0.00006, weight_decay=0.01
        )


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
