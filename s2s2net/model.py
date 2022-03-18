"""
The S2S2Net model architecture and data loading modules.

Code structure adapted from Pytorch Lightning project seed at
https://github.com/PyTorchLightning/deep-learning-project-template
"""

import glob
import os
import typing

import mmseg.models
import numpy as np
import pytorch_lightning as pl
import rioxarray
import torch
import torchgeo.datasets
import torchmetrics
import torchvision.ops
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

        Using the Segformer MiT-B0 model coupled with upsampling layers.
        Using Pytorch implementation from mmsegmentation library. Details at
        https://github.com/open-mmlab/mmsegmentation/tree/v0.21.1/configs/segformer

        |        Backbone        |       'Neck'       |          Head         |
        |------------------------|--------------- ----|-----------------------|
        |  MixVisionTransformer  |   SegFormer Head   |   Upsample + Conv2D   |

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
            num_classes=16,  # output sixteen channels
            # norm_cfg=dict(type="SyncBN", requires_grad=True), # for multi-GPU
            align_corners=False,
        )

        ## Upsampling layers (Output). 1st one to get back original image size
        # 2nd upsample is to get a super-resolution result. Each of the two
        # upsample layers are followed by a Convolutional 2D layer.
        self.segmmask_upsample_0 = torch.nn.Upsample(scale_factor=4, mode="nearest")
        self.segmmask_post_upsample_conv_layer_0 = torch.nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.segmmask_upsample_1 = torch.nn.Upsample(scale_factor=5, mode="nearest")
        self.segmmask_post_upsample_conv_layer_1 = torch.nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1
        )

        self.superres_upsample_0 = torch.nn.Upsample(scale_factor=4, mode="nearest")
        self.superres_post_upsample_conv_layer_0 = torch.nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.superres_upsample_1 = torch.nn.Upsample(scale_factor=5, mode="nearest")
        self.superres_post_upsample_conv_layer_1 = torch.nn.Conv2d(
            in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1
        )

        # Evaluation metrics to know how good the segmentation results are
        self.iou = torchmetrics.JaccardIndex(num_classes=2)
        self.f1_score = torchmetrics.F1Score(num_classes=1)

    def forward(self, x: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
        """
        Forward pass (Inference/Prediction).

        TODO
        """
        ## Step 1. Pass through SegFormer backbone (Mix Transformer x 4)
        # to get multi-level features F1, F2, F3, F4
        # x: torch.Tensor = torch.randn(8, 4, 512, 512)
        mit_output_tensors: typing.List[torch.Tensor] = self.segformer_backbone(x)
        assert len(mit_output_tensors) == 4
        # f1, f2, f3, f4 = mit_output_tensors
        # print("f1.shape:", f1.shape)  # (8, 32, 128, 128)
        # print("f2.shape:", f2.shape)  # (8, 64, 64, 64)
        # print("f3.shape:", f3.shape)  # (8, 160, 32, 32)
        # print("f4.shape:", f4.shape)  # (8, 256, 16, 16)

        ## Step 2. Pass through SegFormer head (All-MLP Decoder)
        # to get a single tensor of size (N, C, H//4, W//4)
        segformer_output: torch.Tensor = self.segformer_head(mit_output_tensors)
        # print("segformer_output:", segformer_output.shape) # (8, 16, 128, 128)

        ## Step 3. Do a series of bilinear interpolation upsampling + Conv2d
        # Step 3a. Semantic Segmentation Super-Resolution (SSSR)
        segmmask_up_output_0: torch.Tensor = self.segmmask_upsample_0(segformer_output)
        segmmask_conv_output_0: torch.Tensor = self.segmmask_post_upsample_conv_layer_0(
            segmmask_up_output_0
        )
        # print("segmmask_conv_output_0.shape:", segmmask_conv_output_0.shape)  # (8, 8, 512, 512)
        segmmask_up_output_1: torch.Tensor = self.segmmask_upsample_1(
            segmmask_conv_output_0
        )
        segmmask_conv_output_1: torch.Tensor = self.segmmask_post_upsample_conv_layer_1(
            segmmask_up_output_1
        )
        # print("segmmask_conv_output_1.shape:", segmmask_conv_output_1.shape)  # (8, 1, 2560, 2560)

        # Step 3b. Single Image Super-Resolution (SISR)
        superres_up_output_0: torch.Tensor = self.superres_upsample_0(segformer_output)
        superres_conv_output_0: torch.Tensor = self.superres_post_upsample_conv_layer_0(
            superres_up_output_0
        )
        # print("superres_conv_output_0.shape:", superres_conv_output_0.shape)  # (8, 8, 512, 512)
        superres_up_output_1: torch.Tensor = self.superres_upsample_1(
            superres_conv_output_0
        )
        superres_conv_output_1: torch.Tensor = self.superres_post_upsample_conv_layer_1(
            superres_up_output_1
        )
        # print("superres_conv_output_1.shape:", superres_conv_output_1.shape)  # (8, 1, 2560, 2560)

        return {
            "segmmask_conv_output_0": segmmask_conv_output_0,  # for FA loss
            "segmmask_conv_output_1": segmmask_conv_output_1,  # segmentation output
            "superres_conv_output_0": superres_conv_output_0,  # for FA loss
            "superres_conv_output_1": superres_conv_output_1,  # super-resolution output
        }

    def evaluate(
        self, batch: typing.Dict[str, torch.Tensor]
    ) -> typing.Dict[str, torch.Tensor]:
        """
        Compute the loss for a single batch in the training or validation step.

        For each batch:
        1. Get the image and corresponding groundtruth label from each batch
        2. Pass the image through the neural network to get a predicted label
        3. Calculate the loss between the predicted label and groundtruth label

        Returns
        -------
        loss_and_metrics : dict
            A dictionary containing the total loss, 3 of the unweighted loss
            values making up the total loss, and 2 metrics. These include:

            - loss (Total loss)
            - loss_feataffy (Feature Affinity loss)
            - loss_segmmask (Segmentation loss)
            - loss_superres (Super-Resolution loss)
            - iou (Intersection over Union)
            - f1 (F1 score)
        """
        x: torch.Tensor = batch["image"].float()  # Input Sentinel-2 image
        y: torch.Tensor = batch["mask"]  # Groundtruth binary mask
        y_highres: torch.Tensor = batch["hres"]  # High resolution image
        # y = torch.randn(8, 1, 2560, 2560)
        # y_highres = torch.randn(8, 4, 2560, 2560)

        y_hat: typing.Dict[str, torch.Tensor] = self(x)

        ## Calculate loss values to minimize
        def similarity_matrix(f):
            # f expected shape (Bs, C', H', W')
            # before computing the relationship of every pair of pixels,
            # subsample the feature map to its 1/8
            f = F.interpolate(
                f, size=(f.shape[2] // 8, f.shape[3] // 8), mode="nearest"
            )
            f = f.permute((0, 2, 3, 1))
            f = torch.reshape(f, (f.shape[0], -1, f.shape[3]))  # shape (Bs, H'xW', C')
            f_n = torch.linalg.norm(f, ord=None, dim=2).unsqueeze(
                -1
            )  # ord=None indicates 2-Norm,
            # unsqueeze last dimension to broadcast later
            eps = 1e-8
            f_norm = f / torch.max(f_n, eps * torch.ones_like(f_n))
            sim_mt = f_norm @ f_norm.transpose(2, 1)
            return sim_mt

        # 1: Feature Affinity loss calculation
        _segmmask_sim_matrix: torch.Tensor = similarity_matrix(
            f=y_hat["segmmask_conv_output_0"]
        )
        _superres_sim_matrix: torch.Tensor = similarity_matrix(
            f=y_hat["superres_conv_output_0"]
        )
        _n_elements: int = (
            _segmmask_sim_matrix.shape[-2] * _segmmask_sim_matrix.shape[-1]
        )
        _abs_dist: torch.Tensor = torch.abs(_segmmask_sim_matrix - _superres_sim_matrix)
        feature_affinity_loss: torch.Tensor = torch.mean(
            (1 / _n_elements) * torch.sum(input=_abs_dist, dim=[-2, -1])
        )

        # 2: Semantic Segmentation loss (Focal Loss)
        segmmask_loss: torch.Tensor = torchvision.ops.sigmoid_focal_loss(
            inputs=y_hat["segmmask_conv_output_1"],
            targets=y,
            alpha=0.75,
            gamma=2,
            reduction="mean",
        )
        # 3: Super-Resolution loss (Mean Absolute Error)
        superres_loss: torch.Tensor = torchmetrics.functional.mean_absolute_error(
            preds=y_hat["superres_conv_output_1"],
            target=y_highres.to(dtype=torch.float16),
        )

        # 1 + 2 + 3: Calculate total loss and log to console
        total_loss: torch.Tensor = (
            (1.0 * feature_affinity_loss) + segmmask_loss + (0.001 * superres_loss)
        )
        losses: typing.Dict[str, torch.Tensor] = {
            # Component losses (Feature Affinity, Segmentation, Super-Resolution)
            "loss_feataffy": feature_affinity_loss.detach(),
            "loss_segmmask": segmmask_loss.detach(),
            "loss_superres": superres_loss.detach(),
        }

        # Calculate metrics to determine how good results are
        iou_score: torch.Tensor = self.iou(  # Intersection over Union
            preds=y_hat["segmmask_conv_output_1"].squeeze(),
            target=(y > 0.5).squeeze().to(dtype=torch.int8),  # binarize
        )
        f1_score: torch.Tensor = self.f1_score(  # F1 Score
            preds=y_hat["segmmask_conv_output_1"].ravel(),
            target=(y > 0.5).ravel().to(dtype=torch.int8),  # binarize
        )
        metrics: typing.Dict[str, torch.Tensor] = {"iou": iou_score, "f1": f1_score}

        return {"loss": total_loss, **losses, **metrics}

    def training_step(
        self, batch: typing.Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Logic for the neural network's training loop.
        """
        losses_and_metrics: dict = self.evaluate(batch=batch)

        self.log_dict(dictionary=losses_and_metrics, prog_bar=True)

        # Log training loss and metrics to Tensorboard
        if self.logger is not None and hasattr(self.logger.experiment, "add_scalars"):
            for metric_name, metric_value in losses_and_metrics.items():
                self.logger.experiment.add_scalars(
                    main_tag=metric_name,
                    tag_scalar_dict={"train": metric_value},
                    global_step=self.global_step,
                    # epoch=self.current_epoch,
                )

        return losses_and_metrics["loss"]

    def validation_step(
        self,
        batch: typing.Tuple[typing.List[torch.Tensor], typing.List[typing.Dict]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Logic for the neural network's validation loop.
        """
        val_losses_and_metrics: dict = self.evaluate(batch=batch)

        self.log_dict(
            dictionary={
                f"val_{key}": value for key, value in val_losses_and_metrics.items()
            },
            prog_bar=True,
        )

        # Log validation loss and metrics to Tensorboard
        if self.logger is not None and hasattr(self.logger.experiment, "add_scalars"):
            for metric_name, metric_value in val_losses_and_metrics.items():
                self.logger.experiment.add_scalars(
                    main_tag=metric_name,
                    tag_scalar_dict={"validation": metric_value},
                    global_step=self.global_step,
                    # epoch=self.current_epoch,
                )

        return val_losses_and_metrics["loss"]

    def predict_step(
        self,
        batch: typing.Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: typing.Optional[int] = None,
    ):
        """
        Logic for the neural network's prediction loop.
        """
        x: torch.Tensor = batch["image"].float()  # Input Sentinel-2 image

        y_hat: torch.Tensor = self(x)

        return (
            torch.sigmoid(input=y_hat["segmmask_conv_output_1"]),
            y_hat["superres_conv_output_1"],
        )

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
        root: str = "SuperResolution/chips/npy",  # Train/Validation chips
        # root: str = "SuperResolution/aligned",  # Inference on actual images
        train: bool = True,  # Whether to load training set or predict only
        transforms: typing.Optional[
            typing.Callable[
                [typing.Dict[str, torch.Tensor]], typing.Dict[str, torch.Tensor]
            ]
        ] = None,
    ):
        self.root: str = root
        self.train: bool = train
        self.transforms = transforms

        img_path: str = (
            os.path.join(self.root, "image") if self.train else os.path.join(self.root)
        )
        self.ids: list = [int(id) for id, _ in enumerate(os.listdir(path=img_path))]

    def __getitem__(self, index: int = 0) -> typing.Dict[str, torch.Tensor]:
        """
        Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index
        """
        if self.train:
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

        else:
            filename: str = glob.glob(
                os.path.join(self.root, f"{index:04d}", "S2*.tif")
            )[0]
            with rioxarray.open_rasterio(filename=filename) as rds:
                assert rds.ndim == 3  # Channel, Height, Width
                assert rds.shape[0] == 4  # 4 bands/channels (RGB+NIR)
                sample: dict = {
                    "image": torch.as_tensor(data=rds.data.astype(np.int16))
                }

        if self.transforms is not None:
            sample: typing.Dict[str, torch.Tensor] = self.transforms(sample)

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
        if stage == "fit" or stage is None:  # Training/Validation on chips
            # Combine Sentinel2 and Worldview datasets into one!
            self.dataset: torch.utils.data.Dataset = S2S2Dataset(
                root="SuperResolution/chips/npy", train=True
            )

            # Training/Validation split (80%/20%)
            train_length: int = int(len(self.dataset) * 0.8)
            val_length: int = len(self.dataset) - train_length
            self.dataset_train, self.dataset_val = torch.utils.data.random_split(
                dataset=self.dataset, lengths=(train_length, val_length)
            )

        elif stage == "predict":  # Inference on actual images
            self.dataset: torch.utils.data.Dataset = S2S2Dataset(
                root="SuperResolution/aligned", train=False
            )

        return self.dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the training loop.
        Set the training batch size here too.
        """
        return torch.utils.data.DataLoader(
            dataset=self.dataset_train, batch_size=32, num_workers=4
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the validation loop.
        Set the validation batch size here too.
        """
        # TODO use an independent validation set from different geographic region
        return torch.utils.data.DataLoader(
            dataset=self.dataset_val, batch_size=32, num_workers=4
        )
        # for batch in torch.utils.data.DataLoader(
        #     dataset=self.dataset_val, batch_size=8
        # ):
        #     break

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the prediction loop.
        Set the prediction batch size here too.
        """
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_size=1)

        # for batch in torch.utils.data.DataLoader(dataset=self.dataset, batch_size=1):
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

    # Setup Tensorboard logger
    tensorboard_logger: pl.loggers.LightningLoggerBase = pl.loggers.TensorBoardLogger(
        save_dir="tb_logs", name="s2s2net"
    )

    # Training
    # TODO contribute to pytorch lightning so that deterministic="warn" works
    # Only works in Pytorch 1.11 or 1.12 I think
    # https://github.com/facebookresearch/ReAgent/pull/582/files
    # torch.use_deterministic_algorithms(True, warn_only=True)
    trainer: pl.Trainer = pl.Trainer(
        # deterministic=True,
        gpus=2,
        strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
        logger=tensorboard_logger,
        max_epochs=27,
        precision=16,
    )

    trainer.fit(model=model, datamodule=datamodule)

    # Export Model
    trainer.save_checkpoint(filepath="s2s2net.ckpt")

    print("Done!")


# %%
if __name__ == "__main__":
    cli_main()
