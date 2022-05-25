"""
The S2S2Net model architecture and data loading modules.

Code structure adapted from Pytorch Lightning project seed at
https://github.com/PyTorchLightning/deep-learning-project-template
"""

import glob
import os
import typing

try:
    import cucim
    import cupy
except ImportError:
    pass

import mmseg.models
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.utilities.deepspeed
import rasterio
import rioxarray
import skimage.exposure
import torch
import torchgeo.datasets
import torchmetrics
import torchvision.ops
import tqdm
import xarray as xr
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
            in_channels=6,  # RGB+NIR+SWIR
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
        self, batch: typing.Dict[str, torch.Tensor], calc_loss: bool = True
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
        dtype = torch.float16 if self.precision == 16 else torch.float
        x: torch.Tensor = batch["image"].to(dtype=dtype)  # Input Sentinel-2 image
        y: torch.Tensor = batch["mask"]  # Groundtruth binary mask
        if calc_loss:
            y_highres: torch.Tensor = batch["hres"]  # High resolution image
        # y = torch.randn(8, 1, 2560, 2560)
        # y_highres = torch.randn(8, 4, 2560, 2560)

        y_hat: typing.Dict[str, torch.Tensor] = self(x)

        ## Calculate loss values to minimize
        if calc_loss:  # only on training and val step

            def similarity_matrix(f):
                # f expected shape (Bs, C', H', W')
                # before computing the relationship of every pair of pixels,
                # subsample the feature map to its 1/8
                f = F.interpolate(
                    f, size=(f.shape[2] // 8, f.shape[3] // 8), mode="nearest"
                )
                f = f.permute((0, 2, 3, 1))
                f = torch.reshape(
                    f, (f.shape[0], -1, f.shape[3])
                )  # shape (Bs, H'xW', C')
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
            _abs_dist: torch.Tensor = torch.abs(
                _segmmask_sim_matrix - _superres_sim_matrix
            )
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
                # Total loss
                "loss": total_loss,
                # Component losses (Feature Affinity, Segmentation, Super-Resolution)
                "loss_feataffy": feature_affinity_loss.detach(),
                "loss_segmmask": segmmask_loss.detach(),
                "loss_superres": superres_loss.detach(),
            }
        else:  # if calc_loss is False, i.e. only on test step
            losses: dict = {}

        # Calculate metrics to determine how good results are
        preds = y_hat["segmmask_conv_output_1"]
        target = (y > 0.5).to(dtype=torch.int8)  # binarize
        if preds.shape != target.shape:  # resize prediction to target shape
            preds = F.interpolate(input=preds, size=target.shape[-2:], mode="bilinear")
            # print(x.shape, preds.shape, target.shape)

        iou_score: torch.Tensor = self.iou(  # Intersection over Union
            preds=preds.squeeze(), target=target.squeeze()
        )
        f1_score: torch.Tensor = self.f1_score(  # F1 Score
            preds=preds.ravel(), target=target.ravel()
        )
        metrics: typing.Dict[str, torch.Tensor] = {"iou": iou_score, "f1": f1_score}

        return {**losses, **metrics}

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
        self, batch: typing.Dict[str, torch.Tensor], batch_idx: int
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
        batch: typing.Dict[str, typing.Any],
        batch_idx: int,
        dataloader_idx: typing.Optional[int] = None,
    ) -> typing.Union[  # output depends on whether input has CRS info
        typing.Tuple[torch.Tensor, torch.Tensor],  # without CRS
        typing.List[typing.Tuple[xr.DataArray, xr.DataArray]],  # with CRS
    ]:
        """
        Logic for the neural network's prediction loop.
        """
        dtype = torch.float16 if self.precision == 16 else torch.float
        x: torch.Tensor = batch["image"].to(dtype=dtype)  # Input Sentinel-2 image

        # Pass the image through neural network model to get predicted images
        y_hat: typing.Dict[str, torch.Tensor] = self(x)
        segmmask: torch.Tensor = torch.sigmoid(input=y_hat["segmmask_conv_output_1"])
        superres: torch.Tensor = y_hat["superres_conv_output_1"]
        _, bands, height, width = segmmask.shape

        try:
            # Coordintate Reference System of input image
            crses: typing.List[rasterio.crs.CRS] = batch["crs"]
            # Bounding box extent of input image
            extents: typing.List[rasterio.coords.BoundingBox] = batch["bbox"]
        except KeyError:  # If input batch sample has no CRS, yield pred images
            return (segmmask, superres)

        results: list = []
        for idx in tqdm.trange(0, len(x)):
            crs: rasterio.crs.CRS = crses[idx]
            extent: rasterio.coords.BoundingBox = extents[idx]

            # Georeference segmentation result using rioxarray
            geo_segmmask = xr.DataArray(
                data=segmmask.squeeze().cpu(),  # TODO don't move to CPU
                coords=dict(
                    y=np.linspace(start=extent.top, stop=extent.bottom, num=height),
                    x=np.linspace(start=extent.left, stop=extent.right, num=width),
                ),
                dims=("y", "x"),
            )
            _ = geo_segmmask.rio.set_crs(input_crs=crs)

            # Histogram Equalize and Georeference super-resolution result
            # Note: Use cucim if on GPU, use skimage if on CPU (or GPU out of memory)
            _superres: cupy.ndarray = 2**8 * cucim.skimage.exposure.equalize_hist(
                image=cupy.asanyarray(a=superres.squeeze())
            )
            # _superres: np.ndarray = 2**8 * skimage.exposure.equalize_hist(
            #     image=superres.squeeze().cpu().numpy()
            # )
            geo_superres = xr.DataArray(
                data=_superres.get(),
                coords=dict(
                    band=[8, 4, 3, 2],
                    y=np.linspace(start=extent.top, stop=extent.bottom, num=height),
                    x=np.linspace(start=extent.left, stop=extent.right, num=width),
                ),
                dims=("band", "y", "x"),
            )
            _ = geo_superres.rio.set_crs(input_crs=crs)

            results.append((geo_segmmask, geo_superres))

        return results

    def test_step(
        self, batch: typing.Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Logic for the neural network's test loop.
        """
        test_metrics: dict = self.evaluate(batch=batch, calc_loss=False)

        self.log_dict(
            dictionary={f"test_{key}": value for key, value in test_metrics.items()},
            prog_bar=True,
        )

        # Log test metrics to Tensorboard
        if self.logger is not None and hasattr(self.logger.experiment, "add_scalars"):
            for metric_name, metric_value in test_metrics.items():
                self.logger.experiment.add_scalars(
                    main_tag=metric_name,
                    tag_scalar_dict={"test": metric_value},
                    global_step=self.global_step,
                    # epoch=self.current_epoch,
                )

        return test_metrics["f1"]

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
    1. image - Sentinel-2 RGB-NIR-SWIR image at 10m resolution (6, 512, 512)
    2. mask - Binary segmentation mask at 2m resolution (1, 2560, 2560)
    3. hres - High resolution RGB-NIR image at 2m resolution (4, 2560, 2560)

    Parameters
    ----------
    root : str
        Root directory of the satellite datasets.
    image_set : str (optional)
        Select the image_set to use, either:

        - 'trainval' - Training/Validation set loaded from npy files
        - 'predict' - Predict set, load only the Sentinel-2 GeoTIFF
        - 'test' - Test set, load both Sentinel-2 and binary mask GeoTIFFs
    transforms : func (optional)
        A function/transform that takes input sample and its target as entry
        and returns a transformed version.
    ids : list[str] (optional)
        A list of folder ids like ['0123', '0124'] to run inference on
        (predict/test), ignored if `image_set=='trainval'`.
    """

    def __init__(
        self,
        root: str = "SuperResolution/chips/npy",  # Train/Validation chips
        # root: str = "SuperResolution/aligned",  # Inference on actual images
        image_set: str = "trainval",  # Whether to load train/val, predict or test set
        transforms: typing.Optional[
            typing.Callable[
                [typing.Dict[str, torch.Tensor]], typing.Dict[str, torch.Tensor]
            ]
        ] = None,
        ids: typing.Optional[typing.List[str]] = None,
    ):
        self.root: str = root
        self.image_set: bool = image_set
        self.transforms = transforms

        img_path: str = (
            os.path.join(self.root, "image")
            if self.image_set == "trainval"
            else self.root
        )
        self.ids: list[str] = ids or [path for path in os.listdir(path=img_path)]

    def __getitem__(
        self, index: int = 0
    ) -> typing.Dict[
        str, typing.Union[torch.Tensor, rasterio.crs.CRS, rasterio.coords.BoundingBox]
    ]:
        """
        Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index
        """
        if self.image_set == "trainval":
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

        elif self.image_set in ["predict", "test"]:
            idx: str = self.ids[index]  # e.g. 0123
            image_filename: str = glob.glob(os.path.join(self.root, idx, "S2*.tif"))[0]
            with rioxarray.open_rasterio(filename=image_filename) as rds_image:
                assert rds_image.ndim == 3  # Channel, Height, Width
                assert rds_image.shape[0] == 6  # 6 bands/channels (RGB+NIR+SWIR)
                image = rds_image
                sample: dict = {"crs": image.rio.crs}

            # For test dataloader, also need to get mask to compute metrics
            if self.image_set == "test":
                mask_filename: str = glob.glob(
                    os.path.join(self.root, idx, "*_mask_*.tif")
                )[0]
                with rioxarray.open_rasterio(filename=mask_filename) as rds_mask:
                    assert rds_mask.ndim == 3  # Channel, Height, Width
                    assert rds_mask.shape[0] == 1  # 1 band/channel

                    # Clip to bounding box extent of mask with non-NaN values
                    # Need to use low-res (10m) extent instead of 2m extent
                    mask_extent = (
                        rds_mask.rio.reproject(
                            dst_crs=image.rio.crs, resolution=image.rio.resolution()
                        )
                        .where(
                            cond=~rds_mask.isnull(),  # keep non-NaN areas
                            # cond=rds_mask == 1,# keep with-valid-pixel areas
                            drop=True,
                        )
                        .rio.bounds()
                    )
                    sample["mask"] = torch.as_tensor(
                        data=rds_mask.rio.clip_box(*mask_extent).data  # float32
                    )

                # Clip image to match geographical extent of binary mask
                assert rds_mask.rio.crs == rds_image.rio.crs
                image = image.rio.clip_box(*mask_extent)

            left, bottom, right, top = image.rio.bounds()
            sample["bbox"] = rasterio.coords.BoundingBox(
                left=left, right=right, bottom=bottom, top=top
            )
            sample["image"] = torch.as_tensor(
                data=image.data.astype(np.int16)  # uint16 to int16
            )
            # assert sample["mask"].shape[1] == sample["image"].shape[1] * 5
            # assert sample["mask"].shape[2] == sample["image"].shape[2] * 5
        else:
            raise ValueError(
                f"Unknown image_set: {self.image_set}, "
                "should be either 'trainval', 'predict' or 'test'"
            )

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

    def __init__(self, ids: typing.Optional[typing.List[str]] = None):
        """
        Parameters
        ----------
        ids : list[str] (optional)
            A list of folder ids like ['0123', '0124'] to run inference on
            (predict/test), ignored during model fit (train/val) stage.
        """
        super().__init__()
        self.ids: list[str] = ids

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
                root="SuperResolution/chips/npy", image_set="trainval"
            )

            # Training/Validation split (80%/20%)
            train_length: int = int(len(self.dataset) * 0.8)
            val_length: int = len(self.dataset) - train_length
            self.dataset_train, self.dataset_val = torch.utils.data.random_split(
                dataset=self.dataset, lengths=(train_length, val_length)
            )

        elif stage == "predict":  # Inference on actual images
            self.dataset: torch.utils.data.Dataset = S2S2Dataset(
                root="SuperResolution/aligned", image_set=stage, ids=self.ids
            )
        elif stage == "test":  # Inference on test images
            self.dataset: torch.utils.data.Dataset = S2S2Dataset(
                root="SuperResolution/aligned",
                image_set=stage,
                ids=["0123", "0124", "0125", "0126", "0211", "0223", "0157", "0439"]
                or self.ids,
            )
        else:
            raise ValueError(
                f"Unknown stage: {stage}, "
                "should be either 'fit', 'predict' or 'test'"
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
        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=4,
            collate_fn=torchgeo.datasets.stack_samples,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the test loop.
        Set the test batch size here too.
        """
        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=4,
            collate_fn=torchgeo.datasets.stack_samples,
        )

        # for batch in torch.utils.data.DataLoader(
        #     dataset=self.dataset,
        #     batch_size=1,
        #     num_workers=1,
        #     collate_fn=torchgeo.datasets.stack_samples,
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

    # Setup Tensorboard logger
    tensorboard_logger: pl.loggers.LightningLoggerBase = pl.loggers.TensorBoardLogger(
        save_dir="tb_logs", name="s2s2net"
    )
    # Setup automatic checkpointing of best model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{val_f1:.2f}-{step}", monitor="val_f1", mode="max"
    )

    # Training
    # TODO contribute to pytorch lightning so that deterministic="warn" works
    # Only works in Pytorch 1.11+
    # https://github.com/facebookresearch/ReAgent/pull/582/files
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    trainer: pl.Trainer = pl.Trainer(
        # deterministic=True,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        devices="auto",
        strategy="deepspeed_stage_2",
        logger=tensorboard_logger,
        max_epochs=52,
        precision=16,
    )
    trainer.fit(model=model, datamodule=datamodule)

    # Testing
    if trainer.num_devices > 1:
        trainer.test(model=model, datamodule=datamodule)

    # Export Model
    # Convert deepspeed checkpoint directory to single checkpoint file
    # https://pytorch-lightning.readthedocs.io/en/1.6.3/advanced/model_parallel.html#deepspeed-zero-stage-3-single-file
    # trainer.save_checkpoint(filepath="s2s2net_ckpt")
    if checkpoint_callback.best_model_path:
        print(f"Saving {checkpoint_callback.best_model_path} to s2s2net.ckpt")
        pl.utilities.deepspeed.convert_zero_checkpoint_to_fp32_state_dict(
            checkpoint_dir=checkpoint_callback.best_model_path,
            output_file="s2s2net.ckpt",
        )

    print("Done!")


# %%
if __name__ == "__main__":
    cli_main()
