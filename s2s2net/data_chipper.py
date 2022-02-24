# %%
import glob
import os

import numpy as np
import rioxarray
import tqdm

# %%
# Create subdirectories to put the data in
# os.makedirs("SuperResolution/chips/geotiff/image", exist_ok=True)
# os.makedirs("SuperResolution/chips/geotiff/mask", exist_ok=True)
os.makedirs("SuperResolution/chips/npy/image", exist_ok=True)
os.makedirs("SuperResolution/chips/npy/mask", exist_ok=True)
os.makedirs("SuperResolution/chips/npy/hres", exist_ok=True)

# %%
# Main loop to create the image chips that will become the training dataset
j: int = 0
for folder in tqdm.tqdm(sorted(os.listdir("SuperResolution/"))[:11]):
    if folder == "chips":
        continue
    sen2_file = glob.glob(f"SuperResolution/{folder}/S2*.tif")[0]
    mask_file, hres_file = sorted(
        glob.glob(f"SuperResolution/{folder}/*_reprojected.tif")
    )

    with (
        rioxarray.open_rasterio(filename=sen2_file) as ds_sen2,
        rioxarray.open_rasterio(filename=mask_file) as ds_mask,
        rioxarray.open_rasterio(filename=hres_file) as ds_hres,
    ):
        for x in range(int(ds_sen2.x.min()), int(ds_sen2.x.max()) - 5120, 5120):
            for y in range(int(ds_sen2.y.min()), int(ds_sen2.y.max()) - 5120, 5120):

                crop_ds_sen2 = ds_sen2.rio.clip_box(
                    minx=x, miny=y, maxx=x + 5120 - 10, maxy=y + 5120 - 10
                )
                if crop_ds_sen2.shape == (4, 512, 512):  # full size tiles only
                    crop_ds_mask = ds_mask.rio.clip_box(
                        minx=x, miny=y, maxx=x + 5120 - 10, maxy=y + 5120 - 10
                    )
                    assert crop_ds_mask.shape == (1, 2556, 2556)

                    crop_ds_hres = ds_hres.rio.clip_box(
                        minx=x, miny=y, maxx=x + 5120 - 10, maxy=y + 5120 - 10
                    )
                    assert crop_ds_hres.shape == (4, 2556, 2556)

                    # Don't save chips with NaN values, or empty masks
                    if crop_ds_sen2.min().isnull() or crop_ds_mask.max() == 0:
                        continue

                    # Save as npy file format
                    np.save(
                        file=f"SuperResolution/chips/npy/image/SEN2_{j:04d}.npy",
                        arr=crop_ds_sen2,
                    )
                    np.save(
                        file=f"SuperResolution/chips/npy/mask/MASK_{j:04d}.npy",
                        arr=crop_ds_mask,
                    )
                    np.save(
                        file=f"SuperResolution/chips/npy/hres/HRES_{j:04d}.npy",
                        arr=crop_ds_hres,
                    )

                    # Save as geotiff file format
                    # crop_ds_sen2.rio.to_raster(
                    #     raster_path=f"SuperResolution/chips/geotiff/image/SEN2_{j:04d}.tif"
                    # )
                    # crop_ds_mask.rio.to_raster(
                    #     raster_path=f"SuperResolution/chips/geotiff/mask/MASK_{j:04d}.tif"
                    # )
                    # crop_ds_mask.rio.to_raster(
                    #     raster_path=f"SuperResolution/chips/geotiff/hres/HRES_{j:04d}.tif"
                    # )

                    j += 1
