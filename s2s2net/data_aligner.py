import glob
import os

import numpy as np
import pandas as pd
import pygmt
import rasterio
import rioxarray
import tqdm
import xarray as xr

# %%
# Load Sentinel2 to High Resolution imagery lookup table
df_raw: pd.DataFrame = pd.read_csv(
    filepath_or_buffer="coords_files.csv",
    names=[
        "s2_index",
        "x",
        "y",
        "longitude",
        "latitude",
        "hres_month",
        "hres_date",
        "hres_year",
        "hres_filename",
    ],
    skipinitialspace=True,
)
valid_s2_indexes = df_raw.s2_index.isin(
    [int(n) for n in os.listdir("by_date/sentinel2")]
)
df: pd.DataFrame = df_raw.loc[valid_s2_indexes]


# Create dictionary mapping 1 highres images to N lowres Sentinel-2 images
# by_date/sentinel2/{s2_index}/{dirname1}/{dirname1}_B01.tif
#                                                    B12.tif
#                             /{dirname2}/{dirname2}_B01.tif
#                                                    B12.tif
assert len(df) == len(df.hres_filename.unique())

hres_s2_dict: dict = {}  # dict with key: highres_filename, val: s2 path(s)
for _, row in df.iterrows():
    hres_filename: str = row.hres_filename
    s2_index: int = row.s2_index

    dirnames: list = [
        os.path.basename(p=p)
        for p in glob.glob(f"by_date/sentinel2/{s2_index}/mosaic/*")
    ]
    if len(dirnames) > 0:
        hres_s2_dict[hres_filename] = [
            f"by_date/sentinel2/{s2_index}/mosaic/{dirname}/{dirname}_B??.tif"
            for dirname in dirnames
        ]

    # Check that each directory has 12 files corresponding to 12 Sentinel bands
    # for dirname in dirnames:
    #     filenames: list = sorted(
    #         glob.glob(f"by_date/sentinel2/{s2_index}/mosaic/{dirname}/S2*.tif")
    #     )
    # assert len(filenames) == 12  # 12 Sentinel bands

assert len(hres_s2_dict) <= len(df)

# %%
# Ensure pixel coordinates are aligned between
# Sentinel 2 and reprojected High Resolution imagery
def align_lowres_highres_pair(
    img_highres: xr.DataArray, img_lowres: xr.DataArray
) -> (xr.DataArray, xr.DataArray):
    """
    Create low resolution and high resolution satellite image pairs.
    Does cropping and alignment of the pixel corners so that both images cover
    the exact same spatial extent.

    Assumes that the low resolution image covers a larger spatial extent than
    the high resolution image. Also hardcoded so that lower spatial resolution
    is 10 metres, and higher spatial resolution is 2 metres.
    """

    assert img_highres.rio.crs == img_lowres.rio.crs

    left, bottom, right, top = img_highres.rio.bounds()
    try:
        assert bottom < top
    except AssertionError:
        (bottom, top) = (top, bottom)
    # bounds = rasterio.coords.BoundingBox(left=left, bottom=bottom, right=right, top=top)
    # print(bounds)

    # Get highres mask image bounds, rounded nicely to 10m increments
    grid_info = pygmt.grdinfo(
        grid=img_highres.isel(band=-1),
        # ensure new bounds are within the origin bounds
        region=[left + 20, right - 20, bottom + 20, top - 20],
        spacing=10,
        per_column=True,
        verbose="e",
    )
    # Workaround until https://github.com/GenericMappingTools/pygmt/issues/593
    # is completed
    minx, maxx, miny, maxy = [int(num) for num in grid_info.split()[:4]]
    assert minx > left
    assert maxx < right

    try:
        assert miny > bottom
    except AssertionError:
        miny += 10
        assert miny > bottom

    try:
        assert maxy < top
    except AssertionError:
        maxy -= 10
        assert maxy < top

    # Calculate new coordinates to interpolate highres image onto
    res: int = 2
    offset: int = res / 2  # Shift highres coordinates by 1m (half a 2m pixel)
    new_y = np.linspace(
        start=maxy - offset, stop=miny - offset, num=int((maxy - miny) / 2) + 1
    )
    new_x = np.linspace(
        start=minx + offset, stop=maxx + offset, num=int((maxx - minx) / 2) + 1
    )

    # Interpolate highres image to exact rounded xy coordinates
    aligned_highres: xr.DataArray = img_highres.interp(
        method="linear", y=new_y, x=new_x
    ).astype(dtype=img_highres.dtype)
    assert aligned_highres.dtype == np.float32  # in (np.uint8, np.uint16)

    assert new_y[0] in aligned_highres.y
    assert new_x[0] in aligned_highres.x

    # Crop lowres image to spatial extent of highres image
    aligned_lowres: xr.DataArray = img_lowres.rio.clip_box(
        minx=minx, miny=miny, maxx=maxx, maxy=maxy
    )
    assert aligned_lowres.dtype == np.uint16

    return aligned_lowres, aligned_highres


# %%
# Main loop to do the image alignment and file saving in parallel.
j: int = 0
for maskname in tqdm.tqdm(hres_s2_dict.keys()):
    # Get High Resolution (Quickbird/Worldview) image mask
    # maskname: str = row.hres_filename
    hresname: str = (
        maskname.replace("_DL_", "")
        .replace("mask_pp.tif", ".tif")
        .replace("_mask.tif", ".tif")
        .replace("_.tif", ".tif")
    )
    with (
        rioxarray.open_rasterio(
            filename=os.path.join("Nov_2021", maskname), lock=False
        ) as highresmask,
        rioxarray.open_rasterio(
            filename=os.path.join("imagery", hresname), lock=False
        ) as highresimg,
    ):

        # Get corresponding Sentinel 2 10m resolution input image(s)
        pathnames: list = hres_s2_dict[maskname]
        for pathname in pathnames:
            filenames: list = glob.glob(pathname=pathname)

            objs: list = []
            for filename in sorted(filenames):
                if filename.endswith(("B08.tif", "B04.tif", "B03.tif", "B02.tif")):
                    band: str = filename[-6:-4]
                    da: xr.DataArray = rioxarray.open_rasterio(
                        filename=filename, lock=False
                    )
                    da["band"] = int(band) * da.band
                    objs.append(da)
            sentinel2img: xr.DataArray = xr.concat(objs=objs, dim="band")

            # Reproject HighRes image mask to match projection of Sentinel 2 image
            # https://corteva.github.io/rioxarray/latest/examples/reproject_match.html
            highresimg_reprojected: xr.DataArray = highresimg.rio.reproject(
                dst_crs=sentinel2img.rio.crs
            )
            _highresmask = highresmask.astype(np.float32)
            _highresmask.rio.set_nodata(input_nodata=np.nan, inplace=True)
            _highresmask_reprojected: xr.DataArray = _highresmask.rio.reproject(
                dst_crs=sentinel2img.rio.crs
            )
            highresmask_reprojected: xr.DataArray = _highresmask_reprojected.where(
                cond=highresimg_reprojected.sum(dim="band")
                != 0
                # Turn areas with no optical imagery into NaN in the mask too
            )

            # Stack the mask as an extra channel to the RGB-NIR highres image
            highresimgmask_reprojected: xr.DataArray = xr.concat(
                objs=[highresimg_reprojected, highresmask_reprojected], dim="band"
            )

            try:
                assert highresimgmask_reprojected.dtype == np.float32
            except AssertionError:
                highresimgmask_reprojected: xr.DataArray = (
                    highresimgmask_reprojected.astype(dtype=np.float32)
                )

            # Perform the spatial alignment
            aligned_lowres, aligned_highres = align_lowres_highres_pair(
                img_highres=highresimgmask_reprojected, img_lowres=sentinel2img
            )

            # Save aligned lowres and highres images to GeoTIFF
            os.makedirs(name=f"SuperResolution/aligned/{j:04d}", exist_ok=True)

            aligned_highresmask_name: str = (
                f"SuperResolution/aligned/{j:04d}/{maskname[:-4]}_reprojected.tif"
            )
            (
                aligned_highres.isel(band=-1) / 255
            ).rio.to_raster(  # 1 channel mask, scaled to 0-1
                raster_path=aligned_highresmask_name,
                dtype=np.float32,  # store as float so that NaN can be represented
                compress="zstd",
                tfw="yes",
            )

            aligned_highresimg_name: str = (
                f"SuperResolution/aligned/{j:04d}/{hresname[:-4]}_reprojected.tif"
            )
            aligned_highres[0:4, :, :].rio.to_raster(  # 4 channel RGB+NIR image
                raster_path=aligned_highresimg_name,
                dtype=np.uint16,
                compress="zstd",
                tfw="yes",
            )

            aligned_lowres_name: str = f"SuperResolution/aligned/{j:04d}/{os.path.basename(filename[:-8])}_B8432_cropped.tif"
            aligned_lowres.rio.to_raster(
                raster_path=aligned_lowres_name, compress="zstd", tfw="yes"
            )

            j += 1
