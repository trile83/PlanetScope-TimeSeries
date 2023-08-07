import rioxarray as rxr
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import xarray as xr
import cv2
import glob

def floodfill(array):
    print(array.shape)
    crop_array = array.astype(np.uint8)
    print(crop_array.dtype)
    h, w = crop_array.shape[:2]
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    mask = np.zeros((h + 4, w + 4), np.uint8)

    canvas[1:h + 1, 1:w + 1] = crop_array.copy()
    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)
    array_flt = (~canvas | crop_array.astype(np.uint8))
    print(np.unique(array_flt))
    plt.imshow(np.squeeze(array_flt))
    plt.show()
    plt.close()

    return array_flt

def filtering_holes(mask_file):

    raster = rxr.open_rasterio(mask_file)
    
    original_array = raster.values
    original_array = np.squeeze(original_array)
    original_array = np.nan_to_num(original_array, nan=0.0)

    # corn_array = original_array.copy()
    # soy_array = original_array.copy()
    # water_array = original_array.copy()

    # corn_array[corn_array != 1] = 0
    # soy_array[soy_array != 2] = 0
    # soy_array[soy_array == 2] = 1

    # water_array[water_array != 3] = 0
    # water_array[water_array == 3] = 1

    # corn_array_flt = ndimage.binary_fill_holes(corn_array, structure=np.ones((3,3))).astype(int)
    # soy_array_flt = ndimage.binary_fill_holes(soy_array, structure=np.ones((3,3))).astype(int)
    # water_array_flt = ndimage.binary_fill_holes(water_array, structure=np.ones((3,3))).astype(int)

    # # crop_array_flt = floodfill(crop_array)
    # ## stack the classes back together
    # new_array = original_array.copy()
    # # new_array[original_array == 0] = 0
    # new_array[corn_array_flt == 1] = 1
    # new_array[soy_array_flt == 1] = 2
    # new_array[water_array_flt == 1] = 3

    nodata_array = original_array.copy()

    nodata_array[nodata_array == 0] = 9
    nodata_array[nodata_array < 9 ] = 0
    nodata_array[nodata_array == 9 ] = 1

    nodata_array_flt = ndimage.binary_fill_holes(nodata_array, structure=np.ones((3,3))).astype(int)

    # crop_array_flt = floodfill(crop_array)
    ## stack the classes back together
    new_array = original_array.copy()
    # new_array[original_array == 1] = 1
    # new_array[original_array == 2] = 2
    # new_array[original_array == 3] = 3
    new_array[nodata_array_flt == 1] = 0

    return np.squeeze(new_array)

def save_tif(out_array, mask_file):

    outdir='/home/geoint/tri/Planet_khuong/output/training-data/'

    ts_name = "tile-01" # mask segs reclassified
    # ts_name = mask_file[-53:-9]
    print(ts_name)

    ref_im = rxr.open_rasterio(mask_file)
    ref_im = ref_im.transpose("y", "x", "band")

    ref_im = ref_im.drop(
        dim="band",
        labels=ref_im.coords["band"].values[1:],
        drop=True
    )

    out_array = xr.DataArray(
        np.expand_dims(out_array, axis=-1),
        name='train',
        coords=ref_im.coords,
        dims=ref_im.dims,
        attrs=ref_im.attrs
    )

    out_array.attrs['long_name'] = ('filter')
    out_array = out_array.transpose("band", "y", "x")

    # Set nodata values on mask
    nodata = out_array.rio.nodata
    prediction = out_array.where(ref_im != nodata)
    prediction.rio.write_nodata(
        255, encoded=True, inplace=True)

    # TODO: ADD CLOUDMASKING STEP HERE
    # REMOVE CLOUDS USING THE CURRENT MASK

    # Save COG file to disk
    prediction.rio.to_raster(
        f'{outdir}{ts_name}_mask.tiff',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        # num_threads='all_cpus',
        driver='GTiff',
        dtype='uint8'
    )



if __name__ == "__main__":

    colors = ['green','orange','brown','brown','gray','white','black']
    colormap = pltc.ListedColormap(colors)
    # mask_file = '/home/geoint/tri/nasa_senegal/post_processing_mask/Tappan26_WV03_20160225_M1BS_104001001912BE00_mask.tif'
    # mask_file = '/home/geoint/tri/nasa_senegal/CAS_West_masks/Tappan26_WV03_20160225_M1BS_104001001912BE00_mask_segs_reclassified.tif'
    # mask = np.squeeze(rxr.open_rasterio(mask_file).values)

    mask_file = '/home/geoint/tri/Planet_khuong/output/training-data/label-tile01-0802.tif'

    out_mask = filtering_holes(mask_file)
    print(out_mask.shape)
    save_tif(out_mask, mask_file)

    # print("mask class counts: ",np.unique(mask, return_counts=True))
    # print("out mask class counts: ",np.unique(out_mask, return_counts=True))

    # plt.figure(figsize=(20,20))
    # plt.subplot(1,2,1)
    # plt.imshow(mask[:1000,:1000], cmap=colormap)
    # plt.subplot(1, 2, 2)
    # plt.imshow(out_mask[:1000,:1000], cmap=colormap)
    # plt.show()
    # plt.close()