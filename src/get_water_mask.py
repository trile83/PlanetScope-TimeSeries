import rioxarray as rxr
import numpy as np
import xarray as xr


def get_water_mask(file_path):

    img_data = np.squeeze(rxr.open_rasterio(file_path, masked=True).values)

    if img_data.ndim > 2:
        img_data = np.transpose(img_data, (1,2,0))

    img_data[img_data != 3] = 0 
    img_data[img_data == 3] = 1

    return img_data


def save_raster(ref_im, prediction, name):
    ref_im = ref_im.transpose("y", "x", "band")

    ref_im = ref_im.drop(
            dim="band",
            labels=ref_im.coords["band"].values[1:],
            drop=True
        )
    
    
    prediction = xr.DataArray(
                np.expand_dims(prediction, axis=-1),
                name='water',
                coords=ref_im.coords,
                dims=ref_im.dims,
                attrs=ref_im.attrs
            )

    # prediction = prediction.where(xraster != -9999)

    prediction.attrs['long_name'] = ('water')
    prediction = prediction.transpose("band", "y", "x")

    # Set nodata values on mask
    nodata = prediction.rio.nodata
    prediction = prediction.where(ref_im != nodata)
    prediction.rio.write_nodata(
        255, encoded=True, inplace=True)

    # TODO: ADD CLOUDMASKING STEP HERE
    # REMOVE CLOUDS USING THE CURRENT MASK

    # Save COG file to disk
    prediction.rio.to_raster(
        f'output/tile01-water-0705.tiff',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        # num_threads='all_cpus',
        driver='GTiff',
        dtype='uint8'
    )


if __name__ == "__main__":


    file_path = "/home/geoint/tri/Planet_khuong/output/4912910_1459221_2021-09-18_242d_BGRN_SR-kmeans-5.tiff"
    name = file_path[-44:-5]

    ref_im = rxr.open_rasterio(file_path)

    img = get_water_mask(file_path)
    save_raster(ref_im, img, name)