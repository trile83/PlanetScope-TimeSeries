from osgeo import gdal
import glob
import re
import xarray as xr
import numpy as np


if __name__ == '__main__':

    file_dir = 'gaussian/'

    fl_lst = sorted(glob.glob(file_dir+'/*.tif'))

    # resample the senegal mask
    for file in fl_lst:
        name = re.search(r'gaussian/(.*?)_mask.tif', file).group(1)
        # print(name)
        ds = gdal.Open(file)
        proj = ds.GetProjection()
    
        input = file
        output = f"resampled_planet/{name}_hls_mask.tif"
    
        options = gdal.WarpOptions(
            xRes=30,
            yRes=30,
            # creationOptions="COMPRESS=LZW",
            creationOptions=None,
            dstSRS=proj,
            dstNodata=0,
            targetAlignedPixels = True,
            resampleAlg="mode"
            # options='overwrite'
        )
    
        gdal.Warp(output, input, options=options)

    # check resample data size
    mask_fl_lst = sorted(glob.glob("/home/geoint/tri/resampled_senegal_hls/mode/ETZ/*.tif"))
    for mask_fl in mask_fl_lst:
        mask = np.squeeze(xr.open_rasterio(mask_fl).values)
        print(mask.shape)
        print(np.unique(mask))

        d_mask = np.ma.where(mask<255, True, False)
        # print(d_mask)


