import numpy as np    
import cv2    
from sklearn.cluster import KMeans
import rioxarray as rxr
import matplotlib.pyplot as plt
import logging
from skimage import exposure
import time
import xarray as xr
import pickle
import os
import re
from sklearn.decomposition import PCA
import glob
import json
from osgeo import gdal
from inference.inference import sliding_window_tiler_composite

def rescale_image(image: np.ndarray, rescale_type: str = 'per-image'):
    """
    Rescale image [0, 1] per-image or per-channel.
    Args:
        image (np.ndarray): array to rescale
        rescale_type (str): rescaling strategy
    Returns:
        rescaled np.ndarray
    """
    image = image.astype(np.float32)
    if rescale_type == 'per-image':
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    elif rescale_type == 'per-channel':
        for i in range(image.shape[0]):
            image[i, :, :] = (
                image[i, :, :] - np.min(image[i, :, :])) / \
                (np.max(image[i, :, :]) - np.min(image[i, :, :]))
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
    return image

def rescale_truncate(image):
    if np.amin(image) < 0:
        image = np.where(image < 0,0,image)
    if np.amax(image) > 1:
        image = np.where(image > 1,1,image) 

    map_img =  np.zeros(image.shape)
    for band in range(3):
        p2, p98 = np.percentile(image[:,:,band], (2, 98))
        map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    return map_img

def read_udm_mask(cloud_fl):

    cloud_data = np.squeeze(rxr.open_rasterio(cloud_fl, masked=True).values)
    # mask_data = np.where(cloud_data[0]==1,True,False)

    return cloud_data

def read_json(json_fl):
    
    with open(json_fl, 'r') as f:
        json_data = json.load(f)

    # print(json_data)

    return json_data

def read_imagery(pl_file, mask=False):

    img_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    ref_im = rxr.open_rasterio(pl_file)

    # if mask:
    #     img_data[img_data==9] = 1
    #     img_data[img_data==3] = 2
    #     img_data[img_data==4] = 2
    #     img_data[img_data==6] = 2
    #     img_data[img_data==5] = 3
    #     img_data[img_data==7] = 3
    #     img_data[img_data==10] = 3
    #     img_data[img_data==8] = 4
    #     img_data[img_data==2] = 5

    if mask:
        img_data[img_data==3] = 0
        img_data[img_data==7] = 1
        img_data[img_data==9] = 1
        img_data[img_data==5] = 2
        img_data[img_data==6] = 2
        img_data[img_data==8] = 2
        img_data[img_data==4] = 4
        img_data[img_data==2] = 3
        img_data[img_data==11] = 3
        img_data[img_data==1] = 5

    else:
        
        
        img_data[img_data == 0] = np.nan
        img_data = img_data.astype("float16")
        # print(np.min(img_data))


    # if img_data.ndim > 2:
    #     img_data = np.transpose(img_data, (1,2,0))

    return img_data


def merge_rasters(raster_file_list, width=8000, height=8000): ## Adding width and height as custom parameters if want to change the size of raster
      output_file = os.path.join('output', 'merged.tif')
      ds_lst = list()
      for raster in raster_file_list:
          ds = gdal.Warp('', raster, format='vrt', dstNodata=0,
                       dstSRS="+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0",
                       width=width, height=height)
          ds_lst.append(ds)
          del ds
      dataset = gdal.BuildVRT('', ds_lst, VRTNodata=0, srcNodata=0)
      ds1 = gdal.Translate(output_file, dataset)
      del ds1  
      del dataset
      return output_file

def read_dataset(tile_name='tile01', month="06"):
    data_dir = '/home/geoint/tri/Planet_khuong/'
    count=0
    if tile_name == 'tile01':
        if month == "10":
            img_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/Tile1459221_Oct2021_psorthotile_analytic_sr_udm2/PSOrthoTile/*_BGRN_SR.tif'))
            data_ts = []
            data_ts_fls = []
            for img_fl in img_dir:
                
                name = re.search(r'/PSOrthoTile/(.*?)_BGRN_SR.tif', img_fl).group(1)
                json_fl = os.path.join(os.path.dirname(img_fl), f'{name}_metadata.json')
                cloud_fl = os.path.join(os.path.dirname(img_fl), f'{name}_BGRN_DN_udm.tif')

                ## get metadata for overview and filtering for high-quality images
                metadata = read_json(json_fl)
                date = metadata['properties']['acquired']
                black_fill = metadata['properties']['black_fill']
                cloud_pct = metadata['properties']['cloud_percent']
                light_haze_pct = metadata['properties']['light_haze_percent']
                heavy_haze_pct = metadata['properties']['heavy_haze_percent']

                # if (float(black_fill) < 0.15 and cloud_pct < 12 and light_haze_pct < 8 and heavy_haze_pct < 5):
                if (cloud_pct < 2 and light_haze_pct < 2 and heavy_haze_pct < 1):

                    print('image date: ', date)

                    img = read_imagery(img_fl, mask=False)
                    cloud = read_udm_mask(cloud_fl)
                    # ndvi = cal_ndvi(img, cloud)

                    # print(np.unique(cloud[0], return_counts=True))
                    # for band in range(img.shape[0]):
                    #     img[band,:,:] = img[band,:,:]*cloud[0]


                    # img = img.astype(int)

                    data_ts.append(img)
                    # print(img_fl)
                    data_ts_fls.append(img_fl)
                
                    count+=1

                    del img

        elif month == "06":
            img_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/Tile1459221_Jun2021_psorthotile_analytic_sr_udm2/PSOrthoTile/*_BGRN_SR.tif'))
            data_ts = []
            data_ts_fls = []
            for img_fl in img_dir:
                
                name = re.search(r'/PSOrthoTile/(.*?)_BGRN_SR.tif', img_fl).group(1)
                json_fl = os.path.join(os.path.dirname(img_fl), f'{name}_metadata.json')
                cloud_fl = os.path.join(os.path.dirname(img_fl), f'{name}_BGRN_DN_udm.tif')

                ## get metadata for overview and filtering for high-quality images
                metadata = read_json(json_fl)
                date = metadata['properties']['acquired']
                black_fill = metadata['properties']['black_fill']
                cloud_pct = metadata['properties']['cloud_percent']
                light_haze_pct = metadata['properties']['light_haze_percent']
                heavy_haze_pct = metadata['properties']['heavy_haze_percent']

                # if (float(black_fill) < 0.17 and cloud_pct < 5 and light_haze_pct < 3 and heavy_haze_pct < 1):
                if (cloud_pct < 2 and light_haze_pct < 2 and heavy_haze_pct < 1):

                    print('image date: ', date)

                    img = read_imagery(img_fl, mask=False)
                    cloud = read_udm_mask(cloud_fl)
                    # ndvi = cal_ndvi(img, cloud)

                    data_ts.append(img)
                    # print(img_fl)
                    data_ts_fls.append(img_fl)
                
                    count+=1

                    del img

        else:
            pl_dir = f'/home/geoint/tri/Planet_khuong/{month}-21/files/PSOrthoTile/'

            # print(pl_dir)

            # label_fl=f'{data_dir}/output/training-data/label-tile01.tif'

            data_ts = []
            data_ts_fls = []
            img_fls = sorted(glob.glob(f'{pl_dir}/*/'))
            count=0
            for img_dir in img_fls:
            
                json_dir = sorted(glob.glob(f'{img_dir}/*.json'))
                dir = sorted(glob.glob(f'{img_dir}/analytic_sr_udm2/*.tif'))
                fl = [x for x in dir if 'SR' in x]
                cloud_fl = [x for x in dir if x[-8:-4] == 'udm2']


                ## get metadata for overview and filtering for high-quality images
                metadata = read_json(json_dir[0])
                date = metadata['properties']['acquired']
                black_fill = metadata['properties']['black_fill']
                cloud_pct = metadata['properties']['cloud_percent']
                light_haze_pct = metadata['properties']['light_haze_percent']
                heavy_haze_pct = metadata['properties']['heavy_haze_percent']

                # if (float(black_fill) < 0.08 and cloud_pct < 8 and light_haze_pct < 8 and heavy_haze_pct < 3):
                if (cloud_pct < 1 and light_haze_pct < 1 and heavy_haze_pct < 1):

                    print('image date: ', date)

                    img = read_imagery(fl[0], mask=False)
                    cloud = read_udm_mask(cloud_fl[0])

                    # print(np.unique(cloud[0], return_counts=True))

                    # for band in range(img.shape[0]):
                    #     img[band,:,:] = img[band,:,:]*cloud[0]


                    # dims = ['band', 'y', 'x']

                    # img = xr.DataArray(
                    #     img,
                    #     dims=dims
                    # )

                    # img = img.assign_coords(time = date)
                    # img = img.expand_dims(dim="time")

                    # img = img.astype(int)

                    # img[img == 0] = np.nan

                    data_ts.append(img)
                    data_ts_fls.append(fl[0])
                
                    count+=1

                    del img
                    del cloud

    elif tile_name == 'tile02':
        if month == "06":
            img_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/Tile1459222_Jun2021_psorthotile_analytic_sr_udm2/PSOrthoTile/*_BGRN_SR.tif'))
        elif month == "07":
            img_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/Tile1459222_Jul2021_psorthotile_analytic_sr_udm2/PSOrthoTile/*_BGRN_SR.tif'))
        elif month == "08":
            img_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/Tile1459222_Aug2021_psorthotile_analytic_sr_udm2/PSOrthoTile/*_BGRN_SR.tif'))
        elif month == "09":
            img_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/Tile1459222_Sep2021_psorthotile_analytic_sr_udm2/PSOrthoTile/*_BGRN_SR.tif'))
        elif month == "10":
            img_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/Tile1459222_Oct2021_psorthotile_analytic_sr_udm2/PSOrthoTile/*_BGRN_SR.tif'))

        data_ts = []
        data_ts_fls = []
        for img_fl in img_dir:
            
            name = re.search(r'/PSOrthoTile/(.*?)_BGRN_SR.tif', img_fl).group(1)
            json_fl = os.path.join(os.path.dirname(img_fl), f'{name}_metadata.json')
            cloud_fl = os.path.join(os.path.dirname(img_fl), f'{name}_BGRN_DN_udm.tif')

            ## get metadata for overview and filtering for high-quality images
            metadata = read_json(json_fl)
            date = metadata['properties']['acquired']
            black_fill = metadata['properties']['black_fill']
            cloud_pct = metadata['properties']['cloud_percent']
            light_haze_pct = metadata['properties']['light_haze_percent']
            heavy_haze_pct = metadata['properties']['heavy_haze_percent']

            # if (float(black_fill) < 0.10 and cloud_pct < 10 and light_haze_pct < 8 and heavy_haze_pct < 5):
            if (cloud_pct < 2 and light_haze_pct < 2 and heavy_haze_pct < 1):

                print('image date: ', date)

                img = read_imagery(img_fl, mask=False)
                cloud = read_udm_mask(cloud_fl)
                # ndvi = cal_ndvi(img, cloud)

                # print(np.unique(cloud[0], return_counts=True))
                # for band in range(img.shape[0]):
                #     img[band,:,:] = img[band,:,:]*cloud[0]


                # img = img.astype(int)

                data_ts.append(img)
                # print(img_fl)
                data_ts_fls.append(img_fl)
            
                count+=1

                del img

    print(len(data_ts))

    # out_fl_path = merge_rasters(data_ts_fls)
    # print(f"Save merge raster at {out_fl_path}.")

    out_ts = np.stack(data_ts, axis=0)
    # out_ts = xr.combine_by_coords(data_ts)

    del data_ts
    del data_ts_fls

    print('out ts shape: ', out_ts.shape)

    return out_ts

def read_data(fl_path):

    name = fl_path[-43:-4]
    # name = re.search(r'/allCAS/(.*?).tif', fl_path).group(1)
    planet_data = np.squeeze(rxr.open_rasterio(fl_path, masked=True).values)
    ref_im = rxr.open_rasterio(fl_path)

    if planet_data.ndim > 2:
        planet_data = np.transpose(planet_data, (1,2,0))

    return planet_data, name

def save_raster(ref_im, prediction, name):
    ref_im = ref_im.transpose("y", "x", "band")

    prediction = np.transpose(prediction, (1,2,0))

    # prediction = prediction.drop(
    #         dim="time",
    #         drop=True
    #     )
    
    prediction = xr.DataArray(
                prediction,
                name='composit',
                coords=ref_im.coords,
                dims=ref_im.dims,
                attrs=ref_im.attrs
            )

    # prediction = prediction.where(xraster != -9999)

    prediction.attrs['long_name'] = ('composit')
    prediction = prediction.transpose("band", "y", "x")

    # Set nodata values on mask
    # nodata = prediction.rio.nodata
    # prediction = prediction.where(ref_im != nodata)
    # prediction.rio.write_nodata(
    #     255, encoded=True, inplace=True)

    # TODO: ADD CLOUDMASKING STEP HERE
    # REMOVE CLOUDS USING THE CURRENT MASK

    # Save COG file to disk
    prediction.rio.to_raster(
        f'output/{name}_median_composit-2.tiff',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        # num_threads='all_cpus',
        driver='GTiff',
        dtype='uint16'
    )

def composit_ts(ts_array):

    # out_array = ts_array.median(dim="time").compute()

    out_array = sliding_window_tiler_composite(ts_array)

    # out_array  = np.nanmedian(ts_array, axis=0)

    # mean_array = np.mean(ts_array, axis=0)

    print('after median composit shape: ', out_array.shape)

    return out_array

def run():
    #Loading original image
    ts_name = "tile02"
    month = "10"

    if ts_name == 'tile01':
        pl_file ='/home/geoint/tri/Planet_khuong/09-21/files/PSOrthoTile/4912910_1459221_2021-09-18_242d/analytic_sr_udm2/4912910_1459221_2021-09-18_242d_BGRN_SR.tif'
    elif ts_name == 'tile02':
        pl_file ='/home/geoint/tri/Planet_khuong/Tile1459222_Aug2021_psorthotile_analytic_sr_udm2/PSOrthoTile/4753640_1459222_2021-08-01_2421_BGRN_SR.tif'

    ts_arr = read_dataset(tile_name=ts_name, month=month)
    ref_im = rxr.open_rasterio(pl_file)

    out_ts = composit_ts(ts_arr)

    print('max pixel value composit',out_ts.max())
    print('min pixel value composit',out_ts.min())

    out_name = f'{ts_name}-{month}'

    save_raster(ref_im, out_ts, out_name)


    # raster_file_list = ['output/','']
    # merge_rasters(raster_file_list)

   
if __name__ == '__main__':

    run()