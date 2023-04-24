import numpy as np    
import cv2    
from sklearn.cluster import MeanShift, estimate_bandwidth
import rioxarray as rxr
import matplotlib.pyplot as plt
import logging
from skimage import exposure
from skimage.segmentation import quickshift, slic
import time
import scipy.ndimage as nd
import rioxarray as rxr
import xarray as xr
import rasterio as rio
import fiona
import shapely
import geopandas as gpd
import json
import rasterio.mask as mask
import glob
import csv
import h5py

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

def read_imagery(pl_file):

    img_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    ref_im = rxr.open_rasterio(pl_file)

    if img_data.ndim > 2:
        img_data = np.transpose(img_data, (1,2,0))

    return img_data

def read_udm_mask(cloud_fl):

    cloud_data = rxr.open_rasterio(cloud_fl, masked=True)
    mask_data = np.where(cloud_data[0]==1,True,False)

    return mask_data

def read_json(json_fl):
    
    with open(json_fl, 'r') as f:
        json_data = json.load(f)

    # print(json_data)

    return json_data

def cal_ndvi(image):

    # ndvi_masked = np.where((cloud_mask==True) & (image[:,:,3]>0), \
    #                        np.divide((image[:,:,3]-image[:,:,2]), \
    #                                  (image[:,:,3]+image[:,:,2])), 0)
    
    ndvi = np.divide((image[:,:,3]-image[:,:,2]), (image[:,:,3]+image[:,:,2]))

    return ndvi

def chipper(ts_stack, mask, input_size=32):
    '''
    stack: input time-series stack to be chipped (TxCxHxW)
    mask: ground truth that need to be chipped (HxW)
    input_size: desire output size
    ** return: output stack with chipped size
    '''
    h, w, c = ts_stack.shape

    i = np.random.randint(0, h-input_size)
    j = np.random.randint(0, w-input_size)
    
    out_ts = np.array([ts_stack[i:(i+input_size), j:(j+input_size), :]])
    out_mask = np.array([mask[i:(i+input_size), j:(j+input_size)]])

    return out_ts, out_mask

def sliding_chip(ts_stack, mask, input_size=32, start_hidx=0, start_widx=0):
    '''
    stack: input time-series stack to be chipped (TxCxHxW)
    mask: ground truth that need to be chipped (HxW)
    input_size: desire output size
    ** return: output stack with chipped size
    '''
    h, w, c = ts_stack.shape

    i = start_hidx
    j = start_widx

    # for i in range(0,ts_stack.shape[0])
    
    out_ts = np.array([ts_stack[i:(i+input_size), j:(j+input_size), :]])
    out_mask = np.array([mask[i:(i+input_size), j:(j+input_size)]])

    return np.squeeze(out_ts), np.squeeze(out_mask)

def get_data(out_raster, out_mask, num_chips=1, input_size=32):

    temp_ts_set = []
    temp_mask_set = []

    i=0
    count=0
    while i < num_chips:
        # print('out raster shape: ', out_raster.shape)
        ts, mask = chipper(out_raster, out_mask, input_size=input_size)
        ts = np.squeeze(ts)
        mask = np.squeeze(mask)
        
        if count == 5:
            break
        if np.any(ts==0):
            count += 1 
            continue

        # print(f'min ts ',np.min(ts))
        # print(f'max ts ',np.max(ts))
        temp_ts_set.append(ts)
        temp_mask_set.append(mask)

        i += 1

    return ts, mask

def get_field_data(field_file, pl_file, date, writer, pixlst):

    vector = gpd.read_file(field_file)

    # print(vector)

    out_raster = []
    out_mask = []

    #save output files as per shapefile features
    for i in range(len(vector)):
        
        with rio.open(pl_file) as src:
            # read imagery file
            vector=vector.to_crs(src.crs)
            geom = []
            coord = shapely.geometry.mapping(vector)["features"][i]["geometry"]
            crop_type = vector["Crop_types"][i]
            
            geom.append(coord)
            out_image, out_transform = mask.mask(src, geom, crop=True)

            # print(out_image.shape)
            ## Check that after the clip, the image is not empty
            # test = out_image[~np.isnan(out_image)]

            # if test[test > 0].shape[0] == 0:
            #     # raise RuntimeError("Empty output")
            #     continue

            out_meta = src.profile
            out_meta.update({"height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform})

            if out_image.ndim > 2:
                out_image = np.transpose(out_image, (1,2,0))

            out_ndvi = cal_ndvi(out_image)

            out_raster.append(out_image)

            out_image, out_ndvi = get_data(out_image, out_ndvi, input_size=16)
            pix_id=0
            for hidx in range(out_image.shape[0]):
                for widx in range(out_image.shape[1]):
                    # if np.any(out_image[hidx,widx,:]) != 0:
                    writer.writerow([i,pix_id,date,out_image[hidx,widx,:],out_ndvi[hidx,widx],crop_type])
                    pix_id+=1

    # return out_raster, out_mask, crop_type

def get_pix_data(pl_file, date, writer, pix_lst, im_size=500, start_hidx=0, start_widx=0):


    out_raster = []
        
    out_image = read_imagery(pl_file)
    
    if out_image.ndim > 2:
        out_image = np.transpose(out_image, (1,2,0))

    out_ndvi = cal_ndvi(out_image)

    out_raster.append(out_image)

    out_image, out_ndvi = sliding_chip(out_image, out_ndvi, input_size=im_size,start_hidx=start_hidx,start_widx=start_widx)
    # print(out_image.shape)
    for hidx in range(out_image.shape[0]):
        for widx in range(out_image.shape[1]):
            # if np.any(out_image[hidx,widx,:]) != 0:
            writer.writerow([date,out_image[hidx,widx,0],out_image[hidx,widx,1]\
                             ,out_image[hidx,widx,2],out_image[hidx,widx,3],out_ndvi[hidx,widx]])
            
            # pix_lst.append([out_image[hidx,widx,0],out_image[hidx,widx,1]\
            #                  ,out_image[hidx,widx,2],out_image[hidx,widx,3],out_ndvi[hidx,widx]])
            
            # if hidx % 10 == 0:
            #     pix_lst = []
            
    return pix_lst
    
def read_imagery(pl_file, mask=False):

    img_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    ref_im = rxr.open_rasterio(pl_file)

    if mask:
        img_data[img_data==3] = 0
        img_data[img_data==4] = 3
        img_data[img_data==5] = 4
        img_data[img_data==6] = 5

    # if img_data.ndim > 2:
    #     img_data = np.transpose(img_data, (1,2,0))

    return img_data

def read_udm_mask(cloud_fl):

    cloud_data = np.squeeze(rxr.open_rasterio(cloud_fl, masked=True).values)
    # mask_data = np.where(cloud_data[0]==1,True,False)

    return cloud_data

def read_json(json_fl):
    
    with open(json_fl, 'r') as f:
        json_data = json.load(f)

    # print(json_data)

    return json_data

def read_dataset(tile_name='tile01', get_label=True, field_fl = '', im_size=8000):
    data_dir = '/home/geoint/tri/Planet_khuong/'
    out_dir = '/home/geoint/tri/Planet_khuong/output/csv/'

    im_size = im_size
    start_hidx = 0000
    start_widx = 2000

    if get_label:
        data_name = 'label'
    else:
        data_name = 'all'
    
    pix_lst = []

    with open(f'{out_dir}dpc-unet-pixel-{data_name}-{im_size}-{start_hidx}_{start_widx}.csv','w') as f1:
        writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
        if get_label:
            writer.writerow(['id','pixid','date','bands','ndvi','crop'])
        else:
            writer.writerow(['date','blue','green','red','nir','ndvi'])

        if tile_name == 'tile01':
            master_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/*-21/'))
            label_fl=f'{data_dir}/output/4906044_1459221_2021-09-16_2447_BGRN_SR_mask_segs_reclassified.tif'

            data_ts = []
            for monthly_dir in master_dir:
                month = monthly_dir[-7:-1]
                pl_dir = f'{str(monthly_dir)}/files/PSOrthoTile/'
                img_fls = sorted(glob.glob(f'{pl_dir}/*/'))
                count=0
                for img_dir in img_fls:
                    
                    # if count == 5:
                    #     break
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

                    if (float(black_fill) < 0.15 and cloud_pct < 12 and light_haze_pct < 5 and heavy_haze_pct < 3):

                        print('image date: ', date)

                        img = read_imagery(fl[0], mask=False)
                        cloud = read_udm_mask(cloud_fl[0])
                        # ndvi = cal_ndvi(img, cloud)

                        if get_label:
                            get_field_data(field_fl, fl[0], date, writer)
                        else:
                            pix_lst = get_pix_data(fl[0], date, writer, pix_lst, im_size, start_hidx, start_widx)

                        # print(np.unique(cloud[0], return_counts=True))
                        for band in range(img.shape[0]):
                            img[band,:,:] = img[band,:,:]*cloud[0]

                        data_ts.append(img)
                    
                        count+=1

    # out_ts = np.stack(data_ts, axis=0)

    # label = read_imagery(label_fl, mask=True)

    # print(np.unique(label, return_counts=True))

    # print('out ts shape: ', out_ts.shape)
    # print('label shape: ', label.shape)

    # return out_ts, label

    return pix_lst


def run():
    #Loading original image

    # pl_file = \
    #         '/home/geoint/tri/Planet_khuong/08-21/files/PSOrthoTile/4854347_1459221_2021-08-31_241e/analytic_sr_udm2/4854347_1459221_2021-08-31_241e_BGRN_SR.tif'
    field_fl = '/home/geoint/tri/Planet_khuong/Field_Survey_Polygons/Field_Survey_Polygons.shp'

    # name = pl_file[-43:-4]
    # date = pl_file[-27:-12]
    # print(date)
    # planet_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    # ref_im = rxr.open_rasterio(pl_file)

    # if planet_data.ndim > 2:
    #     planet_data = np.transpose(planet_data, (1,2,0))

    # ts_arr, mask_arr = read_dataset(tile_name='tile01')

    # size = 8000
    # originImg = planet_data[:size,:size,:]

    # out_raster, out_mask = get_field_data(field_fl, pl_file)

    im_size = 2000
    pix_lst = read_dataset(tile_name='tile01', get_label=False, field_fl=field_fl, im_size = im_size)

    # pix_array = np.stack(pix_lst, axis=0)

    # print(pix_array.shape)

    # array_filename = '/home/geoint/tri/Planet_khuong/output/array/pixarray.pkl'
    # with open(array_filename, 'wb') as outp:  # Overwrites any existing file.
    #     pickle.dump(pix_array, outp, pickle.HIGHEST_PROTOCOL)

    # array_filename = f'/home/geoint/tri/Planet_khuong/output/array/pixarray-{im_size}.hdf5'
    # h = h5py.File(array_filename, 'w')
    # h.create_dataset('data', data=pix_array, compression="gzip", compression_opts=9)

    print("Finish 1D time series output!")

    
if __name__ == '__main__':

    run()