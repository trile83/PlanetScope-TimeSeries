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
import re
import time
import polars as pl
import pickle
import os
import pandas as pd

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
    
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = np.divide((image[:,:,3]-image[:,:,2]), (image[:,:,3]+image[:,:,2]))
    ndvi = np.nan_to_num(ndvi,nan=-10000.0)
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

def get_data_randomchoice(out_raster, out_ndvi, num_pix=100):

    # print(out_raster.shape)
    # print(np.min(out_raster[0]))

    count = 0
    h_indx=[]
    w_indx=[]
    ts_lst=[]
    ndvi_lst=[]
    while True:
        i = np.random.randint(0, out_raster.shape[0])
        j = np.random.randint(0, out_raster.shape[1])
        if out_raster[i,j,3] > 0:
            if count >= num_pix:
                break
            # h_indx.append(i)
            # w_indx.append(j)
            ts_lst.append(out_raster[i,j,:])
            ndvi_lst.append(out_ndvi[i,j])

            count+=1


    # print(len(ts_lst))
    # print(ts_lst[:5])

    # pind = np.random.randint(0,h_indx.size, size=num_pix)

    # ts = out_raster[h_indx[pind],w_indx[pind],:]
    # mask = out_ndvi[h_indx[pind],w_indx[pind]]

    ts = np.stack(ts_lst, axis=0)
    mask = np.stack(ndvi_lst, axis=0)

    del count

    # print(ts.shape)

    return ts, mask

def get_field_data(field_file, pl_file, date, pix_lst):

    vector = gpd.read_file(field_file)

    random_choice = True

    # print(vector)
    input_size = 32

    out_raster = []
    out_mask = []

    area_threshold = 150000

    vector = vector[vector['area']>area_threshold].reset_index()

    print("total polygons: ",len(vector))

    #save output files as per shapefile features
    remove_count = 0
    for i in range(len(vector)):
        
        with rio.open(pl_file) as src:
            # read imagery file
            vector=vector.to_crs(src.crs)
            geom = []
            coord = shapely.geometry.mapping(vector)["features"][i]["geometry"]
            # crop_type = vector["Crop_types"][i]
            crop_type = vector["class"][i]
            area = vector["area"][i]

            if area < area_threshold:
                continue
            
            geom.append(coord)
            out_image, out_transform = mask.mask(src, geom, crop=True)

            # print(out_image.shape)
            ## Check that after the clip, the image is not empty
            test = out_image[~np.isnan(out_image)]

            
            if test[test > 0].shape[0] == 0:
                # raise RuntimeError("Empty output")
                remove_count+=1
                continue


            out_meta = src.profile
            out_meta.update({"height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform})

            if out_image.ndim > 2:
                out_image = np.transpose(out_image, (1,2,0))

            out_ndvi = cal_ndvi(out_image)

            out_raster.append(out_image)

            if out_image.shape[0] < (input_size//2) or out_image.shape[1] < (input_size//2):
                continue

            # print(out_image.shape)
            # print(area)

            if not random_choice:

                out_image, out_ndvi = get_data(out_image, out_ndvi, input_size=input_size)

                pix_id=0
                for hidx in range(out_image.shape[0]):
                    for widx in range(out_image.shape[1]):
                        # if np.any(out_image[hidx,widx,:]) != 0:
                        # writer.writerow([i,pix_id,date,out_image[hidx,widx,0],out_image[hidx,widx,1]\
                        #                 ,out_image[hidx,widx,2],out_image[hidx,widx,3],out_ndvi[hidx,widx],crop_type])
                        pix_lst.append([i,pix_id,date,out_image[hidx,widx,0],out_image[hidx,widx,1]\
                                        ,out_image[hidx,widx,2],out_image[hidx,widx,3],out_ndvi[hidx,widx],crop_type])
                        pix_id+=1

            else:

                # get pixel by random choice
                out_image, out_ndvi = get_data_randomchoice(out_image, out_ndvi, num_pix=500)

                if out_image.shape[0] != 500:
                    print('After random choice: ', out_image.shape)

                pix_id=0

                # for idx in range(out_image.shape[0]):
                #     # if np.any(out_image[hidx,widx,:]) != 0:
                #     writer.writerow([i,pix_id,date,out_image[idx,0],out_image[idx,1]\
                #                     ,out_image[idx,2],out_image[idx,3],out_ndvi[idx],crop_type])
                #     pix_id+=1

                
                for idx in range(out_image.shape[0]):
                    pix_lst.append([i,pix_id,date,out_image[idx,0],out_image[idx,1]\
                                    ,out_image[idx,2],out_image[idx,3],out_ndvi[idx],crop_type])
                    pix_id+=1

    print("polygon removed: ", remove_count)
    print('lenght of pix lst', len(pix_lst))

    return pix_lst

def get_pix_data(pl_file, date, pix_lst, im_size=500, start_hidx=0, start_widx=0):

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
            # writer.writerow([date,out_image[hidx,widx,0],out_image[hidx,widx,1]\
            #                  ,out_image[hidx,widx,2],out_image[hidx,widx,3],out_ndvi[hidx,widx]])
            
            pix_lst.append([date, out_image[hidx,widx,0],out_image[hidx,widx,1]\
                             ,out_image[hidx,widx,2],out_image[hidx,widx,3],out_ndvi[hidx,widx]])
            
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

def save_csv(lst, schema, out_fl, get_label):

    if get_label:
        df = pl.DataFrame(lst, schema=schema)
    else:
        print(len(lst))
        df = pl.DataFrame(lst, schema=schema)
        # df = pd.DataFrame(pix_lst, columns=columns)
    print('out dataframe shape: ', df.shape)

    print(df.head(5))

    df.write_csv(out_fl, has_header=True)

    del df

def read_dataset(tile_name='tile01', get_label=True, field_fl = '', im_size=8000):
    data_dir = '/home/geoint/tri/Planet_khuong/'
    out_dir = '/home/geoint/tri/Planet_khuong/output/csv/'

    im_size = im_size
    # start_hidx = 6000
    # start_widx = 6000

    if get_label:
        data_name = 'label'
        # out_fl = f'{out_dir}dpc-unet-pixel-{tile_name}-{data_name}-32x32-label-0802.csv'
        out_fl = f'{out_dir}dpc-unet-pixel-{tile_name}-{data_name}-500p-label-0803.csv'
    else:
        data_name = 'all'
        
    
    pix_lst = []
    
    label_lst = []
    
    # with open(out_fl,'w') as f1:
    #     writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
        
    if get_label:
        # writer.writerow(['id','pixid','date','blue','green','red','nir','ndvi','class'])
        columns = ['id','pixid','date','blue','green','red','nir','ndvi','class']
        schema=[("id", pl.Int64), ("pixid", pl.Int64), ("date", pl.Utf8), ("blue", pl.Int64), ("green", pl.Int64), ("red", pl.Int64), ("nir", pl.Int64), ("ndvi", pl.Float64), ("class", pl.Utf8)]
    else:
        # writer.writerow(['date','blue','green','red','nir','ndvi'])
        columns = ['date','blue','green','red','nir','ndvi']
        schema=[("date", pl.Utf8), ("blue", pl.Int64), ("green", pl.Int64), ("red", pl.Int64), ("nir", pl.Int64), ("ndvi", pl.Float64)]

    if tile_name == 'tile01':
        master_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/output/median_composite/tile01/*_median_composit.tiff'))
        label_fl=f'{data_dir}/output/4906044_1459221_2021-09-16_2447_BGRN_SR_mask_segs_reclassified.tif'

        # data_ts = []
        tic_month = time.time()
        for monthly_fl in master_dir:

            date = re.search(r'/median_composite/tile01/tile01-(.*?)_median_composit.tiff', monthly_fl).group(1)
            print(date)

            # img = read_imagery(monthly_fl, mask=False)

            if get_label:
                label_lst = get_field_data(field_fl, monthly_fl, date, label_lst)
                save_csv(label_lst, schema, out_fl, get_label)
            else:

                for start_hidx in range(0, 8000, 2000):
                    for start_widx in range(0, 8000, 2000):
                        pix_lst = get_pix_data(monthly_fl, date, pix_lst, im_size, start_hidx, start_widx)
                        

        out_fl = f'{out_dir}planet4month-pixel-{tile_name}-{data_name}-{im_size}-{start_hidx}_{start_widx}.csv'
        save_csv(pix_lst, schema, out_fl, get_label)
        pix_lst = []

        print(f'time to run preprocessing 1 file: {time.time()-tic_month} seconds')
        

    elif tile_name == 'tile02':
        master_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/output/median_composite/tile02/*_median_composit.tiff'))
        label_fl=f'{data_dir}/output/4906044_1459221_2021-09-16_2447_BGRN_SR_mask_segs_reclassified.tif'

        data_ts = []
        tic_month = time.time()
        for monthly_fl in master_dir:

            date = re.search(r'/median_composite/tile02/tile02-(.*?)_median_composit.tiff', monthly_fl).group(1)
            # img = read_imagery(monthly_fl, mask=False)

            if get_label:
                label_lst = get_field_data(field_fl, monthly_fl, date, label_lst)
                save_csv(label_lst, schema, out_fl, get_label)
            else:
                for start_hidx in range(0, 8000, 2000):
                    for start_widx in range(0, 8000, 2000):
                        pix_lst = get_pix_data(monthly_fl, date, pix_lst, im_size, start_hidx, start_widx)

        out_fl = f'{out_dir}planet4month-pixel-{tile_name}-{data_name}-{im_size}-{start_hidx}_{start_widx}.csv'
        save_csv(pix_lst, schema, out_fl, get_label)
        pix_lst = []

        print(f'time to run preprocessing 1 file: {time.time()-tic_month} seconds')


    print(f'time to run preprocessing all pixels: {time.time()-tic_month} seconds')
        

    # df = df.to_numpy()
    # with open(out_fl, 'wb') as file:
    #     pickle.dump(df, file, pickle.HIGHEST_PROTOCOL)

    # out_ts = np.stack(data_ts, axis=0)
    # label = read_imagery(label_fl, mask=True)
    # print(np.unique(label, return_counts=True))

    # print('out ts shape: ', out_ts.shape)
    # print('label shape: ', label.shape)

    # return out_ts, label

    

    # return df


def run():
    #Loading original image

    # field_fl = '/home/geoint/tri/Planet_khuong/Field_Survey_Polygons/Field_Survey_Polygons.shp'
    # field_fl = '/home/geoint/tri/Planet_khuong/output/training-data/training-data-0802.shp'
    field_fl = '/home/geoint/tri/Planet_khuong/output/training-data/large-poly.shp'

    tic = time.time()

    im_size = 2000
    dataframe = read_dataset(tile_name='tile01', get_label=False, field_fl=field_fl, im_size = im_size)

    print("time to run csv output: ", time.time() - tic)
    print("Finish 1D time series output!")

    
if __name__ == '__main__':

    run()