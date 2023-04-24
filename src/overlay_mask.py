import numpy as np
import rioxarray as rxr
import glob
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
import time
import json
import csv
import os
from datetime import datetime
import pickle
import pandas as pd
from skimage import exposure

def rescale_truncate(image):
    if np.amin(image) < 0:
        image = np.where(image < 0,0,image)
    if np.amax(image) > 1:
        image = np.where(image > 1,1,image) 

    map_img =  np.zeros(image.shape)
    if image.ndim > 2:
        for band in range(image.shape[-1]):
            p2, p98 = np.percentile(image[:,:,band], (2, 98))
            map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    else:
        p2, p98 = np.percentile(image[:,:], (2, 98))
        map_img[:,:] = exposure.rescale_intensity(image[:,:], in_range=(p2, p98))
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

def get_field_data(field_file, pl_file):

    # get field data
    pts = gpd.read_file(field_file)
    pts = pts[['X', 'Y', 'crop', 'geometry']]
    pts.index = range(len(pts))
    coords = [(x, y) for x, y in zip(pts.X, pts.Y)]

    # Open the raster and store metadata
    src = rio.open(pl_file)

    # Sample the raster at every point location and store values in DataFrame
    pts['raster_value'] = [x for x in src.sample(coords)]

    return pts

def get_label(pts_arr, data, cloud, ndvi, date, ndvi_dict):

    label = np.zeros((data.shape[0],data.shape[1]))
    for i in range(len(pts_arr)):
        val_lst = pts_arr['raster_value'][i]
        crop = pts_arr['crop'][i]

        # get index in imagery based on pixel values in 4-band dimension
        ind = np.where((data[:,:,0]==val_lst[0])&
                   (data[:,:,1]==val_lst[1])&
                   (data[:,:,2]==val_lst[2])&
                   (data[:,:,3]==val_lst[3]))
        
        if cloud[ind[0][0],ind[1][0]] != 0:

            if crop == 'Soybeans':
                label[ind[0][0],ind[1][0]] = 1
            elif crop == 'Corn':
                label[ind[0][0],ind[1][0]] = 2
            elif crop == 'fall_seeded_small_grain':
                label[ind[0][0],ind[1][0]] = 3

            if (ind[0][0],ind[1][0]) not in ndvi_dict.keys():
                ndvi_dict[(ind[0][0],ind[1][0])] = {}

            if crop not in ndvi_dict[(ind[0][0],ind[1][0])].keys():
                ndvi_dict[(ind[0][0],ind[1][0])][crop] = []

            ndvi_dict[(ind[0][0],ind[1][0])][crop].append((date, ndvi[ind[0][0],ind[1][0]]))
        else:
            label[ind[0][0],ind[1][0]] = 0

    return label.astype(np.int64), ndvi_dict

def cal_ndvi(image, cloud_mask):

    ndvi_masked = np.where((cloud_mask==True) & (image[:,:,3]>0), np.divide((image[:,:,3]-image[:,:,2]), (image[:,:,3]+image[:,:,2])), 0)

    return ndvi_masked


def main():

    master_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/*-21/'))
    field_fl = '/home/geoint/tri/Planet_khuong/field/tile1_field_data.shp'
    edge_red_fl = '/home/geoint/tri/Planet_khuong/output/4912910_1459221_2021-09-18_242d_BGRN_SR-edge-detection-red.tiff'
    edge_nir_fl = '/home/geoint/tri/Planet_khuong/output/4912910_1459221_2021-09-18_242d_BGRN_SR-edge-detection-nir.tiff'
    mask_fl = '/home/geoint/tri/Planet_khuong/output/4906044_1459221_2021-09-16_2447_BGRN_SR_mask_segs_reclassified.tif'
    output_dir = 'output'

    tic = time.time()

    mask = read_imagery(mask_fl)

    edge_red = read_imagery(edge_red_fl)
    edge_ori = read_imagery(edge_red_fl)
    edge_nir = read_imagery(edge_nir_fl)

    edge_nir[edge_nir > 0.19] = 1
    edge_nir[edge_nir <= 0.19] = 0

    edge_red[edge_red >= 0.05] = 1
    edge_red[edge_red < 0.05] = 0
    
    edge_mask = edge_nir+edge_red
    edge_mask[edge_mask >= 1] = 1

    mask_out = mask + edge_mask*10
    mask_out[mask_out >= 10] = 6

    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.title("Edge RED")
    # plt.imshow(rescale_truncate(edge_ori))
    plt.imshow(edge_red)
    # plt.subplot(1,3,2)
    # plt.title("Edge RED")
    # plt.imshow(edge_red)
    plt.subplot(1,2,2)
    plt.title("Mask Out")
    plt.imshow(mask_out)
    # plt.show()
    plt.savefig(f'/home/geoint/tri/Planet_khuong/output/mask-edge.png', dpi=300, bbox_inches='tight')
    plt.close()



    print(f'time to run through all images: ', time.time()-tic)


if __name__ == '__main__':
    main()