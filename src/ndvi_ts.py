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

    # cloud_mask[cloud_mask == 0] = 2
    # cloud_mask = cloud_mask - 1
    # ndvi = (image[:,:,3]-image[:,:,2])/(image[:,:,3]+image[:,:,2])
    # ndvi_masked = np.ma.masked_array(ndvi, mask=cloud_mask)

    # masked = np.where((cloud_mask>0)&(image[:,:,3]>0), image, 0)

    ndvi_masked = np.where((cloud_mask==True) & (image[:,:,3]>0), np.divide((image[:,:,3]-image[:,:,2]), (image[:,:,3]+image[:,:,2])), 0)

    return ndvi_masked

def plot_ndvi_ts(ndvi_vals, date_lst, loc, crop):

    f = "%Y-%m-%dT%H:%M:%S.%fZ"
    out_dates = [datetime.strptime(x, f) for x in date_lst]

    ndvi_df = pd.DataFrame(
    {'datetime': out_dates,
     'ndvi': ndvi_vals
    })

    sorted_ndvi_df = ndvi_df.sort_values(by='datetime')

    # print(loc)
    # print(len(sorted_ndvi_df))

    if len(sorted_ndvi_df) > 1:

        plt.figure(figsize=(10,10))
        plt.title(crop)
        plt.plot(sorted_ndvi_df['datetime'], sorted_ndvi_df['ndvi'])
        # plt.show()
        plt.savefig(f'output/ndvi_ts/{loc}-{crop}.png',dpi=300, bbox_inches='tight')
        plt.close()


def write_csv(img_fls, output_dir, month):

    with open(f'{output_dir}/overview/{month}_overview.csv','w') as f1:
        writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
        writer.writerow(['date','black_fill','cloud_percent','light_haze_percent',\
                         'heavy_haze_percent','snow_ice_percent','shadow_percent','usable_data'])
        
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
            snow_ice_pct = metadata['properties']['snow_ice_percent']
            shadow_pct = metadata['properties']['shadow_percent']
            usable = metadata['properties']['usable_data']
        
            writer.writerow([date,black_fill,cloud_pct,light_haze_pct,\
                        heavy_haze_pct,snow_ice_pct,shadow_pct,usable])
        
        f1.close()


def main():

    master_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/*-21/'))
    field_fl = '/home/geoint/tri/Planet_khuong/field/tile1_field_data.shp'
    output_dir = 'output'

    tic = time.time()

    if not os.path.isfile(f'{output_dir}/ndvi_field.pickle'):

        ndvi_dict = {}
        for monthly_dir in master_dir:
            month = monthly_dir[-6:-1]
            print(month)
            pl_dir = f'/home/geoint/tri/Planet_khuong/{str(month)}/files/PSOrthoTile/'
            img_fls = sorted(glob.glob(f'{pl_dir}/*/'))

            # check if the overview file already exist
            if not os.path.isfile(f'{output_dir}/overview/{month}_overview.csv'):
                write_csv(img_fls, output_dir, month)

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
                # snow_ice_pct = metadata['properties']['snow_ice_percent']
                # shadow_pct = metadata['properties']['shadow_percent']
                # usable = metadata['properties']['usable_data']

                if (float(black_fill) < 0.35 and cloud_pct < 12 and light_haze_pct < 5 and heavy_haze_pct < 3):

                    print('date: ', date)

                    img = read_imagery(fl[0])
                    cloud = read_udm_mask(cloud_fl[0])
                    ndvi = cal_ndvi(img, cloud)

                    pts = get_field_data(field_fl, fl[0])
                    label, ndvi_dict = get_label(pts, img, cloud, ndvi, date, ndvi_dict)

                    # print('unique crop label counts: ', np.unique(label, return_counts=True))

                    plt.imshow(ndvi)
                    plt.savefig(f'output/ndvi/{date}-ndvi.png', dpi=300, bbox_inches='tight')
                    plt.close()

            
        with open(f'{output_dir}/ndvi_field.pickle', 'wb') as handle:
            pickle.dump(ndvi_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        print('Loading pickle file!')
        with open(f'{output_dir}/ndvi_field.pickle', 'rb') as handle:
            ndvi_dict = pickle.load(handle)

    # print(ndvi_dict.keys())
    for loc in ndvi_dict.keys():
        for crop in ndvi_dict[loc].keys():
            value_set = ndvi_dict[loc][crop]
            ndvi_lst = [x[1] for x in value_set]
            date_lst = [x[0] for x in value_set]

        plot_ndvi_ts(ndvi_lst, date_lst, loc, crop)

    print(f'time to run through all images: ', time.time()-tic)


if __name__ == '__main__':
    main()