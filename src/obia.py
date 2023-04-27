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

def chipper(ts_stack, mask, input_size=32):
    '''
    stack: input time-series stack to be chipped (TxCxHxW)
    mask: ground truth that need to be chipped (HxW)
    input_size: desire output size
    ** return: output stack with chipped size
    '''
    t, c, h, w = ts_stack.shape

    i = np.random.randint(0, h-input_size)
    j = np.random.randint(0, w-input_size)
    
    out_ts = np.array([ts_stack[:,:,i:(i+input_size), j:(j+input_size)]])
    out_mask = np.array([mask[i:(i+input_size), j:(j+input_size)]])

    return out_ts, out_mask

def get_field_data(field_file, pl_file):

    vector = gpd.read_file(field_file)

    # print(vector)

    out_rast = {}
    out_mask = {}

    #save output files as per shapefile features
    for i in range(len(vector)):
        
        with rio.open(pl_file) as src:
            # read imagery file
            vector=vector.to_crs(src.crs)
            geom = []
            coord = shapely.geometry.mapping(vector)["features"][i]["geometry"]
            # crop_type = vector["Crop_types"][i]
            crop_type = vector["class"][i]
            
            geom.append(coord)
            out_image, out_transform = mask.mask(src, geom, crop=True)
            full_image, full_transform = mask.mask(src, geom, crop=True, filled=False)

            # print('max of full image: ', np.max(full_image))
            # print('min of full image: ', np.min(full_image))

            # print(out_image.shape)
            # # Check that after the clip, the image is not empty
            # test = out_image[~np.isnan(out_image)]
            # if test[test > 0].shape[0] == 0:
            #     raise RuntimeError("Empty output")

            out_meta = src.profile
            out_meta.update({"height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform})

            # out_rast.append(out_image)

            if i not in out_rast.keys():
                out_rast[i] = []
            if i not in out_mask.keys():
                out_mask[i] = []

            out_rast[i].append(out_image)

            # print(f'min of out image: {np.min(out_image)}')

            if crop_type == 'corn':
                nodata_mask = np.where(out_image[0,:,:]>0,1,0)
                label = nodata_mask
                # print(label.shape)
                out_mask[i].append(label)
            elif crop_type == 'soybean':
                nodata_mask = np.where(out_image[0,:,:]>0,1,0)
                nodata_mask[nodata_mask==1] = 2
                label = nodata_mask
                out_mask[i].append(label)
            else:
                label = np.zeros((out_image.shape[1],out_image.shape[2]))
                out_mask[i].append(label)

            # print(f'crop {crop_type} unique label {np.unique(label)}')

            if full_image.ndim > 2:
                save_image = np.transpose(full_image, (1,2,0))

            # plt.figure(figsize=(20,20))
            # plt.subplot(1,2,1)
            # # plt.imshow(rescale_truncate(rescale_image(save_image[:,:,:3])))
            # plt.imshow(save_image[:,:,:3]/11000)
            # plt.subplot(1,2,2)
            # # plt.imshow(label)
            # plt.imshow(label)
            # plt.savefig(f'/home/geoint/tri/Planet_khuong/output/raster-polygon-{crop_type}-{i}.png', dpi=300, bbox_inches='tight')
            # plt.close()

            del label
            del nodata_mask

    return out_rast, out_mask

def get_raster_from_polygon(field_file, tile_name):

    data_dir = '/home/geoint/tri/Planet_khuong/'
    ts_dict = {}
    ma_dict = {}
    if tile_name == 'tile01':
        master_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/*-21/'))
        label_fl=f'{data_dir}/output/4906044_1459221_2021-09-16_2447_BGRN_SR_mask_segs_reclassified.tif'
        # data_ts = []
        # mask_ts = []
        for monthly_dir in master_dir:
            month = monthly_dir[-7:-1]
            pl_dir = f'{str(monthly_dir)}/files/PSOrthoTile/'
            img_fls = sorted(glob.glob(f'{pl_dir}/*/'))
            count=0
            for img_dir in img_fls:
                if count == 6:
                    break
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
                    rast, mask = get_field_data(field_file, fl[0])
                    if date not in ts_dict.keys():
                        ts_dict[date] = rast
                    if date not in ma_dict.keys():
                        ma_dict[date] = mask

                    count+=1

    ts_arr, ma_arr = stack_dict(ts_dict, ma_dict)

    return ts_arr, ma_arr

def stack_dict(ts_dict, ma_dict):

    im_dict = {}
    ts_arr = []
    ma_arr = []
    for date in ts_dict.keys():
        for i in ts_dict[date].keys():
            if i not in im_dict.keys():
                im_dict[i] = []

            im_dict[i].append(ts_dict[date][i])

    for date in ma_dict.keys():
        for i in ma_dict[date].keys():
            b = ma_dict[date][i][0]
            # print("length b", len(b))
            # print("b", b)
            # print("b shape: ", b.shape)
            ma_arr.append(ma_dict[date][i][0])

            del b
        break

    for key in im_dict.keys():
        a = np.stack(im_dict[key], axis=0)
        a = np.squeeze(a)
        # print('a shape: ', a.shape)
        ts_arr.append(a)

        del a

    # print("ma arr length: ", len(ma_arr))

    return ts_arr, ma_arr

def get_data(out_raster, out_mask, num_chips=1, input_size=8):

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

    if len(temp_ts_set) > 0:
        return temp_ts_set, temp_mask_set
    else:
        return [], []

    
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

    return json_data

def read_dataset(tile_name='tile01'):
    data_dir = '/home/geoint/tri/Planet_khuong/'
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
                
                if count == 5:
                    break
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

                    # print(np.unique(cloud[0], return_counts=True))
                    for band in range(img.shape[0]):
                        img[band,:,:] = img[band,:,:]*cloud[0]

                    data_ts.append(img)
                
                    count+=1

    out_ts = np.stack(data_ts, axis=0)

    label = read_imagery(label_fl, mask=True)

    print(np.unique(label, return_counts=True))

    print('out ts shape: ', out_ts.shape)
    print('label shape: ', label.shape)

    return out_ts, label


def run():
    #Loading original image

    pl_file = \
            '/home/geoint/tri/Planet_khuong/09-21/files/PSOrthoTile/4912910_1459221_2021-09-18_242d/analytic_sr_udm2/4912910_1459221_2021-09-18_242d_BGRN_SR.tif'
    field_fl = '/home/geoint/tri/Planet_khuong/Field_Survey_Polygons/Field_Survey_Polygons.shp'

    name = pl_file[-43:-4]
    planet_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    ref_im = rxr.open_rasterio(pl_file)

    if planet_data.ndim > 2:
        planet_data = np.transpose(planet_data, (1,2,0))

    # ts_arr, mask_arr = read_dataset(tile_name='tile01')

    # size = 8000
    # originImg = planet_data[:size,:size,:]

    # out_raster, out_mask = get_field_data(field_fl, pl_file)

    out_raster, out_mask = get_raster_from_polygon(field_fl, tile_name='tile01')

    # print("length out raster: ", len(out_raster))
    # print("length out mask: ", len(out_mask))

    im_lst = []
    ma_lst = []
    for idx in range(len(out_raster)):
        if np.count_nonzero(out_raster[idx]) > .25*out_raster[idx].shape[1]*out_raster[idx].shape[2]:
            im, mask = get_data(out_raster[idx], out_mask[idx],input_size=8)

            # im = np.squeeze(im)
            # mask = np.squeeze(mask)

            if len(im) > 0:
                print("im 0 shape: ", im[0].shape)
                print("mask 0 shape: ", mask[0].shape)
                im_lst.append(im)
                ma_lst.append(mask)


    print(len(im_lst))
        
    ts_arr = np.stack(im_lst, axis=0)
    ma_arr = np.stack(ma_lst, axis=0)

    print(f'ts shape: {ts_arr.shape}')
    print(f'mask shape: {ma_arr.shape}')

    
if __name__ == '__main__':

    run()