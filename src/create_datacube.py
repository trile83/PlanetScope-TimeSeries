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
from sklearn.mixture import GaussianMixture
# from tslearn.clustering import TimeSeriesKMeans

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


    # if img_data.ndim > 2:
    #     img_data = np.transpose(img_data, (1,2,0))

    return img_data

def read_dataset(tile_name='tile01'):
    data_dir = '/home/geoint/tri/Planet_khuong/'
    if tile_name == 'tile01':
        master_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/*-21/'))
        label_fl=f'{data_dir}/output/training-data/label-tile01.tif'

        data_ts = []
        # count=0
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

                # if count == 20:
                #     break

    out_ts = np.stack(data_ts, axis=0)

    print('out ts shape: ', out_ts.shape)

    return out_ts

def read_data(fl_path, tile='tile01'):

    # fl_path = f'/median_composite/{tile}/'

    if tile=='tile01':
        name = re.search(r'/planet-data/tile01/(.*?).tif', fl_path).group(1)
    elif tile=='tile02':
        name = re.search(r'/planet-data/tile02/(.*?).tif', fl_path).group(1)
    elif tile=='tile03':
        name = re.search(r'/planet-data/tile03/(.*?).tif', fl_path).group(1)
    elif tile=='tile04':
        name = re.search(r'/planet-data/tile04/(.*?).tif', fl_path).group(1)
    elif tile=='tile05':
        name = re.search(r'/planet-data/tile05/(.*?).tif', fl_path).group(1)

    planet_data = np.squeeze(rxr.open_rasterio(fl_path, masked=True).values)
    ref_im = rxr.open_rasterio(fl_path)

    if planet_data.ndim > 2:
        planet_data = np.transpose(planet_data, (1,2,0))

    return planet_data, name

def save_raster(ref_im, prediction, name, n_clusters, mask=True):


    if mask:
        ref_im = ref_im.transpose("y", "x", "band")

        ref_im = ref_im.drop(
                dim="band",
                labels=ref_im.coords["band"].values[1:],
                drop=True
            )
        
        
        prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name='gm',
                    coords=ref_im.coords,
                    dims=ref_im.dims,
                    attrs=ref_im.attrs
                )

        # prediction = prediction.where(xraster != -9999)

        prediction.attrs['long_name'] = ('gm')
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
            f'output/africa/{name}_5month678910-gaussian-mixture-indices-{n_clusters}-0831.tiff',
            BIGTIFF="IF_SAFER",
            compress='LZW',
            num_threads='all_cpus',
            driver='GTiff',
            dtype='uint8'
        )

    else:
        ref_im = ref_im.transpose("y", "x", "band")

        ref_im = ref_im.drop(
                dim="band",
                labels=ref_im.coords["band"].values[3:],
                drop=True
            )
        
        
        prediction = xr.DataArray(
                    prediction,
                    name='pca',
                    coords=ref_im.coords,
                    dims=ref_im.dims,
                    attrs=ref_im.attrs
                )

        # prediction = prediction.where(xraster != -9999)

        prediction.attrs['long_name'] = ('pca')
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
            f'output/{name}_5month678910-pca-{n_clusters}-0815.tiff',
            BIGTIFF="IF_SAFER",
            compress='LZW',
            # num_threads='all_cpus',
            driver='GTiff',
            dtype='uint8'
        )


def run_pca(data, components: int):

    pca = PCA(components) # we need 4 principal components.
    converted_data = pca.fit_transform(data)
    
    print('Image shape after PCA: ',converted_data.shape)

    return converted_data

def cal_ndvi(image):
    
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = np.divide((image[:,:,3]-image[:,:,2]), (image[:,:,3]+image[:,:,2]))
    return ndvi

def cal_ndwi(image):
    
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = np.divide((image[:,:,1]-image[:,:,3]), (image[:,:,1]+image[:,:,3]))
    return ndwi

def cal_osavi(image):
    
    np.seterr(divide='ignore', invalid='ignore')
    osavi = np.divide(((1+0.16)*(image[:,:,3]-image[:,:,2])), (image[:,:,3]+image[:,:,2]+0.16))
    return osavi

def add_indices(data):

    out_array = np.zeros((data.shape[0], data.shape[1], 7))
    ndvi = cal_ndvi(data)
    ndwi = cal_ndwi(data)
    osavi = cal_osavi(data)

    out_array[:,:,0] = data[:,:,0]
    out_array[:,:,1] = data[:,:,1]
    out_array[:,:,2] = data[:,:,2]
    out_array[:,:,3] = data[:,:,3]
    out_array[:,:,4] = ndvi
    out_array[:,:,5] = ndwi
    out_array[:,:,6] = osavi

    print(out_array.shape)

    del data

    # print(out_array[:,:,4])
    # print(out_array[:,:,5])

    return out_array

def create_datacube(filepath):
    array   = filepath
    return array

def run():
    #Loading original image
    tile = "tile05"

    if tile == "tile02":
        pl_file_07 = \
            '/home/geoint/tri/planet-data/tile02/L15-0932E-1118N-07.tif'
        
        pl_file_08 = \
            '/home/geoint/tri/planet-data/tile02/L15-0932E-1118N-08.tif'
        
        pl_file_09 = \
            '/home/geoint/tri/planet-data/tile02/L15-0932E-1118N-09.tif'
        
        pl_file_10 = \
            '/home/geoint/tri/planet-data/tile02/L15-0932E-1118N-10.tif'
        
        pl_file_11 = \
            '/home/geoint/tri/planet-data/tile02/L15-0932E-1118N-11.tif'
        
        pl_file_12 = \
            '/home/geoint/tri/planet-data/tile02/L15-0932E-1118N-12.tif'
        
    elif tile == "tile01":
        pl_file_07 = \
            '/home/geoint/tri/planet-data/tile01/L15-0943E-1098N-07.tif'
        
        pl_file_08 = \
            '/home/geoint/tri/planet-data/tile01/L15-0943E-1098N-08.tif'
        
        pl_file_09 = \
            '/home/geoint/tri/planet-data/tile01/L15-0943E-1098N-09.tif'
        
        pl_file_10 = \
            '/home/geoint/tri/planet-data/tile01/L15-0943E-1098N-10.tif'
        
        pl_file_11 = \
            '/home/geoint/tri/planet-data/tile01/L15-0943E-1098N-11.tif'
        
        pl_file_12 = \
            '/home/geoint/tri/planet-data/tile01/L15-0943E-1098N-12.tif'
        
    elif tile == "tile03":
        pl_file_07 = \
            '/home/geoint/tri/planet-data/tile03/L15-0932E-1118N-07.tif'
        
        pl_file_08 = \
            '/home/geoint/tri/planet-data/tile03/L15-0932E-1118N-08.tif'
        
        pl_file_09 = \
            '/home/geoint/tri/planet-data/tile03/L15-0932E-1118N-09.tif'
        
        pl_file_10 = \
            '/home/geoint/tri/planet-data/tile03/L15-0932E-1118N-10.tif'
        
        pl_file_11 = \
            '/home/geoint/tri/planet-data/tile03/L15-0932E-1118N-11.tif'
        
        pl_file_12 = \
            '/home/geoint/tri/planet-data/tile03/L15-0932E-1118N-12.tif'
        
    elif tile == "tile04":

        pl_file_01 = \
            '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-01.tif'
        
        pl_file_02 = \
            '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-02.tif'
        
        pl_file_03 = \
            '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-03.tif'
        
        pl_file_04 = \
            '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-04.tif'
        
        pl_file_05 = \
            '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-05.tif'
        
        pl_file_06 = \
            '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-06.tif'
        
        pl_file_07 = \
            '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-07.tif'
        
        # pl_file_08 = \
        #     '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-08.tif'
        
        pl_file_09 = \
            '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-09.tif'
        
        pl_file_10 = \
            '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-10.tif'
        
        pl_file_11 = \
            '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-11.tif'
        
        pl_file_12 = \
            '/home/geoint/tri/planet-data/tile04/L15-1621E-1076N-12.tif'
        
    elif tile == "tile05":
        pl_file_01 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-01.tif'
        
        pl_file_02 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-02.tif'
        
        pl_file_03 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-03.tif'
        
        pl_file_04 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-04.tif'
        
        pl_file_05 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-05.tif'
        
        pl_file_06 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-06.tif'
        
        pl_file_07 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-07.tif'
        
        pl_file_08 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-08.tif'
        
        pl_file_09 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-09.tif'
        
        pl_file_10 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-10.tif'
        
        pl_file_11 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-11.tif'
        
        pl_file_12 = \
            '/home/geoint/tri/planet-data/tile05/L15-1633E-1083N-12.tif'
        

    ### stacked only 3 images
    ref_im = rxr.open_rasterio(pl_file_07)

    if tile =="tile05":
        data_01, name_01 = read_data(pl_file_01, tile)
        data_02, name_02 = read_data(pl_file_02, tile)
        data_03, name_03 = read_data(pl_file_03, tile)
        data_04, name_04 = read_data(pl_file_04, tile)
        data_05, name_05 = read_data(pl_file_05, tile)
        data_06, name_06 = read_data(pl_file_06, tile)
        data_07, name_07 = read_data(pl_file_07, tile)
        data_08, name_08 = read_data(pl_file_08, tile)
        data_09, name_09 = read_data(pl_file_09, tile)
        data_10, name_10 = read_data(pl_file_10, tile)
        data_11, name_11 = read_data(pl_file_11, tile)
        data_12, name_12 = read_data(pl_file_12, tile)

        print('data 08 shape: ', data_08.shape)
        print('data 09 shape: ', data_09.shape)
        print('data 10 shape: ', data_10.shape)

        data_01 = add_indices(data_01)
        data_02 = add_indices(data_02)
        data_03 = add_indices(data_03)
        data_04 = add_indices(data_04)
        data_05 = add_indices(data_05)
        data_06 = add_indices(data_06)
        data_07 = add_indices(data_07)
        data_08 = add_indices(data_08)
        data_09 = add_indices(data_09)
        data_10 = add_indices(data_10)
        data_11 = add_indices(data_11)
        data_12 = add_indices(data_12)
        

        # full_img = np.stack((data_01,data_02,data_03,data_04,data_05,data_07,data_08,data_09,data_10,data_11,data_12), axis=2)

        # full_img = np.stack((data_01,data_02,data_03,data_04,data_05,data_07,data_08,data_09,data_10,data_11,data_12), axis=2)
        full_img = np.stack((data_01,data_02,data_03,data_10,data_11,data_12), axis=2)

        data = data_10

        del data_01,data_02,data_03,data_04,data_05,data_06,data_07,data_08,data_09,data_10,data_11,data_12

    elif tile=='tile04':
        data_01, name_01 = read_data(pl_file_01, tile)
        data_02, name_02 = read_data(pl_file_02, tile)
        data_03, name_03 = read_data(pl_file_03, tile)
        data_04, name_04 = read_data(pl_file_04, tile)
        data_05, name_05 = read_data(pl_file_05, tile)
        data_06, name_06 = read_data(pl_file_06, tile)
        data_07, name_07 = read_data(pl_file_07, tile)
        data_09, name_09 = read_data(pl_file_09, tile)
        data_10, name_10 = read_data(pl_file_10, tile)
        data_11, name_11 = read_data(pl_file_11, tile)
        data_12, name_12 = read_data(pl_file_12, tile)

        print('data 09 shape: ', data_09.shape)
        print('data 10 shape: ', data_10.shape)

        data_01 = add_indices(data_01)
        data_02 = add_indices(data_02)
        data_03 = add_indices(data_03)
        data_04 = add_indices(data_04)
        data_05 = add_indices(data_05)
        data_06 = add_indices(data_06)
        data_07 = add_indices(data_07)
        # data_08 = add_indices(data_08)
        data_09 = add_indices(data_09)
        data_10 = add_indices(data_10)
        data_11 = add_indices(data_11)
        data_12 = add_indices(data_12)

        # full_img = np.stack((data_01,data_02,data_03,data_04,data_05,data_06,data_07,data_09,data_10,data_11,data_12), axis=2)

        full_img = np.stack((data_02,data_05,data_07,data_10,data_11,data_12), axis=2)

        data = data_10

        del data_01,data_02,data_03,data_04,data_05,data_06,data_07,data_09,data_10,data_11,data_12

    else:
    
        data_07, name_07 = read_data(pl_file_07, tile)
        data_08, name_08 = read_data(pl_file_08, tile)
        data_09, name_09 = read_data(pl_file_09, tile)
        data_10, name_10 = read_data(pl_file_10, tile)
        data_11, name_11 = read_data(pl_file_11, tile)
        data_12, name_12 = read_data(pl_file_12, tile)

        print('data 08 shape: ', data_08.shape)
        print('data 09 shape: ', data_09.shape)
        print('data 10 shape: ', data_10.shape)


        data_07 = add_indices(data_07)
        data_08 = add_indices(data_08)
        data_09 = add_indices(data_09)
        data_10 = add_indices(data_10)
        data_11 = add_indices(data_11)
        data_12 = add_indices(data_12)
        
        full_img = np.stack((data_07,data_08,data_09,data_10,data_11,data_12), axis=2)

        data = data_10

        del data_07,data_08,data_09,data_10,data_11,data_12


    print('full stacked images: ', full_img.shape)
    # full_img[full_img < 0] = -10000

    fullimShape = full_img.shape

    data_shape = data.shape

    start_idx = 1000
    size = 7000
    originImg = full_img[start_idx:size,start_idx:size,:,:]

    # Shape of original image    
    originShape = originImg.shape
    print('origin shape', originShape)

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities
    flatImg = originImg.reshape((originImg.shape[0] * originImg.shape[1], originImg.shape[2] * originImg.shape[3]))

    full_flat = full_img.reshape((full_img.shape[0] * full_img.shape[1], full_img.shape[2] * full_img.shape[3]))

    del originImg
    del full_img

    pca = False

    # run PCA on multi-spec
    if pca:
        pca_arr = run_pca(full_flat, components=3)
        pca_flat = pca_arr

        X_pca = pca_arr.reshape((4096,4096,3))

        save_raster(ref_im, X_pca[:,:,0], name_10, 1, mask=True)
        del pca_arr
        del X_pca

        print('pca flat: ', pca_flat.shape)

    else:

        flatImg = flatImg

    # save kmeans model
    model_dir = 'output/africa/gm_model/'
    n_clusters = 6

    # Open a file and use dump()
    if tile=='tile04' or tile=='tile05':
        filename = f'{model_dir}gaussian-mixture-5month89101112-indices-{n_clusters}-{tile}-asia-0831.pkl'
    else:
        filename = f'{model_dir}gaussian-mixture-5month89101112-indices-{n_clusters}-{tile}-africa-0831.pkl'
    # filename = f'{model_dir}gaussian-mixture-5month678910-{n_clusters}-tile02-southdakota-0817.pkl'
    if os.path.isfile(filename):
        tic = time.time()
        print("Load model from file.")
        with open(filename, 'rb') as file:
            gm = pickle.load(file)
    else:
        tic = time.time()  
        # Run Gaussian Mixture clustering
        print("Run Gaussian Mixture from data.")

        gm = GaussianMixture(n_components=n_clusters, random_state=0)
    
        if pca:
            gm.fit(pca_flat)
        else:
            gm.fit(flatImg)
            # tskmeans.fit(flatImg)

        with open(filename, 'wb') as file:
            pickle.dump(gm, file)

    print(f'time to run kmeans: {time.time()-tic} seconds')

    # Predict data

    print("Start prediction")

    if pca:
        prediction = gm.predict(pca_flat)
        del pca_flat
    else:
        prediction = gm.predict(full_flat)
        del full_flat
        # prediction = tskmeans.predict(full_flat)


    print("Finished with prediction!")

    print("prediction shape: ", prediction.shape)
    X_cluster = prediction
    del prediction
    del gm
    X_cluster = X_cluster.reshape((4096,4096))

    print('X_cluster shape: ', X_cluster.shape)

    # X_cluster = np.argmax(X_cluster, axis=2)
    

    # save raster
    save_raster(ref_im, X_cluster, name_10, n_clusters)

    # plt.figure(figsize=(20,20))
    # plt.subplot(1,2,1)
    # plt.title("Image")
    # plt.imshow(rescale_truncate(rescale_image(data_07[:,:,1:4])))

    # plt.subplot(1,2,2)
    # plt.title("KMeans")
    # plt.imshow(X_cluster.astype(np.uint8), cmap="hsv")
    # plt.savefig(f'output/{name_07}_3month-kmeans-{size}-{n_clusters}-clusters-pca-southdakota-ts01.png', dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close()

if __name__ == '__main__':

    run()
