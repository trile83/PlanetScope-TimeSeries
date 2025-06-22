import numpy as np    
# import cv2    
from sklearn.cluster import KMeans
# import rioxarray as rxr
# import matplotlib.pyplot as plt
# import matplotlib.colors as pltc
# import logging
# from skimage import exposure
# import time
# import xarray as xr
# import pickle
# import os
# import re
# from sklearn.decomposition import PCA
# import glob
# import json
# from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans
from clustering.isodata import isodata_classification



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



def read_data(fl_path, tile="tile01"):

    # fl_path = f'/median_composite/{tile}/'

    name = re.search(f'/median_composite/{tile}/(.*?).tiff', fl_path).group(1)
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
                    name='kmeans',
                    coords=ref_im.coords,
                    dims=ref_im.dims,
                    attrs=ref_im.attrs
                )

        # prediction = prediction.where(xraster != -9999)

        prediction.attrs['long_name'] = ('kmeans')
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
            f'/home/geoint/tri/Planet_khuong/output/{name}_5month678910-isodata-ts-{n_clusters}.tiff',
            BIGTIFF="IF_SAFER",
            compress='LZW',
            # num_threads='all_cpus',
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
            f'/home/geoint/tri/Planet_khuong/output/{name}_5month678910-pca-{n_clusters}-0815.tiff',
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



def run():
    #Loading original image
    tile = "tile01"

    if tile == "tile02":
        pl_file_09 = \
            '/home/geoint/tri/Planet_khuong/output/median_composite/tile02/tile02-09_median_composit-2.tiff'
        
        pl_file_07 = \
            '/home/geoint/tri/Planet_khuong/output/median_composite/tile02/tile02-07_median_composit-2.tiff'
        
        pl_file_06 = \
            '/home/geoint/tri/Planet_khuong/output/median_composite/tile02/tile02-06_median_composit-2.tiff'
        
        pl_file_08 = \
            '/home/geoint/tri/Planet_khuong/output/median_composite/tile02/tile02-08_median_composit-2.tiff'
        
        pl_file_10 = \
            '/home/geoint/tri/Planet_khuong/output/median_composite/tile02/tile02-10_median_composit-2.tiff'
        
    elif tile == "tile01":
        pl_file_09 = \
            '/home/geoint/tri/Planet_khuong/output/median_composite/tile01/tile01-09_median_composit-2.tiff'
        
        pl_file_07 = \
            '/home/geoint/tri/Planet_khuong/output/median_composite/tile01/tile01-07_median_composit-2.tiff'
        
        pl_file_06 = \
            '/home/geoint/tri/Planet_khuong/output/median_composite/tile01/tile01-06_median_composit-2.tiff'
        
        pl_file_08 = \
            '/home/geoint/tri/Planet_khuong/output/median_composite/tile01/tile01-08_median_composit-2.tiff'
        
        pl_file_10 = \
            '/home/geoint/tri/Planet_khuong/output/median_composite/tile01/tile01-10_median_composit-2.tiff'
        


    # ts_name = "tile01"
    # ts_arr = read_dataset(tile_name=ts_name)

    ### stacked only 3 images
    ref_im = rxr.open_rasterio(pl_file_07)

    data_06, name_06 = read_data(pl_file_06, tile)
    data_07, name_07 = read_data(pl_file_07, tile)
    data_08, name_08 = read_data(pl_file_08, tile)
    data_09, name_09 = read_data(pl_file_09, tile)
    data_10, name_10 = read_data(pl_file_10, tile)

    data_06 = add_indices(data_06)
    data_07 = add_indices(data_07)
    data_08 = add_indices(data_08)
    data_09 = add_indices(data_09)
    data_10 = add_indices(data_10)

    # data_07 = np.nan_to_num(data_07, nan=-10000)
    # data_08 = np.nan_to_num(data_08, nan=-10000)
    # data_09 = np.nan_to_num(data_09, nan=-10000)
    # data_10 = np.nan_to_num(data_10, nan=-10000)

    print('data 07 shape: ', data_07.shape)
    print('data 08 shape: ', data_08.shape)
    print('data 09 shape: ', data_09.shape)
    print('data 10 shape: ', data_10.shape)
    

    full_img = np.stack((data_06,data_07,data_08,data_09,data_10), axis=2)
    # full_img = np.stack((data_08,data_09,data_10), axis=2)

    full_img = np.transpose(full_img, (2,0,1,3))

    print('full ts image shape: ', full_img.shape)
    # full_img[full_img < 0] = -10000

    data_shape = data_07.shape


    start_idx = 0000
    size = 4000
    originImg = full_img[start_idx:size,start_idx:size,:,:]

    # Shape of original image    
    originShape = originImg.shape
    print('origin shape', originShape)

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities
    flatImg = originImg.reshape((originImg.shape[0], originImg.shape[1]* originImg.shape[2], originImg.shape[3]))
    
    ## flatten different way
    # flatImg = originImg.reshape((originImg.shape[0] * originImg.shape[1] * originImg.shape[2], originImg.shape[3]))

    full_flat = full_img.reshape((full_img.shape[0], full_img.shape[1] * full_img.shape[2], full_img.shape[3]))

    # full_flat = full_flat.reshape((full_img.shape[0] * full_img.shape[1] * full_img.shape[2], full_img.shape[3]))


    # full_flat = np.nan_to_num(full_flat, nan=-10000)

    # print('flat img: ', flatImg.shape)
    # print('full flat: ', full_flat.shape)
    
    data = data_07

    del originImg
    del full_img
    del data_06
    del data_07
    del data_08
    del data_09
    del data_10


    pca = False

    # run PCA on multi-spec
    if pca:
        pca_arr = run_pca(full_flat, components=3)
        pca_flat = pca_arr

        X_pca = pca_arr.reshape((8000,8000,3))

        save_raster(ref_im, X_pca[:,:,0], name_07, 1, mask=True)
        del pca_arr
        del X_pca

        print('pca flat: ', pca_flat.shape)

    else:

        flatImg = flatImg

    # save kmeans model
    model_dir = '/home/geoint/tri/Planet_khuong/output/kmeans_model/'
    n_clusters = 7

    # Open a file and use dump()
    filename = f'{model_dir}tskmeans-ts-{n_clusters}-{tile}-southdakota.pkl'
    if os.path.isfile(filename):
        tic = time.time()
        print("Load model from file.")
        with open(filename, 'rb') as file:
            tskmeans = pickle.load(file)
    else:
        tic = time.time()  

        print("Run tskmeans from data.")

        print("number of clusters: ", n_clusters)
        

        tskmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10, random_state=42)
        
        if pca:
            # k_means.fit(pca_flat)
            tskmeans.fit(pca_flat)
        else:
            # k_means.fit(flatImg)
            tskmeans.fit(flatImg)

        with open(filename, 'wb') as file:
            pickle.dump(tskmeans, file)

    print(f'time to run isodata: {time.time()-tic} seconds')

    # Predict data

    if pca:
        prediction = tskmeans.predict(pca_flat)
    else:
        prediction = tskmeans.predict(full_flat)

    print("prediction shape: ", prediction.shape)
    X_cluster = prediction
    # X_cluster = X_cluster.reshape((8000,8000,n_clusters))

    X_cluster = X_cluster.reshape((8000,8000))

    print('X_cluster shape: ', X_cluster.shape)

    # X_cluster = np.argmax(X_cluster, axis=2)
    
    data = data[:,:,1:4]
    
    colors = ['brown','gray','lightgreen','gold','green']
    
    colormap_pred = pltc.ListedColormap(colors)
    

    # save raster
    save_raster(ref_im, X_cluster, name_07, n_clusters)

    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(rescale_truncate(rescale_image(data[:,:,::-1])))

    plt.subplot(1,2,2)
    plt.title("KMeans")
    plt.imshow(X_cluster.astype(np.uint8), cmap=colormap_pred)
    plt.savefig(f'/home/geoint/tri/Planet_khuong/output/{name_07}_3month-kmeans-{size}-{n_clusters}-clusters-pca-southdakota-ts01.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':

    run()
