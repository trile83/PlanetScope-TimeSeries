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

def read_data(fl_path):

    # name = fl_path[-43:-4]
    name = re.search(r'/allCAS/(.*?).tif', fl_path).group(1)
    planet_data = np.squeeze(rxr.open_rasterio(fl_path, masked=True).values)
    ref_im = rxr.open_rasterio(fl_path)

    if planet_data.ndim > 2:
        planet_data = np.transpose(planet_data, (1,2,0))

    return planet_data, name

def save_raster(ref_im, prediction, name, n_clusters):
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
        f'output/{name}_3month-kmeans-{n_clusters}.tiff',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        # num_threads='all_cpus',
        driver='GTiff',
        dtype='uint8'
    )



def run():
    #Loading original image

    # pl_file_09 = \
    #     '/home/geoint/tri/Planet_khuong/09-21/files/PSOrthoTile/4912910_1459221_2021-09-18_242d/analytic_sr_udm2/4912910_1459221_2021-09-18_242d_BGRN_SR.tif'
    
    # pl_file_07 = \
    #     '/home/geoint/tri/Planet_khuong/07-21/files/PSOrthoTile/4660870_1459221_2021-07-05_100a/analytic_sr_udm2/4660870_1459221_2021-07-05_100a_BGRN_SR.tif'
    
    # pl_file_08 = \
    #     '/home/geoint/tri/Planet_khuong/08-21/files/PSOrthoTile/4794326_1459221_2021-08-13_1010/analytic_sr_udm2/4794326_1459221_2021-08-13_1010_BGRN_SR.tif'
    

    pl_file_09 = \
        '/home/geoint/tri/allCAS/Tappan01_WV02_20110430_M1BS_103001000A27E100_data.tif'
    
    pl_file_07 = \
        '/home/geoint/tri/allCAS/Tappan01_WV02_20121014_M1BS_103001001B793900_data.tif'
    
    pl_file_08 = \
        '/home/geoint/tri/allCAS/Tappan01_WV02_20130414_M1BS_103001001F227000_data.tif'
    
    
    # field_file = '/home/geoint/tri/Planet_khuong/field/tile1_field_data.shp'

    # name = pl_file[-43:-4]
    # planet_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    # ref_im = rxr.open_rasterio(pl_file)

    # if planet_data.ndim > 2:
    #     planet_data = np.transpose(planet_data, (1,2,0))

    ref_im = rxr.open_rasterio(pl_file_07)

    data_07, name_07 = read_data(pl_file_07)
    data_08, name_08 = read_data(pl_file_08)
    data_09, name_09 = read_data(pl_file_09)

    data_07 = np.nan_to_num(data_07, nan=1000)

    print('data 07 shape: ', data_07.shape)
    print('data 08 shape: ', data_08.shape)
    print('data 09 shape: ', data_09.shape)
    

    full_img = np.stack((data_07,data_08,data_09), axis=2)
    print('full stacked images: ', full_img.shape)
    full_img[full_img < 0] = 1000

    data_shape = data_07.shape
    flat07 = data_07.reshape(-1,data_07.shape[-1])
    flat08 = data_08.reshape(-1,data_08.shape[-1])
    flat09 = data_09.reshape(-1,data_09.shape[-1])


    flat07 = np.nan_to_num(flat07, nan=1000)
    flat08 = np.nan_to_num(flat07, nan=1000)
    flat09 = np.nan_to_num(flat07, nan=1000)

    start_idx = 1000
    size = 4000
    originImg = full_img[start_idx:size,start_idx:size,:,:]

    # Shape of original image    
    originShape = originImg.shape
    print('origin shape', originShape)

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities
    flatImg=originImg.reshape((originImg.shape[0] * originImg.shape[1], originImg.shape[2] * originImg.shape[3]))

    full_flat=full_img.reshape((full_img.shape[0] * full_img.shape[1], full_img.shape[2] * full_img.shape[3]))
    full_flat = np.nan_to_num(full_flat, nan=1000)

    print('flat img: ', flatImg.shape)
    print('full flat: ', full_flat.shape)

    # save kmeans model
    model_dir = 'output/kmeans_model/'
    n_clusters = 60

    # Open a file and use dump()
    filename = f'{model_dir}k-means-stacked-{n_clusters}-senegal.pkl'
    if os.path.isfile(filename):
        tic = time.time()
        print("Load model from file.")
        with open(filename, 'rb') as file:
            k_means = pickle.load(file)
    else:
        tic = time.time()  
        # Run Kmeans clustering
        print("Run KMeans from data.")
        
        k_means = KMeans(n_clusters=n_clusters)
        k_means.fit(flatImg)

        with open(filename, 'wb') as file:
            pickle.dump(k_means, file)

    print(f'time to run kmeans: {time.time()-tic} seconds')

    # Predict data
    prediction = k_means.predict(full_flat)
    print("prediction shape: ", prediction.shape)
    X_cluster = prediction
    X_cluster = X_cluster.reshape(data_shape[:2])

    print('X_cluster shape: ', X_cluster.shape)
    

    # save raster
    save_raster(ref_im, X_cluster, name_07, n_clusters)

    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(rescale_truncate(rescale_image(data_07[:,:,1:4])))

    plt.subplot(1,2,2)
    plt.title("KMeans")
    plt.imshow(X_cluster.astype(np.uint8), cmap="hsv")
    plt.savefig(f'output/{name_07}_3month-kmeans-{size}-{n_clusters}-clusters-senegal-ts01.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':

    run()