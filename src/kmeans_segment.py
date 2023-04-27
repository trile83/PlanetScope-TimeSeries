import numpy as np    
import cv2    
from sklearn.cluster import KMeans
import rioxarray as rxr
import matplotlib.pyplot as plt
import logging
from skimage import exposure
import time
import xarray as xr

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

def run():
    #Loading original image
    # originImg = cv2.imread('Swimming_Pool.jpg')

    pl_file = \
        '/home/geoint/tri/Planet_khuong/09-21/files/PSOrthoTile/4912910_1459221_2021-09-18_242d/analytic_sr_udm2/4912910_1459221_2021-09-18_242d_BGRN_SR.tif'
    
    # pl_file = \
    #     '/home/geoint/tri/Planet_khuong/sep-21/files/PSOrthoTile/4906044_1459221_2021-09-16_2447/analytic_sr_udm2/4906044_1459221_2021-09-16_2447_BGRN_SR.tif'
    
    
    field_file = '/home/geoint/tri/Planet_khuong/field/tile1_field_data.shp'

    name = pl_file[-43:-4]
    planet_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    ref_im = rxr.open_rasterio(pl_file)

    if planet_data.ndim > 2:
        planet_data = np.transpose(planet_data, (1,2,0))

    size = 8000
    originImg = planet_data[:size,:size,:]

    # Shape of original image    
    originShape = originImg.shape
    print(originShape)

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities
    flatImg=originImg.reshape(-1,originImg.shape[-1])

    tic = time.time()  
    # Run Kmeans clustering
    n_clusters = 5
    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(flatImg)

    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(originShape[:2])

    # Performing meanshift on flatImg

    # Displaying segmented image    
    print(f'time to run kmeans: {time.time()-tic} seconds')

    # save raster
    ref_im = ref_im.transpose("y", "x", "band")

    ref_im = ref_im.drop(
            dim="band",
            labels=ref_im.coords["band"].values[1:],
            drop=True
        )
    
    prediction = X_cluster
    
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
        f'output/{name}-kmeans-{n_clusters}.tiff',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        # num_threads='all_cpus',
        driver='GTiff',
        dtype='uint8'
    )

    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(rescale_truncate(rescale_image(originImg[:,:,:3])))

    plt.subplot(1,2,2)
    plt.title("KMeans")
    plt.imshow(X_cluster.astype(np.uint8), cmap="hsv")
    plt.savefig(f'output/{name}-kmeans-{size}-{n_clusters}-clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':

    run()