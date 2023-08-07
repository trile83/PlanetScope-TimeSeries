import numpy as np    
import cv2    
from sklearn.cluster import MeanShift, estimate_bandwidth
import rioxarray as rxr
import matplotlib.pyplot as plt
import logging
from skimage import exposure
import time
import scipy.ndimage as nd
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
    # for band in range(image.shape[2]):
    #     p2, p98 = np.percentile(image[:,:,band], (2, 98))
    #     map_img[:,:,band] = exposure.rescale_intensity(image[:,:,band], in_range=(p2, p98))
    p2, p98 = np.percentile(image[:,:], (2, 98))
    map_img[:,:] = exposure.rescale_intensity(image[:,:], in_range=(p2, p98))
    return map_img

def run():
    #Loading original image
    # originImg = cv2.imread('Swimming_Pool.jpg')

    pl_file = \
            '/home/geoint/tri/Planet_khuong/09-21/files/PSOrthoTile/4912910_1459221_2021-09-18_242d/analytic_sr_udm2/4912910_1459221_2021-09-18_242d_BGRN_SR.tif'
    field_file = '/home/geoint/tri/Planet_khuong/field/tile1_field_data.shp'

    name = pl_file[-43:-4]
    print(name)
    planet_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    ref_im = rxr.open_rasterio(pl_file)

    if planet_data.ndim > 2:
        planet_data = np.transpose(planet_data, (1,2,0))

    print(f'max input image: {np.max(planet_data)}')
    print(f'min input_image: {np.min(planet_data)}')

    size = 8000
    originImg = planet_data[:size,:size,:]

    # Shape of original image    
    originShape = originImg.shape
    print(originShape)

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities
    flatImg=originImg.reshape(-1,originImg.shape[-1])

    tic = time.time()
    # Performing sobel edge detection
    band=0
    sobelX_0 = nd.sobel(originImg[:,:,band]+0,axis=0)
    sobelY_0 = nd.sobel(originImg[:,:,band]+0,axis=1)

    band=1
    sobelX_1 = nd.sobel(originImg[:,:,band]+0,axis=0)
    sobelY_1 = nd.sobel(originImg[:,:,band]+0,axis=1)

    band=2
    sobelX_2 = nd.sobel(originImg[:,:,band]+0,axis=0)
    sobelY_2 = nd.sobel(originImg[:,:,band]+0,axis=1)

    band=3
    sobelX_3 = nd.sobel(originImg[:,:,band]+0,axis=0)
    sobelY_3 = nd.sobel(originImg[:,:,band]+0,axis=1)

    edj_0 = abs(sobelX_0)+abs(sobelY_0)
    edj_1 = abs(sobelX_1)+abs(sobelY_1)
    edj_2 = abs(sobelX_2)+abs(sobelY_2)
    edj_3 = abs(sobelX_3)+abs(sobelY_3)

    # edj_1 = rescale_image(edj_1)
    # edj_2 = rescale_image(edj_2)
    # edj_3 = rescale_image(edj_3)
    
    print(f'max edj green : {np.max(edj_2)}')
    print(f'max edj red : {np.max(edj_2)}')
    print(f'min edj nir : {np.min(edj_2)}')

    output_blue = edj_0
    output_green = edj_1
    output_red = edj_2
    output_nir = edj_3

    # save raster
    ref_im = ref_im.transpose("y", "x", "band")

    ref_im = ref_im.drop(
            dim="band",
            labels=ref_im.coords["band"].values[1:],
            drop=True
        )
    
    for idx, prediction in enumerate([output_red, output_nir, output_green, output_blue]):
        if idx == 0:
            band_name = 'red'
        elif idx == 1:
            band_name = 'nir'
        elif idx == 2:
            band_name = 'green'
        elif idx == 3:
            band_name = 'blue'

        prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name='edge-detect',
                    coords=ref_im.coords,
                    dims=ref_im.dims,
                    attrs=ref_im.attrs
                )

        # prediction = prediction.where(xraster != -9999)

        prediction.attrs['long_name'] = ('edge-detect')
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
            f'output/{name}-edge-detection-{band_name}.tiff',
            BIGTIFF="IF_SAFER",
            compress='LZW',
            # num_threads='all_cpus',
            driver='GTiff'
        )


    plt.figure(figsize=(20,20))
    plt.subplot(1,4,1)
    plt.title("Image")
    plt.imshow(rescale_truncate(rescale_image(originImg[:,:,:3])))

    
    plt.subplot(1,4,2)
    plt.title("Sobel for Red Band")
    plt.imshow(rescale_truncate(output_red),cmap='gray')

    
    plt.subplot(1,4,3)
    plt.title("Sobel for NIR band")
    plt.imshow(rescale_truncate(output_nir),cmap='gray')

    plt.subplot(1,4,4)
    plt.title("Sobel for GREEN band")
    plt.imshow(rescale_truncate(output_green),cmap='gray')
    plt.savefig(f'output/edge-detection-{name}-{size}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    print(f'time to run edge detection: {time.time()-tic} seconds')

if __name__ == '__main__':

    run()