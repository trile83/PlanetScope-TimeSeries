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
import geopandas as gpd
import json
import re
import os

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

def get_label(pts_arr, data):

    label = np.zeros((data.shape[0],data.shape[1]))
    out_lst = []
    for i in range(len(pts_arr)):
        val_lst = pts_arr['raster_value'][i]
        crop = pts_arr['crop'][i]

        out_lst.append(val_lst)

        # # get index in imagery based on pixel values in 4-band dimension
        # ind = np.where((data[:,:,0]==val_lst[0])&
        #            (data[:,:,1]==val_lst[1])&
        #            (data[:,:,2]==val_lst[2])&
        #            (data[:,:,3]==val_lst[3]))
        
        # if cloud[ind[0][0],ind[1][0]] != 0:

        #     if crop == 'Soybeans':
        #         label[ind[0][0],ind[1][0]] = 1
        #     elif crop == 'Corn':
        #         label[ind[0][0],ind[1][0]] = 2
        #     elif crop == 'fall_seeded_small_grain':
        #         label[ind[0][0],ind[1][0]] = 3

        # else:
        #     label[ind[0][0],ind[1][0]] = 0

    # return label.astype(np.int64)

    output = np.array(out_lst)
    print(output.shape)
    return output


def save_raster(ref_im, prediction, name, n_clusters, model_option, mask=True):

    # saved_date = date.today()
    if mask:
        ref_im = ref_im.transpose("y", "x", "band")

        ref_im = ref_im.drop(
                dim="band",
                labels=ref_im.coords["band"].values[1:],
                drop=True
            )
        
        
        prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name=name,
                    coords=ref_im.coords,
                    dims=ref_im.dims,
                    attrs=ref_im.attrs
                )

        # prediction = prediction.where(xraster != -9999)

        prediction.attrs['long_name'] = (name)
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
            f'/home/geoint/tri/object-detection/{name}-{n_clusters}-{model_option}.tiff',
            BIGTIFF="IF_SAFER",
            compress='LZW',
            num_threads='all_cpus',
            driver='GTiff',
            dtype='uint32'
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
            f'/home/geoint/tri/Planet_khuong/output/wv/{name}_pca_{n_clusters}-1121.tiff',
            BIGTIFF="IF_SAFER",
            compress='LZW',
            # num_threads='all_cpus',
            driver='GTiff',
            dtype='uint64'
        )


def run():
    #Loading original image
    # originImg = cv2.imread('Swimming_Pool.jpg')

    pl_file = '/home/geoint/tri/WVsharpenPlanet/subset/Tappan26_WV03_20190426-NE.tif'

    # pl_file = '/home/geoint/tri/nasa_senegal/ETZ-new/Tappan30_WV02_20120122_M1BS_1030010010980300_data.tif'

    # pl_file = '/home/geoint/tri/planet-data/tile14-ts26/L15-0930E-1097N-122019.tif'

    field_fl = '/home/geoint/tri/Planet_khuong/field/tile1_field_data.shp'

    # name='Tappan26_WV03_20190426-NE'

    fl_basename = os.path.basename(pl_file)
    name = fl_basename[:-4]
    # print(fl_basename)
    # name = re.search(f'{fl_basename}(.*?).tif', pl_file).group(1)
    planet_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    ref_im = rxr.open_rasterio(pl_file)

    if planet_data.ndim > 2:
        planet_data = np.transpose(planet_data, (1,2,0))

    size = 8000
    originImg = planet_data[:size,:size,:]

    pts = get_field_data(field_fl, pl_file)
    output_points = get_label(pts, planet_data)

    # Shape of original image    
    originShape = originImg.shape
    print(originShape)

    ###### SLIC needs pixel value to be between 0 and 1, float data type

    print('SLIC segment: ')
    segment_slic = slic(planet_data[:,:,:]/4000,n_segments=1000,compactness=0.15,sigma=0.0,start_label=1)

    print(segment_slic.shape)

    save_raster(ref_im, segment_slic, name, '1000', 'slic')

    ###### Quickshift needs pixel value to be in double data type

    # print('Quickshift segment: ')
    # segment_quickshift = quickshift(planet_data[:,:,:].astype(np.double),kernel_size=7,max_dist=500,ratio=0.5,convert2lab=False)

    # print(segment_quickshift.shape)

    # save_raster(ref_im, segment_quickshift, name, '8', 'quickshift')

    

    ##################################################
    # # ### Mean Shift
    # flatImg = originImg.reshape((originImg.shape[0] * originImg.shape[1], originImg.shape[2]))

    # # # Converting image into array of dimension [nb of pixels in originImage, 3]
    # # based on r g b intensities
    # # flatImg=originImg.reshape(-1,originImg.shape[-1])

    # # Estimate bandwidth for meanshift algorithm
    # n_jobs = 4
    # # bandwidth = estimate_bandwidth(output_points, quantile=0.1, n_samples=100, random_state=42) # 0.3 can be used for cloud masking
    # # bandwidth=5
    # # print(bandwidth)
    # # ms = MeanShift(bandwidth = bandwidth, min_bin_freq=10, bin_seeding=True, max_iter=5,n_jobs=n_jobs)
    # ms = MeanShift(bandwidth = 400, min_bin_freq=10, bin_seeding=True, max_iter=5,n_jobs=n_jobs)


    # # Performing meanshift on flatImg
    # print('Mean Shift: ')
    # tic = time.time()  
    # ms.fit(flatImg)

    # out = ms.predict(flatImg)
    # print('out shape', out.shape)
    # out = out.reshape(originShape[:2])
    # print('out shape after reshape', out.shape)

    # # (r,g,b) vectors corresponding to the different clusters after meanshift    
    # labels=ms.labels_

    # # Remaining colors after meanshift    
    # cluster_centers = ms.cluster_centers_
    # # X_cluster = ms.cluster_centers_  

    # # Finding and diplaying the number of clusters    
    # labels_unique = np.unique(labels)
    # n_clusters_ = len(labels_unique)
    # print("number of estimated clusters : %d" % n_clusters_)

    # # Displaying segmented image    
    # segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
    # # X_cluster = X_cluster.reshape(originShape[:2])
    # print('segmentedImg shape: ',segmentedImg.shape)
    # print(f'time to run meanshift: {time.time()-tic} seconds')

    # save_raster(ref_im, out, name, n_clusters_, 'meanshift')

    # plt.figure(figsize=(20,20))
    # plt.subplot(1,2,1)
    # plt.title("Image")
    # plt.imshow(rescale_truncate(rescale_image(originImg[:,:,:3])))

    # plt.subplot(1,2,2)
    # plt.title("Meanshift")
    # plt.imshow(out.astype(np.uint8), cmap="hsv")
    # plt.savefig(f'output/meanshift-{size}-{n_clusters_}-clusters.png', dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close()



if __name__ == '__main__':

    run()