import numpy as np    
import cv2    
from sklearn.cluster import KMeans
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
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
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
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

def read_data(fl_path, tile='tile01'):

    name = re.search(f'/planet-data/{tile}/(.*?).tif', fl_path).group(1)

    planet_data = np.squeeze(rxr.open_rasterio(fl_path, masked=True).values)
    ref_im = rxr.open_rasterio(fl_path)

    if planet_data.ndim > 2:
        planet_data = np.transpose(planet_data, (1,2,0))

    return planet_data, name

def save_raster(ref_im, prediction, name, n_clusters, model_option, mask=True):


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
            f'/home/geoint/tri/Planet_khuong/output/kmeans_outputs/{name}-ts-bayes-{model_option}-indices-{n_clusters}.tiff',
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
            f'output/kmeans_outputs/{name}_5month678910-pca-{n_clusters}-0815.tiff',
            BIGTIFF="IF_SAFER",
            compress='LZW',
            # num_threads='all_cpus',
            driver='GTiff',
            dtype='uint8'
        )

def save_raster_proba(ref_im, prediction, name):

    ref_im = ref_im.transpose("y", "x", "band")

    # ref_im = ref_im.drop(
    #         dim="band",
    #         labels=ref_im.coords["band"].values[1:],
    #         drop=True
    #     )

    print(len(ref_im['band']))
    # print(ref_im)
    # print(ref_im['spatial_ref'])
    
    prediction = xr.DataArray(
                prediction,
                name='probability',
                coords={"x": ref_im["x"], "y": ref_im["y"], 'band':[1,2,3,4,5,6,7,8,9,10], "spatial_ref":ref_im["spatial_ref"]},
                dims=ref_im.dims,
                attrs=ref_im.attrs
            )

    # prediction = prediction.where(xraster != -9999)

    # print(prediction)

    prediction.attrs['long_name'] = ('probability')
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
        f'/home/geoint/tri/Planet_khuong/output/{name}_proba-5month-0808.tiff',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        # num_threads='all_cpus',
        driver='GTiff',
        dtype='float32'
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

def add_indices(data, bands=4):

    if bands == 4 or bands == 5:
        out_array = np.zeros((data.shape[0], data.shape[1], bands+3))
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

    elif bands == 8:

        out_array = np.zeros((data.shape[0], data.shape[1], bands+3))
        ndvi = cal_ndvi(data)
        ndwi = cal_ndwi(data)
        osavi = cal_osavi(data)

        out_array[:,:,0] = data[:,:,0]
        out_array[:,:,1] = data[:,:,1]
        out_array[:,:,2] = data[:,:,2]
        out_array[:,:,3] = data[:,:,3]
        out_array[:,:,4] = data[:,:,4]
        out_array[:,:,5] = data[:,:,5]
        out_array[:,:,6] = data[:,:,6]
        out_array[:,:,7] = data[:,:,7]
        out_array[:,:,8] = ndvi
        out_array[:,:,9] = ndwi
        out_array[:,:,10] = osavi

        print(out_array.shape)

        del data

    return out_array



def run():
    #Loading original image
    tile = "tile01"
    in_data_dir = '/home/geoint/tri/planet-data'

    files = sorted(glob.glob(f'{in_data_dir}/{tile}/*.tif'))

    # files = sorted(glob.glob(f'{in_data_dir}/{tile}/*.tiff'))      

    file_list = []
    for index, file in enumerate(files):
        if index == 0:
            ref_im = rxr.open_rasterio(file)
            pl_file_07 = file

        data, name = read_data(file, tile)
        data = add_indices(data, data.shape[2])
        file_list.append(data)

    full_img = np.stack(file_list, axis=2)


    print('full stacked images: ', full_img.shape)
    # full_img[full_img < 0] = -10000

    fullimShape = full_img.shape

    data_shape = data.shape

    start_idx = 0000
    size = 8000
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

    # save model
    model_option = "gaussian-mixture" ## "gaussian-mixture" or "kmeans"

    if model_option == "gaussian-mixture":
    	model_dir = '/home/geoint/tri/Planet_khuong/output/africa/gm_model/'
    elif model_option == "kmeans":
    	model_dir = '/home/geoint/tri/Planet_khuong/output/africa/kmeans_model/'
    	
    n_clusters = 10

    # Open a file and use dump()
    if tile=='tile04' or tile=='tile08':
        filename = f'{model_dir}-trad-{model_option}-ts-indices-{n_clusters}-{tile}-asia.pkl'
    elif tile=='tile01' or tile=='tile03':
        filename = f'{model_dir}-bayes-{model_option}-ts-indices-{n_clusters}-{tile}-africa.pkl'
    elif tile=='tile09' or tile=='tile10':
        filename = f'{model_dir}-trad-{model_option}-ts-indices-{n_clusters}-{tile}-southam.pkl'
    else:
        print("No data for selected tile!")

        # filename = f'{model_dir}{model_option}-stacked-indices-{n_clusters}-{tile}-1.pkl'
    # filename = f'{model_dir}gaussian-mixture-5month678910-{n_clusters}-tile02-southdakota-0817.pkl'
    if os.path.isfile(filename):
        tic = time.time()
        print("Load model from file.")
        with open(filename, 'rb') as file:
            model = pickle.load(file)
    else:
        tic = time.time()  
        # Run clustering
        print(f"Run {model_option} from data.")
        
        if model_option == "gaussian-mixture":
        	# model = GaussianMixture(n_components=n_clusters, random_state=42)
            model = BayesianGaussianMixture(n_components=n_clusters, random_state=42)
        elif model_option == "kmeans":
        	model = KMeans(n_clusters=n_clusters, random_state=42)
    
        if pca:
            model.fit(pca_flat)
        else:
            model.fit(flatImg)

        with open(filename, 'wb') as file:
            pickle.dump(model, file)

    print(f'time to run {model_option}: {time.time()-tic} seconds')

    # Predict data

    print("Start prediction")

    if pca:
        prediction = model.predict(pca_flat)
        del pca_flat
    else:
        prediction = model.predict(full_flat)

        # proba = model.predict_proba(full_flat)

        # del full_flat

    # X_proba = proba

    # X_proba = X_proba.reshape((4096,4096,n_clusters))

    # save raster
    # save_raster_proba(ref_im, X_proba, name)


    print("Finished with prediction!")

    # print("probability shape: ", proba.shape)

    print("prediction shape: ", prediction.shape)
    X_cluster = prediction
    del prediction
    del model
    X_cluster = X_cluster.reshape((data.shape[0],data.shape[1]))

    print('X_cluster shape: ', X_cluster.shape)

    # X_cluster = np.argmax(X_cluster, axis=2)
    data = data[:,:,1:4]
    
    print("data after choose RGB: ", data.shape)

    # save raster
    save_raster(ref_im, X_cluster, name, n_clusters, model_option)
    
    colors = ['gold','green','lightgreen','blue','gray','brown']
    colormap = pltc.ListedColormap(colors)

    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(rescale_truncate(rescale_image(data[:,:,::-1])))

    plt.subplot(1,2,2)
    plt.title(f"{model_option}")
    plt.imshow(X_cluster.astype(np.uint8), cmap = colormap)
    plt.savefig(f'/home/geoint/tri/Planet_khuong/output/{tile}_5month_{model_option}-{size}-{n_clusters}-clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':

    run()
