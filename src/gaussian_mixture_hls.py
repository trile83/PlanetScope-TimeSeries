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
from sklearn.mixture import GaussianMixture
import h5py
from datetime import date

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

def save_raster(ref_im, prediction, name, n_clusters, model_option, mask=True):

    saved_date = date.today()
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
            f'/home/geoint/tri/Planet_khuong/output/hls/{name}-{model_option}-indices-{n_clusters}-{saved_date}.tiff',
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
            f'/home/geoint/tri/Planet_khuong/output/{name}_pca_{n_clusters}-1121.tiff',
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
    ndvi = np.divide((image[:,:,8]-image[:,:,3]), (image[:,:,8]+image[:,:,3]))
    return ndvi

def cal_ndwi(image):
    
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = np.divide((image[:,:,2]-image[:,:,8]), (image[:,:,2]+image[:,:,8]))
    return ndwi

def cal_osavi(image):
    
    np.seterr(divide='ignore', invalid='ignore')
    osavi = np.divide(((1+0.16)*(image[:,:,8]-image[:,:,3])), (image[:,:,8]+image[:,:,3]+0.16))
    return osavi

def add_indices(data):

    out_array = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    ndvi = cal_ndvi(data)
    ndwi = cal_ndwi(data)
    osavi = cal_osavi(data)

    out_array[:,:,0] = data[:,:,1]
    out_array[:,:,1] = data[:,:,2]
    out_array[:,:,2] = data[:,:,3]
    out_array[:,:,3] = data[:,:,4]
    out_array[:,:,4] = data[:,:,5]
    out_array[:,:,5] = data[:,:,6]
    out_array[:,:,6] = data[:,:,7]
    out_array[:,:,7] = data[:,:,8]
    out_array[:,:,8] = data[:,:,9]
    out_array[:,:,9] = data[:,:,10]
    # out_array[:,:,10] = ndvi
    # out_array[:,:,11] = ndwi
    # out_array[:,:,12] = osavi

    #print(out_array.shape)

    del data

    # print(out_array[:,:,4])
    # print(out_array[:,:,5])

    return out_array

def get_composite(ts_arr):

    # to get time series length closer to 10, take total frames // 10 to obtain steps
    step = ts_arr.shape[2] // 10

    out_lst = []

    # use median composite for frames within steps, e.g. if steps = 3, the composite 3 consecutive frames
    for i in range(0,ts_arr.shape[2], step):
        out_lst.append(np.median(ts_arr[:,:,i:i+step,:], axis=2))

    out_array = np.stack(out_lst, axis=2)
    del ts_arr

    return out_array


def run():
    #Loading original image
    ts_name = 'Tappan26_WV02_20171201'
    tile = 'PCV'
    
    if tile == "PEV":
    	filename = "/home/geoint/tri/hls_datacube/hls-ecas-PEV-0901.hdf5"
    elif tile == "PFV":
    	filename = "/home/geoint/tri/hls_datacube/hls-ecas-PFV-0901.hdf5"
    elif tile == "PEA":
    	filename = "/home/geoint/tri/hls_datacube/hls-PEA.hdf5"
    elif tile == "PGA":
        filename = "/home/geoint/tri/hls_datacube/hls-PGA.hdf5"
    elif tile == "PCV":
        filename = "/home/geoint/tri/hls_datacube/hls-PCV-ts26.hdf5"
    
    
    with h5py.File(filename, "r") as file:

        all_keys = sorted(list(file.keys()))
        print("all keys: ", all_keys)

        for index in range(0, len(all_keys), 2):
            if all_keys[index] == (str(ts_name)+f"_{tile}_mask"):
                print("Yes! Can get the time series!")
                mask_arr = file[all_keys[index]][()]
                ts_arr = file[all_keys[index+1]][()]

                ts_name = re.search(f'(.*?)_{tile}', all_keys[index]).group(1)

                ## transpose time series
                ts_arr = np.transpose(ts_arr, (0,2,3,1))

                print("ts arr shape: ", ts_arr.shape)

                train_ts_set = ts_arr

                # if ts_arr.shape[0] > 9:
                #     train_ts_set = np.concatenate((ts_arr[:,1:-4,:,:], ts_arr[:,-2:,:,:]), axis=1)

                ref_im_fl = f"/home/geoint/tri/resampled_senegal_hls/trimmed/{tile}/{str(ts_name)}.tif"
                ref_im = rxr.open_rasterio(ref_im_fl)

                out_ts = []

                for image in train_ts_set:
                    out_ts.append(add_indices(image))

                train_ts_set = np.stack(out_ts, axis=2)

                print("train ts set: ", train_ts_set.shape)

                #if train_ts_set.shape[2] > 15:
                    #train_ts_set = get_composite(train_ts_set)

                # mask_arr[mask_arr != 2] = 0
                # mask_arr[mask_arr == 2] = 1

                print(f'data dict {ts_name} ts shape: {train_ts_set.shape}')
                print(f'data dict {ts_name} mask shape: {mask_arr.shape}')
        
        full_img = train_ts_set

    print('full stacked images: ', full_img.shape)
    # full_img[full_img < 0] = -10000

    fullimShape = full_img.shape

    originImg = full_img

    # Shape of original image    
    originShape = originImg.shape
    print('origin shape', originShape)

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities
    flatImg = originImg.reshape((originImg.shape[0] * originImg.shape[1], originImg.shape[2] * originImg.shape[3]))

    full_flat = full_img.reshape((full_img.shape[0] * full_img.shape[1], full_img.shape[2] * full_img.shape[3]))

    del originImg

    pca = False

    # run PCA on multi-spec
    if pca:
        pca_arr = run_pca(full_flat, components=3)
        pca_flat = pca_arr

        X_pca = pca_arr.reshape((4096,4096,3))

        save_raster(ref_im, X_pca[:,:,0], ts_name, 1, mask=True)
        del pca_arr
        del X_pca

        print('pca flat: ', pca_flat.shape)

    else:
        flatImg = flatImg

    # save model
    model_option = "gaussian-mixture" ## "gaussian-mixture" or "kmeans"
    
    if model_option == "gaussian-mixture":
    	model_dir = f'/home/geoint/tri/Planet_khuong/output/hls/gm_model/'
    elif model_option == "kmeans":
    	model_dir = f'/home/geoint/tri/Planet_khuong/output/hls/kmeans_model/'
    
    n_clusters = 10

    # Open a file and use dump()
    filename = f'{model_dir}{model_option}-hls-indices-{n_clusters}-{ts_name}-{tile}-1121.pkl'

    if os.path.isfile(filename):
        tic = time.time()
        print("Load model from file.")
        with open(filename, 'rb') as file:
            model = pickle.load(file)
    else:
        tic = time.time()  
        # Run Gaussian Mixture clustering
        print(f"Run {model_option} from data.")
        
        if model_option == "gaussian-mixture":
        	model = GaussianMixture(n_components=n_clusters, random_state=0)
        elif model_option == "kmeans":
        	model = KMeans(n_clusters=n_clusters)
    
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
        del full_flat
        # prediction = tskmeans.predict(full_flat)


    print("Finished with prediction!")

    print("prediction shape: ", prediction.shape)
    X_cluster = prediction
    del prediction
    del model
    X_cluster = X_cluster.reshape((full_img.shape[0],full_img.shape[1]))

    print('X_cluster shape: ', X_cluster.shape)

    # X_cluster = np.argmax(X_cluster, axis=2)
    
    
    # Set color for visualization
    colors_label = ['green','gold','lightgreen','gray']
    colors_pred = ['brown','gray','lightgreen','gold','green']
    
    colormap_label = pltc.ListedColormap(colors_label)
    colormap_pred = pltc.ListedColormap(colors_pred)

    # save raster
    save_raster(ref_im, X_cluster, ts_name, n_clusters, model_option)

    plt.figure(figsize=(20,20))
    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(rescale_truncate(rescale_image(ts_arr[3,:,:,1:4])))

    plt.subplot(1,3,2)
    plt.title("Label")
    plt.imshow(mask_arr.astype(np.uint8), cmap=colormap_label)

    plt.subplot(1,3,3)
    plt.title(f"{model_option}")
    plt.imshow(X_cluster.astype(np.uint8), cmap=colormap_pred)
    plt.savefig(f'/home/geoint/tri/Planet_khuong/output/hls/{ts_name}-{model_option}-{n_clusters}-clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':

    run()
