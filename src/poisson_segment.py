import numpy as np    
import cv2    
from sklearn.cluster import MeanShift, estimate_bandwidth
import rioxarray as rxr
import matplotlib.pyplot as plt
import logging
from skimage import exposure
import time
import graphlearning as gl
import geopandas as gpd
import rasterio as rio

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
    for i in range(len(pts_arr)):
        val_lst = pts_arr['raster_value'][i]
        crop = pts_arr['crop'][i]

        ind = np.where((data[:,:,0]==val_lst[0])&
                   (data[:,:,1]==val_lst[1])&
                   (data[:,:,2]==val_lst[2])&
                   (data[:,:,3]==val_lst[3]))

        if crop == 'Soybeans':
            label[ind[0][0],ind[1][0]] = 1
        elif crop == 'Corn':
            label[ind[0][0],ind[1][0]] = 2
        elif crop == 'fall_seeded_small_grain':
            label[ind[0][0],ind[1][0]] = 3

    return label.astype(np.int64)

def stack_features(X, label, num_label=20):

    label = label.reshape((label.shape[0]*label.shape[1]))
    X = X.reshape((X.shape[0]*X.shape[1],X.shape[2]))

    pixel_vals = np.float32(X)
    
    train_ind = gl.trainsets.generate(label, rate=num_label)
    train_labels = label[train_ind]

    return pixel_vals, train_ind, train_labels

def poisson_predict(stacked_features, train_ind, train_labels):

    X = stacked_features

   #Build a knn graph
    k = 1000
    W = gl.weightmatrix.knn(X, k=k)
    
    #Run Poisson learning
    # class_priors = gl.utils.class_priors(label)
    model = gl.ssl.poisson(W, solver='conjugate_gradient') # gradient_descent
    pred_label = model.fit_predict(train_ind, train_labels)

    return pred_label

def run():
    #Loading original image
    # originImg = cv2.imread('Swimming_Pool.jpg')

    pl_file = \
            '/home/geoint/tri/Planet_khuong/jul-21/files/PSOrthoTile/4648313_1459221_2021-07-02_222b/analytic_sr_udm2/4648313_1459221_2021-07-02_222b_BGRN_SR.tif'
    field_file = '/home/geoint/tri/Planet_khuong/field/tile1_field_data.shp'

    planet_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)

    if planet_data.ndim > 2:
        planet_data = np.transpose(planet_data, (1,2,0))

    # Performing poisson
    tic = time.time()
    num_label = 2

    pts_arr = get_field_data(field_file, pl_file)
    # print(pts_arr)
    label = get_label(pts_arr, planet_data)
    print(np.unique(label))
    print(planet_data[3300:3600,3300:3600,:].shape)

    input_arr = planet_data[3300:3600,3300:3600,:]
    label_arr = label[3300:3600,3300:3600]
    features, ind_arr, lab_arr = stack_features(input_arr, label_arr, num_label)
    print(lab_arr)
    labels_poisson = poisson_predict(features, ind_arr, lab_arr)

    segmentedImg = labels_poisson.reshape(label_arr.shape)

    print(f'time to run poisson: {time.time()-tic} seconds')

    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(rescale_truncate(rescale_image(input_arr)))

    plt.subplot(1,2,2)
    plt.title("Poisson")
    plt.imshow(segmentedImg)
    plt.savefig(f'output/poisson-num-label-{num_label}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':

    run()