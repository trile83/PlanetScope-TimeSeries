import numpy as np
import rioxarray as rxr
import xarray as xr
import glob
import re
import logging
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib.colors as pltc

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
    mask = np.where(image[0,:,:]>0,True,False)
    if rescale_type == 'per-image':
        image = (image - np.min(image,initial=6000,where=mask)) / \
            (np.max(image,initial=6000,where=mask) - np.min(image,initial=6000,where=mask))
    elif rescale_type == 'per-channel':
        for i in range(image.shape[0]):
            image[i, :, :] = (
                image[i, :, :] - np.min(image[i, :, :])) / \
                (np.max(image[i, :, :]) - np.min(image[i, :, :]))
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
    return image

def save_tiff(prediction, model_option, tile='tile01'):

    data_dir = 'output/rf_out/'

    if tile == 'tile01':
        ref_im_fl = \
            '/home/geoint/tri/Planet_khuong/08-21/files/PSOrthoTile/4854347_1459221_2021-08-31_241e/analytic_sr_udm2/4854347_1459221_2021-08-31_241e_BGRN_SR.tif'
    elif tile == 'tile02':
        ref_im_fl = \
            '/home/geoint/tri/Planet_khuong/Tile1459222_Aug2021_psorthotile_analytic_sr_udm2/PSOrthoTile/4753640_1459222_2021-08-01_2421_BGRN_SR.tif'
    
    ref_im = rxr.open_rasterio(ref_im_fl)
    ref_im = ref_im.transpose("y", "x", "band")

    #save Tiff file output
    # Drop image band to allow for a merge of mask
    ref_im = ref_im.drop(
        dim="band",
        labels=ref_im.coords["band"].values[1:],
        drop=True
    )

    prediction = xr.DataArray(
                np.expand_dims(prediction, axis=-1),
                name=model_option,
                coords=ref_im.coords,
                dims=ref_im.dims,
                attrs=ref_im.attrs
            )

    # prediction = prediction.where(xraster != -9999)

    prediction.attrs['long_name'] = ('predict')
    prediction.attrs['model_name'] = (model_option)
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
        f'{data_dir}{tile}-{model_option}-0818-both.tiff',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        # num_threads='all_cpus',
        driver='GTiff',
        dtype='uint8'
    )

    return np.squeeze(rxr.open_rasterio(ref_im_fl).values)

if __name__ == '__main__':

    array_dir = 'output/rf_out/'
    model_option = 'rf'
    tile="tile02"

    if model_option == 'rf':
        if tile == "tile01":
            array_fls = sorted(glob.glob('output/rf_out/tile01/*.npy'))
        elif tile == "tile02":
            array_fls = sorted(glob.glob('output/rf_out/tile02/*.npy'))
    elif model_option == 'lstm':
        if tile == "tile01":
            array_fls = sorted(glob.glob('output/lstm/tile01/*.npy'))
        elif tile == "tile02":
            array_fls = sorted(glob.glob('output/lstm/tile02/*.npy'))

    edge_fl = 'output/4912910_1459221_2021-09-18_242d_BGRN_SR-edge-detection-red.tiff'
    edge_nir_fl = 'output/4912910_1459221_2021-09-18_242d_BGRN_SR-edge-detection-nir.tiff'

    warr_lst = []
    harr_lst = []
    count = 0
    for file in array_fls:
        print(file)
        if model_option == 'rf':
            search_term = re.search(r'rf.(.*\d*).npy', file).group(1)
            search_widx = re.search(r'\d_(\d*)', search_term).group(1)
            search_hidx = re.search(r'rf-(\d*)', search_term).group(1)
        elif model_option == 'lstm':
            search_term = re.search(r'lstm.(.*\d*).npy', file).group(1)
            search_widx = re.search(r'\d_(\d*)', search_term).group(1)
            search_hidx = re.search(r'lstm-(\d*)', search_term).group(1)

        output = np.load(file)
        # print(output.shape)
        # output = output.reshape((2000,2000))
        harr_lst.append(output)
        if count == 3:
            arr_0 = np.concatenate(harr_lst, axis=1)
            warr_lst.append(arr_0)
            # print(arr_0.shape)
            harr_lst = []
            count = 0
        else:
            count+=1

    output = np.concatenate(warr_lst, axis=0)
    print(output.shape)

    

    ## read edge file
    # edge = np.squeeze(rxr.open_rasterio(edge_fl).values)
    # edge[edge<0.05]=0
    # edge[edge>=0.05]=1

    # edge_nir = np.squeeze(rxr.open_rasterio(edge_nir_fl).values)
    # edge_nir[edge_nir<0.25]=0
    # edge_nir[edge_nir>=0.25]=1

    # output = output+edge+edge_nir

    # output[output>3] = 4

    image = save_tiff(output, model_option, tile)

    image = np.transpose(image, (1,2,0))

    # # colors = ['brown','yellow', 'green', 'pink','lightgreen', 'blue','white']
    # colors = ['lightgreen','yellow', 'green', 'blue','white']
    # colormap = pltc.ListedColormap(colors)

    # ## plot prediction
    # plt.figure(figsize=(20,20))
    # plt.subplot(1,2,1)
    # plt.axis('off')
    # plt.imshow(rescale_truncate(rescale_image(image[:,:,:3])))
    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(output, colormap)
    # # plt.show()
    # plt.savefig(f'output/rf_out/{model_option}-tile02-prediction-0804-1.png', dpi=300, bbox_inches='tight')
    # plt.close()
