import numpy as np
import rioxarray as rxr
import xarray as xr
import glob
import re
import matplotlib.pyplot as plt

def save_tiff(prediction):

    data_dir = 'output/rf_out/'

    ref_im_fl = \
        '/home/geoint/tri/Planet_khuong/08-21/files/PSOrthoTile/4854347_1459221_2021-08-31_241e/analytic_sr_udm2/4854347_1459221_2021-08-31_241e_BGRN_SR.tif'
    
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
                name='otcb',
                coords=ref_im.coords,
                dims=ref_im.dims,
                attrs=ref_im.attrs
            )

    # prediction = prediction.where(xraster != -9999)

    prediction.attrs['long_name'] = ('otcb')
    prediction.attrs['model_name'] = ('lstm')
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
        f'{data_dir}tile01-lstm-edge.tif',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        # num_threads='all_cpus',
        driver='GTiff',
        dtype='uint8'
    )

if __name__ == '__main__':

    array_dir = 'output/rf_out/'
    array_fls = sorted(glob.glob('output/rf_out/lstm/*.npy'))

    edge_fl = 'output/4912910_1459221_2021-09-18_242d_BGRN_SR-edge-detection-red.tiff'
    edge_nir_fl = 'output/4912910_1459221_2021-09-18_242d_BGRN_SR-edge-detection-nir.tiff'

    warr_lst = []
    harr_lst = []
    count = 0
    for file in array_fls:
        print(file)
        search_term = re.search(r'lstm.(.*\d*).npy', file).group(1)
        search_widx = re.search(r'\d_(\d*)', search_term).group(1)
        search_hidx = re.search(r'lstm-(\d*)', search_term).group(1)
        output = np.load(file)
        print(output.shape)
        # output = output.reshape((2000,2000))
        harr_lst.append(output)
        if count == 3:
            arr_0 = np.concatenate(harr_lst, axis=1)
            warr_lst.append(arr_0)
            print(arr_0.shape)
            harr_lst = []
            count = 0
        else:
            count+=1

    output = np.concatenate(warr_lst, axis=0)
    print(output.shape)

    

    ## read edge file
    edge = np.squeeze(rxr.open_rasterio(edge_fl).values)
    edge[edge<0.05]=0
    edge[edge>=0.05]=1

    edge_nir = np.squeeze(rxr.open_rasterio(edge_nir_fl).values)
    edge_nir[edge_nir<0.25]=0
    edge_nir[edge_nir>=0.25]=1

    output = output+edge*3+edge_nir*4

    output[output>4] = 5

    save_tiff(output)

    ## plot prediction
    plt.figure(figsize=(20,20))
    plt.axis('off')
    plt.imshow(output)
    # plt.show()
    plt.savefig('output/rf_out/lstm-prediction-edge-new.png', dpi=300, bbox_inches='tight')
    plt.close()

    
    

