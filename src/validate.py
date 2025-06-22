import rioxarray as rxr
import fiona
import rasterio as rio
import numpy as np
import tifffile
from osgeo import gdal, ogr, osr
from shapely.geometry import box
import matplotlib.pyplot as plt
import re

def validate_cdl(cdl_file, pred_file, tile):

    cdl = tifffile.imread(cdl_file)
    pred = tifffile.imread(pred_file)

    print(pred.shape)

    ## reorganize classes in predictions
    # pred[pred == 0] = 4

    pred_uniques, pred_counts = np.unique(pred, return_counts = True)
    ## get percentage of classes:
    pred_pct = dict(zip(pred_uniques, pred_counts * 100 / (pred.shape[0]*pred.shape[1])))

    ## reorganize classes in CDL data
    cdl[cdl == 255] = 0
    cdl[cdl == 254] = 0
    cdl[cdl == 253] = 0
    cdl[cdl == 252] = 0
    cdl[cdl == 251] = 0
    cdl[cdl == 256] = 0


    cdl[cdl == 1] = 251 # corn
    cdl[cdl == 5] = 252 # soybean
    cdl[cdl == 176] = 253 # other-veg
    cdl[cdl == 36] = 254 # fall-crop
    cdl[cdl == 111] = 255 # water

    cdl[cdl == 121] = 250 # build-up
    cdl[cdl == 122] = 250 # build-up
    cdl[cdl == 123] = 250 # build-up
    cdl[cdl == 124] = 250 # build-up

    cdl[cdl < 250] = 0

    ## reassign values for main classes
    ### tile02 - South Dakota
    if tile=='tile02':
        cdl[cdl == 251] = 5 # corn
        cdl[cdl == 252] = 0 # soybean
        cdl[cdl == 253] = 4 # other-veg
        cdl[cdl == 254] = 3 # fall-crop
        cdl[cdl == 255] = 1 # water
        cdl[cdl == 250] = 2 # build-up

    elif tile=='tile01':
        cdl[cdl == 251] = 1 # corn
        cdl[cdl == 252] = 4 # soybean
        cdl[cdl == 253] = 2 # other-veg
        cdl[cdl == 254] = 5 # fall-crop
        cdl[cdl == 255] = 0 # water
        cdl[cdl == 250] = 3 # build-up
    
    cdl_uniques, cdl_counts = np.unique(cdl, return_counts = True)
    print(cdl_uniques)
    print(cdl_counts)
    ## get percentage of classes:
    cdl_pct = dict(zip(cdl_uniques, cdl_counts * 100 / (cdl.shape[0]*cdl.shape[1])))
    print(cdl_pct)

    ## plot scatterplot for 2 class percentages
    if tile == 'tile02':
        plt.scatter(cdl_pct[0], pred_pct[0], color='green', label="soybean")
        plt.scatter(cdl_pct[1], pred_pct[1], color='blue', label="water")
        plt.scatter(cdl_pct[2], pred_pct[2], color='orange', label="build-up")
        plt.scatter(cdl_pct[3], pred_pct[3], color='deeppink', label="fall-crop")
        plt.scatter(cdl_pct[4], pred_pct[4], color='lightgreen', label="other-vegetation")
        plt.scatter(cdl_pct[5], pred_pct[5], color='gold', label="corn")
    elif tile == 'tile01':
        plt.scatter(cdl_pct[0], pred_pct[0], color='blue', label="water")
        plt.scatter(cdl_pct[1], pred_pct[1], color='gold', label="corn")
        plt.scatter(cdl_pct[2], pred_pct[2], color='lightgreen', label="other-vegetation")
        plt.scatter(cdl_pct[3], pred_pct[3], color='orange', label="build-up")
        plt.scatter(cdl_pct[4], pred_pct[4], color='green', label="soybean")
        plt.scatter(cdl_pct[5], pred_pct[5], color='deeppink', label="fall-crop")

    plt.axline((0, 0), slope=1)
    plt.legend()
    plt.xlim((0,40))
    plt.ylim((0,40))
    plt.grid()
    plt.xlabel("CDL 30m class percentages")
    plt.ylabel("Gaussian Mixture prediction 3m class percentages")
    plt.savefig(f"/home/geoint/tri/Planet_khuong/validation/class-pct-scatterplot-{tile}.png", dpi=300, bbox_inches='tight')
    plt.show()


def run():

    # cdl_file = "/home/geoint/tri/Planet_khuong/Planet_SouthDakota/CDL/CDL_2021_Minnehaha.tif"
    # pred_file = "/home/geoint/tri/Planet_khuong/output/rf_out/tile01-rf-edge-0705.tiff"

    cdl_file = "/home/geoint/tri/Planet_khuong/validation/cdl_clipped-tile02.tif"
    pred_file = "/home/geoint/tri/Planet_khuong/validation/tile02-gm-6.tiff"
    tile = re.search('validation/(.*?)-gm', pred_file).group(1)
    validate_cdl(cdl_file, pred_file, tile)

    print("Finish validation!")
    

if __name__ == "__main__":

    run()