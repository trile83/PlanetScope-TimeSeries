import rioxarray as rxr
import fiona
import rasterio as rio
import numpy as np
import tifffile
from osgeo import gdal, ogr, osr
from shapely.geometry import box
import matplotlib.pyplot as plt

def validate_cdl(cdl_file, pred_file):

    cdl = tifffile.imread(cdl_file)
    pred = tifffile.imread(pred_file)

    print(pred.shape)

    ## reorganize classes in predictions
    pred[pred == 0] = 4

    pred_uniques, pred_counts = np.unique(pred, return_counts = True)
    ## get percentage of classes:
    pred_pct = dict(zip(pred_uniques, pred_counts * 100 / (pred.shape[0]*pred.shape[1])))

    ## reorganize classes in CDL data
    cdl[cdl == 255] = 0
    cdl[cdl == 254] = 0
    cdl[cdl == 253] = 0
    cdl[cdl == 252] = 0
    cdl[cdl == 251] = 0

    cdl[cdl == 1] = 251
    cdl[cdl == 5] = 252
    cdl[cdl == 176] = 253
    cdl[cdl == 36] = 254
    cdl[cdl == 111] = 255

    cdl[cdl < 251] = 0

    ## reassign values for main classes
    cdl[cdl == 251] = 1
    cdl[cdl == 252] = 2
    cdl[cdl == 253] = 4
    cdl[cdl == 254] = 3
    cdl[cdl == 255] = 5
    
    cdl_uniques, cdl_counts = np.unique(cdl, return_counts = True)
    ## get percentage of classes:
    cdl_pct = dict(zip(cdl_uniques, cdl_counts * 100 / (cdl.shape[0]*cdl.shape[1])))

    ## plot scatterplot for 2 class percentages
    plt.scatter(cdl_pct[1], pred_pct[1], color='gold', label="corn")
    plt.scatter(cdl_pct[2], pred_pct[2], color='green', label="soybean")
    plt.scatter(cdl_pct[3], pred_pct[3], color='deeppink', label="fall-crop")
    plt.scatter(cdl_pct[4], pred_pct[4], color='lightgreen', label="other")
    plt.scatter(cdl_pct[5], pred_pct[5], color='blue', label="water")
    plt.axline((0, 0), slope=1)
    plt.legend()
    plt.xlim((0,40))
    plt.ylim((0,40))
    plt.grid()
    plt.xlabel("CDL 30m class percentages")
    plt.ylabel("Random Forest prediction 3m class percentages")
    plt.savefig("/home/geoint/tri/Planet_khuong/validation/class-pct-scatterplot.png", dpi=300, bbox_inches='tight')
    plt.show()


def run():

    # cdl_file = "/home/geoint/tri/Planet_khuong/Planet_SouthDakota/CDL/CDL_2021_Minnehaha.tif"
    # pred_file = "/home/geoint/tri/Planet_khuong/output/rf_out/tile01-rf-edge-0705.tiff"

    cdl_file = "/home/geoint/tri/Planet_khuong/validation/cdl_clipped.tif"
    pred_file = "/home/geoint/tri/Planet_khuong/validation/randomforest_clipped.tif"

    validate_cdl(cdl_file, pred_file)

    print("Finish validation!")
    


if __name__ == "__main__":

    run()