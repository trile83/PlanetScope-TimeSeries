import rasterio
from rasterio.windows import Window
import math
import re
import glob

def cut_raster_into_squares(input_path, output_prefix):
    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height
        
        # Calculate the size of each square
        new_width = math.ceil(width / 2)
        new_height = math.ceil(height / 2)

        name = re.search(r'md/(.*?).tif', input_path).group(1)
        # date = re.search(r'T28PEV_(.*?)T11', input_path).group(1)

        print(name)

        # Iterate over the four quadrants
        for i in range(2):
            for j in range(2):
                # Calculate the window offsets
                x_offset = i * new_width
                y_offset = j * new_height

                # Create a window for the current quadrant
                window = Window(x_offset, y_offset, new_width, new_height)

                # Read the data within the window
                data = src.read(window=window)
                
                # Update metadata for the new sub-raster
                transform = src.window_transform(window)
                metadata = src.meta
                metadata.update({
                    'height': data.shape[1],
                    'width': data.shape[2],
                    'transform': transform
                })

                part = f'part_{i}_{j}'

                # Write the sub-raster to a new file
                output_path = f"{output_prefix}/{part}/{name}_part_{i}_{j}.tif"
                with rasterio.open(output_path, 'w', **metadata) as dst:
                    dst.write(data)


if __name__ == "__main__":

    # input_raster_dir = '/home/geoint/tri/Planet_khuong/output/median_composite/tile01/'

    input_raster_dir = '/home/geoint/tri/planet-data/naip/md/'

    output_file_prefix = '/home/geoint/tri/planet-data/naip/md/cut/'

    ## Cut large S2 raster to 4 square piece
    raster_lst = sorted(glob.glob(f'{input_raster_dir}*.tif'))

    for input_raster_path in raster_lst:
        # input_raster_path = '/home/geoint/PycharmProjects/tensorflow/out_sentinel2/PEV/T28PEV_20160311T112102.tif'
        cut_raster_into_squares(input_raster_path, output_file_prefix)