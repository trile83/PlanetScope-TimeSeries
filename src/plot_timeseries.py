import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_timeseries(df, name='ndvi'):

    class_type = df['class'].unique()
    class_name = ["soybean", "other", "fall-crop", "corn"]
    for i in class_type:
        df_out = df[df["class"]==i]
        # print(df_out.head(5))
        array = df[df["class"]==i].values
        y = ["June","July","August", "September", "October"]
        mean_array = np.mean(array, axis=0)
        mean_array = mean_array[:5,]
        
        if i != 0:
            if i == 1: 
                plt.plot(y, mean_array, label="soybean")
            elif i == 2: 
                plt.plot(y, mean_array, label="other")
            elif i == 3: 
                plt.plot(y, mean_array, label="fall-crop")
            elif i == 4: 
                plt.plot(y, mean_array, label="corn")
    plt.legend()
    plt.savefig(f"output/plot_timeseries_composite/{name}.png", dpi=300)
    plt.show()


if __name__ == "__main__":

    infile = "output/csv/training-pixel-tile01-label-2000p-label-0815.csv"
    df = pd.read_csv(infile)

    # print(df.head(5))

    df_ndvi = df[['ndvi-06','ndvi-07','ndvi-08','ndvi-09','ndvi-10', 'class']]
    df_red = df[['red-06','red-07','red-08','red-09','red-10', 'class']]
    df_nir = df[['nir-06','nir-07','nir-08','nir-09','nir-10', 'class']]
    df_blue = df[['blue-06','blue-07','blue-08','blue-09','blue-10', 'class']]
    df_green = df[['green-06','green-07','green-08','green-09','green-10', 'class']]

    # print(df_ndvi.head(5))

    plot_timeseries(df_ndvi, "ndvi")
    plot_timeseries(df_red, "red")
    plot_timeseries(df_nir, "nir")
    plot_timeseries(df_blue, "blue")
    plot_timeseries(df_green, "green")