import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import polars as pl
import tifffile
import logging
from skimage import exposure
import re
import pickle
import glob


np.set_printoptions(suppress=True)

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

def standardize_image(
    image,
    standardization_type: str,
    mean: list = None,
    std: list = None
):
    """
    Standardize image within parameter, simple scaling of values.
    Loca, Global, and Mixed options.
    """
    image = image.astype(np.float32)
    mask = np.where(image[0, :, :] >= 0, True, False)

    if standardization_type == 'local':
        for i in range(image.shape[0]):
            image[i, :, :] = (
                image[i, :, :] - np.mean(image[i, :, :], where=mask)) / \
                (np.std(image[i, :, :], where=mask) + 1e-8)
    elif standardization_type == 'global':
        for i in range(image.shape[-1]):
            image[:, :, i] = (image[:, :, i] - mean[i]) / (std[i] + 1e-8)
    elif standardization_type == 'mixed':
        raise NotImplementedError
    return image


def train_random_forest(training_file, khuong, scale=False):

    print("Start random_forest!")
    # check if model exists
    if khuong:
        filename = 'output/model_saved/randomforest_model.sav'
    else:
        # filename = 'output/model_saved/randomforest_model_0425.sav'
        filename = 'output/model_saved/randomforest_model_0818-2tile.sav'

    if not os.path.isfile(filename):

        #Read field survey data
        if khuong:
            training_df = pd.read_csv(training_file)
            crop = training_df['crop']
            Y = training_df['crop_code'].values
            X = training_df.iloc[:,3:8].values

            # print(training_df.head(10))
            print(X.shape)
            print(Y.shape)

            #arrange into 93 fields and 28 dates
            X_all = []
            Y_all = []
            for i in range(0,len(X),93*256): ## 93 polygons with 16x16 (256) chip in each polygon
                #print(i,i+93*256,X[i:i+93*256,:].shape)
                X_all.append(X[i:i+93*256,:])
                
            X = np.concatenate((X_all),axis=1)
            Y=Y[:93*256]

            print(X.shape)
            print(Y.shape)

            print(np.unique(Y, return_counts=True))

            class_number=2 # corn and soybeans
            test_size=0.2
            random_state = 42

            X_train = [] 
            y_train = [] 
            X_test = []    
            y_test = []

            croptypes=['corn','soybean']
            for i in range(class_number):
                Y_class = Y[Y==i+1] #corn=1 and soybean=2
                X_class = X[Y==i+1]

                # X_class[np.isnan(X_class)] = 0
                
                X_train_class, X_test_class, y_train_class, y_test_class = \
                    train_test_split(X_class, Y_class, test_size=test_size, random_state=random_state)
                if i+1 == 1:
                    print("corn",y_train_class.shape,\
                        X_train_class.shape)
                else:
                    print("soybean",y_train_class.shape,\
                        X_train_class.shape)
            
                X_train.append(X_train_class)
                X_test.append(X_test_class)
                y_train.append(y_train_class)
                y_test.append(y_test_class)
            
            # Merge multiple classes      
            X_train_all = np.concatenate(X_train, axis=0)
            y_train_all = np.concatenate(y_train, axis=0).astype('uint8')
            X_test_all = np.concatenate(X_test, axis=0)    
            y_test_all = np.concatenate(y_test, axis=0).astype('uint8')

        else:

            # training_df = pl.read_csv(training_file)
            # training_df = pd.read_csv(training_file)

            training_df_1 = pd.read_csv(training_file[0])
            training_df_2 = pd.read_csv(training_file[1])

            ## depends on the data
            training_df_1 = training_df_1.replace(0,6)
            training_df_1 = training_df_1.replace(1,0)
            training_df_1 = training_df_1.replace(5,1)
            training_df_1 = training_df_1.replace(2,7)
            training_df_1 = training_df_1.replace(4,3)
            training_df_1 = training_df_1.replace(6,2)
            training_df_1 = training_df_1.replace(7,6)
            
            print(training_df_1['class'].unique())
            print(training_df_2['class'].unique())

            frames = [training_df_1, training_df_2]

            training_df = pd.concat(frames)

            print(training_df['class'].unique())
            
            Y = training_df['class'].values
            # X = training_df.iloc[:,3:8].values
            X = training_df.iloc[:,2:27].values

            print(X.shape)
            print(Y.shape)

            # class_number=6 # other, corn, soybean, fall-crop, other-crop, water
            # class_number=5 # other, corn, soybean, fall-crop, water
            test_size=0.2
            random_state = 42

            X_train = [] 
            y_train = [] 
            X_test = []    
            y_test = []

            class_types=training_df['class'].unique()
            for i in class_types:
                Y_class = Y[Y==i] # other=0, corn=1, soybean=2, fall-crop=3, other-crop=4, water=5
                X_class = X[Y==i]

                # X_class[np.isnan(X_class)] = -10000
                
                X_train_class, X_test_class, y_train_class, y_test_class = \
                    train_test_split(X_class, Y_class, test_size=test_size, random_state=random_state)

                print(i,y_train_class.shape,X_train_class.shape)

                X_train.append(X_train_class)
                X_test.append(X_test_class)
                y_train.append(y_train_class)
                y_test.append(y_test_class)
            
            # Merge multiple classes      
            X_train_all = np.concatenate(X_train, axis=0)
            y_train_all = np.concatenate(y_train, axis=0).astype('uint8')
            X_test_all = np.concatenate(X_test, axis=0)    
            y_test_all = np.concatenate(y_test, axis=0).astype('uint8')

        X_train = X_train_all
        L_train = y_train_all
        X_test = X_test_all
        L_test = y_test_all

        print("X_train shape: ", X_train.shape)
        print("L_train shape: ", L_train.shape)
        print("X_test shape: ", X_test.shape)
        print("L_test shape: ", L_test.shape)

        # run training for random forest model
        # class_weights = [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 5}]
        class_weights = {0:1, 1:5, 2:1, 3:1, 4:5, 5:1}

        model_RF = RandomForestClassifier(n_estimators=100, criterion='entropy', class_weight=class_weights, random_state=42, n_jobs=4)
        # model_RF = RandomForestClassifier()
        model_RF.fit(X_train,L_train)
        L_pred_train = model_RF.predict(X_train)
        print('Model Training accuracy: % .3f' % accuracy_score(L_train, L_pred_train))
        print('Model Training kappa: % .3f' % cohen_kappa_score(L_train, L_pred_train))
        print('Model Training f-score: % .3f' % f1_score(L_train, L_pred_train, average = 'weighted'))
        cm_training = confusion_matrix(L_train, L_pred_train)
        print(cm_training)
        # plt.clf()
        # plt.figure(figsize=(7, 6))
        # sns.heatmap(cm_testing, cmap = 'jet',annot=True,fmt='d')
        # tick_marks = np.arange(len(croptypes))+0.5
        # plt.xticks(tick_marks,croptypes)
        # plt.yticks(tick_marks,croptypes)
        # plt.xlabel('Predicted label\nAccuracy={:0.2f}'.format(accuracy_score(L_train, L_pred_train)))
        # plt.ylabel('True')

        # performance on test set
        L_pred_test = model_RF.predict(X_test)
        print('Model Testing accuracy: % .3f' % accuracy_score(L_test, L_pred_test))
        print('Model Testing kappa: % .3f' % cohen_kappa_score(L_test, L_pred_test))
        print('Model Testing f-score: % .3f' % f1_score(L_test, L_pred_test, average = 'weighted'))
        cm_testing = confusion_matrix(L_test, L_pred_test)
        print(cm_testing)
        # plt.clf()
        # plt.figure(figsize=(7, 6))
        # sns.heatmap(cm_testing, cmap = 'jet',annot=True,fmt='d')
        # tick_marks = np.arange(len(croptypes))+0.5
        # plt.xticks(tick_marks,croptypes)
        # plt.yticks(tick_marks,croptypes)
        # plt.xlabel('Predicted label\nAccuracy={:0.2f}'.format(accuracy_score(L_test, L_pred_test)))
        # plt.ylabel('True')

        # save model to disk
        pickle.dump(model_RF, open(filename, 'wb'))
        print("Finished Training Random Forest!")
    else:
        model_RF = pickle.load(open(filename, 'rb'))
        print("Model loaded from file!")

    return model_RF

def prepare_ts(ts_file, im_size, scale=False):

    big_df = pl.read_csv(ts_file, has_header=True)
    X_im = big_df.select(['blue','green','red','nir','ndvi'])
    X_im = X_im.to_pandas()

    # X_im = pd.read_csv(ts_file, header=0)
    # X_im = X_im[['blue','green','red','nir','nvdi']]

    # X_im = X_im.replace(0, -10000)
    X_array = X_im.to_numpy()
    print("X_array shape: ", X_array.shape)

    X_lst = []
    for i in range(0,len(X_array),im_size*im_size):
        X_lst.append(X_array[i:i+(im_size*im_size),:])
        
    X_out = np.concatenate((X_lst),axis=1)
    # X_out[np.isnan(X_out)] = -10000
    print("X_out shape: ", X_out.shape)

    return X_out

def predict_image(model_RF, X_array, start_hidx: int, start_widx: int, im_size: int):

    pl_file = '/home/geoint/tri/Planet_khuong/08-21/files/PSOrthoTile/4854347_1459221_2021-08-31_241e/analytic_sr_udm2/4854347_1459221_2021-08-31_241e_BGRN_SR.tif'

    pl_file = '/home/geoint/tri/Planet_khuong/Tile1459222_Aug2021_psorthotile_analytic_sr_udm2/PSOrthoTile/4753640_1459222_2021-08-01_2421_BGRN_SR.tif'

    planet_data = tifffile.imread(pl_file)
    # print(planet_data.shape)

    # X_array = X_array.replace(0, -10000)

    im_pred = model_RF.predict(X_array)
    im_out = im_pred.reshape((im_size,im_size))

    # colors = ['orange', 'green']
    # colormap = pltc.ListedColormap(colors)

    # plt.figure(figsize=(20,20))
    # plt.subplot(1,2,1)
    # plt.imshow(rescale_truncate(\
    #     rescale_image(planet_data[start_hidx:(start_hidx+im_size),start_widx:(start_widx+im_size),:3])))
    # plt.subplot(1,2,2)
    # plt.imshow(im_out, cmap=colormap)
    # plt.savefig(f'./output/random_forest_prediction-tile01-{im_size}-{start_hidx}-{start_widx}.png')
    # # plt.show()
    # plt.close()

    # del planet_data

    return im_out


if __name__ =='__main__':

    
    # ts_file = 'output/csv/dpc-unet-pixel-all-2000-2000_2000.csv'

    khuong = False

    print("Start the python scripts!")

    if khuong:
        training_file = 'dpc-unet-pixel_rearrranged_kt.csv' ## khuong;s file
    else:
        # training_file = 'dpc-unet-pixel-label-label.csv'
        # training_file = 'dpc-unet-pixel-label-label-0529.csv'
        # training_file = 'dpc-unet-pixel-label-200perPoly-label-0703.pkl'
        # training_file = 'dpc-unet-pixel-tile01-label-32x32-label-0802.csv'
        training_file_t2 = '/home/geoint/tri/Planet_khuong/output/csv/training-pixel-GM-tile02-label-2000p-label-0817.csv'

        training_file_t1 = '/home/geoint/tri/Planet_khuong/output/csv/training-pixel-GM-tile01-label-2000p-label-0817.csv'

    # with open(training_file, 'rb') as file:
    #     df = pickle.load(file)

    # df = pl.read_csv(training_file)
    # print(df.shape)
    # print(df.head(5))

    # search_term = re.search(r'all.(.*\d*).csv', ts_file).group(1)
    # search_widx = re.search(r'\d_(\d*)', search_term).group(1)
    # search_hidx = re.search(r'\d-(\d*)', search_term).group(1)

    im_size=2000
    tile="tile02"
    scale=False
    if khuong:
        model=train_random_forest([training_file], khuong, scale = scale)
    else:
        model=train_random_forest([training_file_t1,training_file_t2], khuong, scale = scale)
    
    if tile == "tile01":
        files_lst = sorted(glob.glob('output/csv/tile01/*.csv'))
    elif tile=="tile02":
        files_lst = sorted(glob.glob('output/csv/tile02/*.csv'))

    # X_array = prepare_ts(ts_file, im_size=im_size)
    # predict_image(model, X_array,start_hidx=int(search_hidx), start_widx=int(search_widx), im_size=im_size)

    for ts_file in files_lst:
        search_term = re.search(r'all.(.*\d*).csv', ts_file).group(1)
        search_widx = re.search(r'\d_(\d*)', search_term).group(1)
        search_hidx = re.search(r'\d-(\d*)', search_term).group(1)

        print(ts_file)
        X_array = prepare_ts(ts_file, im_size=im_size, scale=scale)
        output = predict_image(model, X_array,start_hidx=int(search_hidx), \
                               start_widx=int(search_widx), im_size=im_size)
        
        np.save(f'output/rf_out/{tile}/rf-{tile}-{search_hidx}_{search_widx}.npy', output)