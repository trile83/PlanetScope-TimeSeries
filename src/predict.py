import time
import re
import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from einops import rearrange
import json
import os
import glob
from torch import nn
from utils.augmentation import *
from utils.utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy
from tqdm import tqdm
import argparse
import h5py
import logging
import csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
import rioxarray as rxr
import rasterio
import xarray as xr
from inference import inference
from tensorboardX import SummaryWriter
from benchmod.convlstm import ConvLSTM_Seg, BConvLSTM_Seg
from benchmod.convgru import ConvGRU_Seg

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='unet', type=str, help='encoder for the DPC')
parser.add_argument('--model', default='convgru', type=str, help='convlstm, dpc-unet, unet')
parser.add_argument('--dataset', default='tile01', type=str, help='tile01, tile02')
parser.add_argument('--seq_len', default=6, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=4, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=64, type=int)
parser.add_argument('--ts_length', default=10, type=int)
parser.add_argument('--pad_size', default=0, type=int)
parser.add_argument('--num_classes', default=4, type=int)


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
        for i in range(image.shape[0]):
            image[i, :, :] = (image[i, :, :] - mean[i]) / (std[i] + 1e-8)
    elif standardization_type == 'mixed':
        raise NotImplementedError
    return image


class satDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y):
        'Initialization'
        self.data = X
        self.mask = Y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.mask

        return {
            'x': X,
            'mask': Y
        }

class tsDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y):
        'Initialization'
        self.data = X
        self.mask = Y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.mask[index]

        return {
            'ts': torch.tensor(X),
            'mask': torch.LongTensor(Y)
        }


class segmentDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, Y, Z):
        'Initialization'
        self.data = X
        self.mask = Y
        self.imge = Z
        # self.transforms = transforms.ToTensor()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.mask[index]
        Z = self.imge[index]

        return {
            'x': X,
            'mask': Y,
            'ts': Z
        }

def get_seq(sequence, seq_length):
    '''
    sequence: long time-series input
    seq_length: length of window for time-series chunk; default = 5
    return: array with all time-series chunk; prediction size =1, num_seq = 4
    '''
    (I,L,C,H,W) = sequence.shape
    all_arr = np.zeros((I,L-seq_length+1,seq_length,C,H,W))
    for j in range(I):
        for i in range(seq_length, L+1):
            array = sequence[j,i-seq_length:i,:,:,:] # SL, C, H, W
            all_arr[j,i-seq_length] = array

    return all_arr

def get_chunks(windows, num_seq):
    '''
    TODO: match with get_seq function
    windows: number of windows in 1 time-series
    number: number of window for time-series chunk; default = 4
    return: array with all time-series chunk; prediction size =1, num_seq (N) = 4; N x 6 x 5 x 32 x 32
    '''
    (I,L1,SL,C,H,W) = windows.shape
    all_arr = np.zeros((I,L1-num_seq+1,num_seq,SL,C,H,W))
    for j in range(I):
        for i in range(num_seq, L1+1):
            array = windows[j,i-num_seq:i,:,:,:,:] # N, SL, C, H, W
            # if not array.any():
            #     print(f"i {i}")
            all_arr[j,i-num_seq] = array

    return all_arr

def reverse_chunks(chunks, num_seq):
    '''
    reverse the chunk code -> to window size
    '''
    (I,L2,N,SL,C,H,W) = chunks.shape
    
    all_arr = np.zeros((I,L2+num_seq-1,SL,C,H,W))
    for j in range(I):
        for i in range(L2):
            if i < L2-1:
                array = chunks[j,i,0,:,:,:,:] # L2, N, SL, C, H, W
                all_arr[j,i,:,:,:,:] = array
            elif i == L2-1:
                array = chunks[j,i,:,:,:,:,:]
                all_arr[j,i:i+num_seq,:,:,:,:] = array
            del array

    return all_arr

def reverse_seq(window, seq_length):
    '''
    reverse the chunk code -> to window size
    '''
    (I,L1,SL,C,H,W) = window.shape
    all_arr = np.zeros((I,L1+seq_length-1,C,H,W))
    for j in range(I):
        for i in range(L1):
            if i < L1-1:
                array = window[j,i,0,:,:,:] # L2, N, SL, C, H, W
                all_arr[j,i,:,:,:] = array
            elif i == L1-1:
                array = window[j,i,:,:,:,:]
                all_arr[j,i:i+seq_length,:,:,:] = array
            del array

    return all_arr


def get_accuracy(y_pred, y_true):

    target_names = ['non-crop','cropland']

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # get overall weighted accuracy
    accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    precision = report['cropland']['precision']
    recall = report['cropland']['recall']
    f1_score = report['cropland']['f1-score']
    return accuracy, precision, recall, f1_score

def read_imagery(pl_file, mask=False):

    img_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    ref_im = rxr.open_rasterio(pl_file)

    # if mask:
    #     img_data[img_data==3] = 0
    #     img_data[img_data==4] = 3
    #     img_data[img_data==5] = 4

    #     return img_data
    
    if mask:
        img_data = np.nan_to_num(img_data, nan=0.0)

        print(np.unique(img_data))

        return img_data

    # if img_data.ndim > 2:
    #     img_data = np.transpose(img_data, (1,2,0))

    else:

        return img_data, ref_im



def read_dataset(tile_name='tile01'):
    data_dir = '/home/geoint/tri/Planet_khuong/'

    
    if tile_name == 'tile01':
        master_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/output/median_composite/*_median_composit.tiff'))
        label_fl=f'{data_dir}output/training-data/label-tile01-0802.tif'

        data_ts = []
        for monthly_fl in master_dir:

            print(monthly_fl)

            img, ref_im = read_imagery(monthly_fl, mask=False)
            data_ts.append(img)

            print("img shape: ", img.shape)

    out_ts = np.stack(data_ts, axis=0)

    del data_ts

    label = read_imagery(label_fl, mask=True)

    print(np.unique(label, return_counts=True))

    print('out ts shape: ', out_ts.shape)
    print('label shape: ', label.shape)

    return out_ts, label, rxr.open_rasterio(label_fl)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    # prepare data
    ##### REMEMBER TO CHECK IF THE IMAGE IS CHIPPED IN THE NO-DATA REGION, MAKE SURE IT HAS DATA.
    ts_name=args.dataset
    ts_arr, mask_arr, ref_im = read_dataset(tile_name=ts_name)

    input_size = args.img_dim
    total_ts_len = args.ts_length # L

    padding_size = args.pad_size

    train_ts_set = ts_arr[:total_ts_len,:,:,:]

    print(f'data dict {ts_name} ts shape: {train_ts_set.shape}')
    print(f'data dict {ts_name} mask shape: {mask_arr.shape}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_option = args.model # 'dpc-unet' and 'unet', convlstm

    if model_option == 'convlstm':
        model_dir = "/home/geoint/tri/Planet_khuong/output/checkpoints/"
        model = ConvLSTM_Seg(
            num_classes=args.num_classes,
            input_size=(input_size,input_size),
            hidden_dim=160,
            input_dim=4,
            kernel_size=(3, 3)
            )
 
        model = nn.DataParallel(model)

        model_checkpoint = f'{str(model_dir)}convlstm_planet_4band_0802_epoch_63.pth'
        if torch.cuda.is_available():
            model = model.to(cuda)

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        xraster = train_ts_set
        temporary_tif = xr.where(xraster > 0, xraster, 50) # 2000 is the optimal value for the nodata
 
        # temporary_tif = xr.where(xraster > -9000, xraster, 120)
        # temporary_tif = rescale_image(temporary_tif)

        prediction = inference.sliding_window_tiler(
            xraster=temporary_tif,
            model=model,
            n_classes=args.num_classes,
            overlap=0.5,
            batch_size=16,
            standardization='local',
            mean=0,
            std=0,
            normalize=10000.0,
            rescale=None,
            im_type='planet',
            model_option=model_option
        )

    elif model_option == 'biconvlstm':
        model_dir = "/home/geoint/tri/Planet_khuong/output/checkpoints/"
        model = BConvLSTM_Seg(
            num_classes=args.num_classes,
            input_size=(input_size,input_size),
            hidden_dim=160,
            input_dim=4,
            kernel_size=(3, 3)
            )
 
        model = nn.DataParallel(model)

        model_checkpoint = f'{str(model_dir)}biconvlstm__planet_4band_epoch_53.pth'
        if torch.cuda.is_available():
            model = model.to(cuda)

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        xraster = train_ts_set
        temporary_tif = xr.where(xraster > 0, xraster, 50) # 2000 is the optimal value for the nodata
 
        # temporary_tif = xr.where(xraster > -9000, xraster, 120)
        # temporary_tif = rescale_image(temporary_tif)

        prediction = inference.sliding_window_tiler(
            xraster=temporary_tif,
            model=model,
            n_classes=args.num_classes,
            overlap=0.5,
            batch_size=16,
            standardization='local',
            mean=0,
            std=0,
            normalize=10000.0,
            rescale=None,
            im_type='planet',
            model_option=model_option
        )

    elif model_option == 'convgru':
        model_dir = "/home/geoint/tri/Planet_khuong/output/checkpoints/"
        model = ConvGRU_Seg(
                num_classes=args.num_classes,
                input_size=(input_size,input_size),
                input_dim=4,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
        
        model = nn.DataParallel(model)
 
        ### 10 bands
        model_checkpoint = f'{str(model_dir)}convgru_planet_4band_0802_epoch_58.pth'
        if torch.cuda.is_available():
            model = model.to(cuda)

        model.load_state_dict(torch.load(model_checkpoint)['state_dict'])

        print('Finish loading model weights!!')

        xraster = train_ts_set
        temporary_tif = xr.where(xraster > 0, xraster, 50) # 2000 is the optimal value for the nodata
 
        # temporary_tif = xr.where(xraster > -9000, xraster, 120)
        # temporary_tif = rescale_image(temporary_tif)

        prediction = inference.sliding_window_tiler(
            xraster=temporary_tif,
            model=model,
            n_classes=args.num_classes,
            overlap=0.5,
            batch_size=16,
            standardization='local',
            mean=0,
            std=0,
            normalize=10000.0,
            rescale=None,
            im_type='planet',
            model_option=model_option
        )

    ref_im = ref_im.transpose("y", "x", "band")

    if prediction.shape[0] > 1:
        prediction = np.argmax(prediction, axis=0)
    else:
        prediction = np.squeeze(
            np.where(prediction > 0.5, 1, 0).astype(np.int16)
        )

    print('prediction shape after final process: ', prediction.shape)

    data_dir = '/home/geoint/tri/Planet_khuong/output/'

    # plt.figure(figsize=(20,20))
    # plt.subplot(1,2,1)
    # plt.title("Image")
    # image = np.transpose(train_ts_set[5,:3,:,:], (1,2,0))
    
    # image= rescale_image(xr.where(image > 0, image, 100))
    # # image = np.transpose(z_mean[0,:,:,:], (1,2,0))
    # plt.imshow(rescale_truncate(image))
    # # # plt.savefig(f"{str(data_dir)}{ts_name}-input.png")
    # # plt.subplot(1,3,2)
    # # plt.title("Segmentation Label")
    # # # image = np.transpose(train_mask_set[:,:], (0,1))
    # # image = train_mask_set
    # # plt.imshow(image)
    # # # plt.savefig(f"{str(data_dir)}{ts_name}-label.png")

    # plt.subplot(1,2,2)
    # plt.title(f"Segmentation Prediction")
    # image = prediction
    # plt.imshow(image[4000:5000,4000:5000])
    # plt.savefig(f"{str(data_dir)}{ts_name}-{model_option}-4000-4000.png", dpi=300, bbox_inches='tight')
    # plt.close()

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

    prediction.attrs['long_name'] = ('conv')
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
        f'{data_dir}{ts_name}-{model_option}-0802.tiff',
        BIGTIFF="IF_SAFER",
        compress='LZW',
        # num_threads='all_cpus',
        driver='GTiff',
        dtype='uint8'
    )


def predict_dpc(data_loader, dpc_model):

    dpc_model.eval()
    global iteration

    feature_lst = []

    for idx, input in enumerate(data_loader):
        tic = time.time()
        input_seq = input["x"]

        (B,N,SL,C,H,W) = input_seq.shape
        input_seq = input_seq.to(cuda, dtype=torch.float32)
        B = input_seq.size(0)
        features = dpc_model(input_seq)
        feature_lst.append(features)

    feature_arr = torch.cat(feature_lst, dim=0)

    return feature_arr.cpu().detach()


if __name__ == '__main__':
    main()

    # python models/predict_sliding.py --gpu 0 --model convlstm --dataset Tappan13
    # python models/predict_sliding.py --gpu 0 --model convgru --dataset Tappan13
    # python models/predict_sliding.py --gpu 0 --model dpc-unet --net unet-vae --dataset Tappan16
