import re
import torch
from torch import nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from einops import rearrange
import os
import glob
from benchmod.convlstm import ConvLSTM_Seg, BConvLSTM_Seg
from benchmod.convgru import ConvGRU_Seg
from tqdm import tqdm
import argparse
import h5py
import logging
import cv2
import geopandas as gpd
import rasterio as rio
import shapely
import rasterio.mask as mask
import rioxarray as rxr
from tensorboardX import SummaryWriter
import json
from utils.utils import AverageMeter, save_checkpoint

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='convlstm', type=str, help='convlstm, convgru')
parser.add_argument('--dataset', default='tile01', type=str, help='tile01, tile02')
parser.add_argument('--seq_len', default=6, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=4, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=3e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=16, type=int)
parser.add_argument('--ts_length', default=15, type=int)
parser.add_argument('--pad_size', default=0, type=int)
parser.add_argument('--num_chips', default=800, type=int)
parser.add_argument('--num_val', default=10, type=int)
parser.add_argument('--num_classes', default=3, type=int)


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
    def __init__(self, X, Y):
        'Initialization'
        self.data = X
        self.mask = Y
        # self.transforms = transforms.ToTensor()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        Y = self.mask[index]

        return {
            'x': X,
            'mask': Y
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

        # for i in range(L-seq_length): # same results
        #     array = sequence[j,i:i+seq_length,:,:,:] # SL, C, H, W
        #     all_arr[j,i] = array

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
            if not array.any():
                print(f"i {i}")
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

    # for j in range(I):
    #     all_arr[j,L2+num_seq-1,:,:,:,:] = chunks[j,L2-1,-1,:,:,:,:]

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

def chipper(ts_stack, mask, input_size=32):
    '''
    stack: input time-series stack to be chipped (TxCxHxW)
    mask: ground truth that need to be chipped (HxW)
    input_size: desire output size
    ** return: output stack with chipped size
    '''
    t, c, h, w = ts_stack.shape

    i = np.random.randint(0, h-input_size)
    j = np.random.randint(0, w-input_size)
    
    out_ts = np.array([ts_stack[:, :, i:(i+input_size), j:(j+input_size)]])
    out_mask = np.array([mask[i:(i+input_size), j:(j+input_size)]])

    # print(out_ts.shape)

    return out_ts, out_mask

def specific_chipper(ts_stack, mask, h_index, w_index, input_size=32):
    '''
    stack: input time-series stack to be chipped (TxCxHxW)
    mask: ground truth that need to be chipped (HxW)
    input_size: desire output size
    ** return: output stack with chipped size
    '''
    t, c, h, w = ts_stack.shape

    i = h_index
    j = w_index
    
    out_ts = np.array([ts_stack[:, :, i:(i+input_size), j:(j+input_size)]])
    out_mask = np.array([mask[i:(i+input_size), j:(j+input_size)]])

    return out_ts, out_mask


def padding_ts(ts, mask, padding_size=10):
    '''
    Args:
        ts: time series input
        mask: ground truth
    Return:
        padded_ts
        padded_mask
    '''
    extra_top = extra_bottom = extra_left = extra_right = padding_size
    npad_ts = ((0, 0), (extra_top, extra_bottom), (extra_left, extra_right))
    npad_mask = ((extra_top, extra_bottom), (extra_left, extra_right))

    padded_ts = np.zeros((ts.shape[0],ts.shape[1],ts.shape[2]+padding_size*2,ts.shape[3]+padding_size*2))
    for i in range(ts.shape[0]):
        # pad border

        p_ts_i = np.copy(np.pad(ts[i], (npad_ts), mode='reflect'))

        padded_ts[i,:,:,:] = p_ts_i

        # print('padded_ts i',padded_ts.shape)

    plt.imshow(np.transpose(padded_ts[5,1:4,:,:], (1,2,0)))
    plt.savefig('/home/geoint/tri/dpc_test_out/train_input_im.png')
    plt.close()

    padded_mask = np.copy(np.pad(mask, (npad_mask), mode='constant', constant_values = 0))
    padded_mask = padded_mask.reshape((padded_mask.shape[0], padded_mask.shape[1]))

    del p_ts_i

    return padded_ts, padded_mask

def get_field_data(field_file, pl_file):

    vector = gpd.read_file(field_file)

    # print(vector)

    out_rast = {}
    out_mask = {}

    #save output files as per shapefile features
    for i in range(len(vector)):
        
        with rio.open(pl_file) as src:
            # read imagery file
            vector=vector.to_crs(src.crs)
            geom = []
            coord = shapely.geometry.mapping(vector)["features"][i]["geometry"]
            crop_type = vector["Crop_types"][i]
            
            geom.append(coord)
            out_image, out_transform = mask.mask(src, geom, crop=True)
            full_image, full_transform = mask.mask(src, geom, crop=True, filled=False)

            # print('max of full image: ', np.max(full_image))
            # print('min of full image: ', np.min(full_image))

            # print(out_image.shape)
            # # Check that after the clip, the image is not empty
            # test = out_image[~np.isnan(out_image)]
            # if test[test > 0].shape[0] == 0:
            #     raise RuntimeError("Empty output")

            out_meta = src.profile
            out_meta.update({"height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform})

            # out_rast.append(out_image)

            if i not in out_rast.keys():
                out_rast[i] = []
            if i not in out_mask.keys():
                out_mask[i] = []

            out_rast[i].append(out_image)

            # print(f'min of out image: {np.min(out_image)}')

            if crop_type == 'corn':
                nodata_mask = np.where(out_image[0,:,:]>0,1,0)
                label = nodata_mask
                # print(label.shape)
                out_mask[i].append(label)
            elif crop_type == 'soybean':
                nodata_mask = np.where(out_image[0,:,:]>0,1,0)
                nodata_mask[nodata_mask==1] = 2
                label = nodata_mask
                out_mask[i].append(label)
            else:
                label = np.zeros((out_image.shape[1],out_image.shape[2]))
                out_mask[i].append(label)

            # print(f'crop {crop_type} unique label {np.unique(label)}')

            if full_image.ndim > 2:
                save_image = np.transpose(full_image, (1,2,0))

            # plt.figure(figsize=(20,20))
            # plt.subplot(1,2,1)
            # # plt.imshow(rescale_truncate(rescale_image(save_image[:,:,:3])))
            # plt.imshow(save_image[:,:,:3]/11000)
            # plt.subplot(1,2,2)
            # # plt.imshow(label)
            # plt.imshow(label)
            # plt.savefig(f'/home/geoint/tri/Planet_khuong/output/raster-polygon-{crop_type}-{i}.png', dpi=300, bbox_inches='tight')
            # plt.close()

            del label
            del nodata_mask

    return out_rast, out_mask

def get_raster_from_polygon(field_file, tile_name):

    data_dir = '/home/geoint/tri/Planet_khuong/'
    ts_dict = {}
    ma_dict = {}
    if tile_name == 'tile01':
        master_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/*-21/'))
        label_fl=f'{data_dir}/output/4906044_1459221_2021-09-16_2447_BGRN_SR_mask_segs_reclassified.tif'
        # data_ts = []
        # mask_ts = []
        for monthly_dir in master_dir:
            month = monthly_dir[-7:-1]
            pl_dir = f'{str(monthly_dir)}/files/PSOrthoTile/'
            img_fls = sorted(glob.glob(f'{pl_dir}/*/'))
            count=0
            for img_dir in img_fls:
                if count == 6:
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
                    rast, mask = get_field_data(field_file, fl[0])
                    if date not in ts_dict.keys():
                        ts_dict[date] = rast
                    if date not in ma_dict.keys():
                        ma_dict[date] = mask

                    count+=1

    ts_arr, ma_arr = stack_dict(ts_dict, ma_dict)

    return ts_arr, ma_arr

def stack_dict(ts_dict, ma_dict):

    im_dict = {}
    ts_arr = []
    ma_arr = []
    for date in ts_dict.keys():
        for i in ts_dict[date].keys():
            if i not in im_dict.keys():
                im_dict[i] = []

            im_dict[i].append(ts_dict[date][i])

    for date in ma_dict.keys():
        for i in ma_dict[date].keys():
            b = ma_dict[date][i][0]
            # print("length b", len(b))
            # print("b", b)
            # print("b shape: ", b.shape)
            ma_arr.append(ma_dict[date][i][0])

            del b
        break

    for key in im_dict.keys():
        a = np.stack(im_dict[key], axis=0)
        a = np.squeeze(a)
        # print('a shape: ', a.shape)
        ts_arr.append(a)

        del a

    # print("ma arr length: ", len(ma_arr))

    return ts_arr, ma_arr

def get_data(out_raster, out_mask, num_chips=1, input_size=32):

    temp_ts_set = []
    temp_mask_set = []

    i=0
    count=0
    while i < num_chips:
        # print('out raster shape: ', out_raster.shape)
        ts, mask = chipper(out_raster, out_mask, input_size=input_size)
        ts = np.squeeze(ts)
        mask = np.squeeze(mask)
        
        if count == 5:
            break
        if np.any(ts==0):
            count += 1 
            continue

        # print(f'min ts ',np.min(ts))
        # print(f'max ts ',np.max(ts))
        temp_ts_set.append(ts)
        temp_mask_set.append(mask)

        i += 1

    if len(temp_ts_set) > 0:
        return temp_ts_set, temp_mask_set
    else:
        return [], []

    
def read_imagery(pl_file, mask=False):

    img_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    ref_im = rxr.open_rasterio(pl_file)

    if mask:
        img_data[img_data==3] = 0
        img_data[img_data==4] = 3
        img_data[img_data==5] = 4
        img_data[img_data==6] = 5

    # if img_data.ndim > 2:
    #     img_data = np.transpose(img_data, (1,2,0))

    return img_data

def read_udm_mask(cloud_fl):

    cloud_data = np.squeeze(rxr.open_rasterio(cloud_fl, masked=True).values)
    # mask_data = np.where(cloud_data[0]==1,True,False)

    return cloud_data

def read_json(json_fl):
    
    with open(json_fl, 'r') as f:
        json_data = json.load(f)

    return json_data

def read_dataset(tile_name='tile01'):
    data_dir = '/home/geoint/tri/Planet_khuong/'
    if tile_name == 'tile01':
        master_dir = sorted(glob.glob('/home/geoint/tri/Planet_khuong/*-21/'))
        label_fl=f'{data_dir}/output/4906044_1459221_2021-09-16_2447_BGRN_SR_mask_segs_reclassified.tif'

        data_ts = []
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

    out_ts = np.stack(data_ts, axis=0)

    label = read_imagery(label_fl, mask=True)

    print(np.unique(label, return_counts=True))

    print('out ts shape: ', out_ts.shape)
    print('label shape: ', label.shape)

    return out_ts, label


def main():
    #Loading original image
    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')


    pl_file = \
            '/home/geoint/tri/Planet_khuong/08-21/files/PSOrthoTile/4854347_1459221_2021-08-31_241e/analytic_sr_udm2/4854347_1459221_2021-08-31_241e_BGRN_SR.tif'
    field_fl = '/home/geoint/tri/Planet_khuong/Field_Survey_Polygons/Field_Survey_Polygons.shp'

    name = pl_file[-43:-4]
    planet_data = np.squeeze(rxr.open_rasterio(pl_file, masked=True).values)
    ref_im = rxr.open_rasterio(pl_file)

    if planet_data.ndim > 2:
        planet_data = np.transpose(planet_data, (1,2,0))

    # ts_arr, mask_arr = read_dataset(tile_name='tile01')

    # size = 8000
    # originImg = planet_data[:size,:size,:]

    # out_raster, out_mask = get_field_data(field_fl, pl_file)

    out_raster, out_mask = get_raster_from_polygon(field_fl, tile_name='tile01')

    # print("length out raster: ", len(out_raster))
    # print("length out mask: ", len(out_mask))

    im_lst = []
    ma_lst = []
    for idx in range(len(out_raster)):
        if np.count_nonzero(out_raster[idx]) > .25*out_raster[idx].shape[1]*out_raster[idx].shape[2]:
            im, mask = get_data(out_raster[idx], out_mask[idx],input_size=16)

            # im = np.squeeze(im)
            # mask = np.squeeze(mask)

            if len(im) > 0:
                # print("im 0 shape: ", im[0].shape)
                # print("mask 0 shape: ", mask[0].shape)
                im_lst.append(im)
                ma_lst.append(mask)


    print(len(im_lst))
        
    ts_arr = np.stack(im_lst, axis=0)
    ma_arr = np.stack(ma_lst, axis=0)

    ts_arr = np.squeeze(ts_arr, axis=1)
    ma_arr = np.squeeze(ma_arr, axis=1)

    print(f'ts shape: {ts_arr.shape}')
    print(f'mask shape: {ma_arr.shape}')

    train_ts_set = ts_arr
    train_mask_set = ma_arr

    for frame in range(len(train_ts_set)):
        train_ts_set[frame] = standardize_image(rescale_image(train_ts_set[frame]),'local')

    train_ts_set = np.array(train_ts_set, dtype=np.float16)

    del ts_arr
    del ma_arr

    print(f"train ts set shape: {train_ts_set.shape}")
    print(f"train mask set shape: {train_mask_set.shape}")

    ### dpc model ###

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Load model')

    model_option = args.model
    if model_option == "convlstm":
        model = ConvLSTM_Seg(
            num_classes=args.num_classes,
            input_size=(args.img_dim,args.img_dim),
            hidden_dim=160,
            input_dim=4,
            kernel_size=(3, 3)
            )
    elif model_option == "convgru":
            model = ConvGRU_Seg(
                num_classes=args.num_classes,
                input_size=(args.img_dim,args.img_dim),
                input_dim=4,
                kernel_size=(3, 3),
                hidden_dim=180,
            )

    elif model_option == "biconvlstm":
            model = BConvLSTM_Seg(
                num_classes=args.num_classes,
                input_size=(args.img_dim,args.img_dim),
                hidden_dim=160,
                input_dim=4,
                kernel_size=(3, 3)
            )

    print(f'Finish loading {model_option} model!')

    model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.to(cuda)

    global criterion; 
    criterion_type = "crossentropy"
    if criterion_type == "crossentropy":
        # criterion = models.losses.MultiTemporalCrossEntropy()
        criterion = criterion = torch.nn.CrossEntropyLoss()

    ### optimizer ###
    # params = model.parameters()
    # optimizer = optim.Adam(params, lr=0.0001, weight_decay=0.0)

    segment_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    args.old_lr = None

    ### load data ###

    print("Start training process!")

    # setup tools
    # global de_normalize; de_normalize = denorm()
    global img_path
    model_dir = "/home/geoint/tri/Planet_khuong/output/checkpoints/"
    
    ### main loop ###
    train_loss_lst = []
    val_loss_lst = []

    # print(f"train mask set shape: {train_mask_set.shape}")

    # train_set = train_ts_set[:-args.num_val]
    # train_mask = train_mask_set[:-args.num_val]
    # val_set = train_ts_set[-args.num_val:]
    # val_mask = train_mask_set[-args.num_val:]
    # print(f"train set shape: {train_set.shape}")
    # print(f"val set shape: {val_set.shape}")

    train_seg_set = tsDataset(train_ts_set[:-args.num_val],  train_mask_set[:-args.num_val])
    val_seg_set = tsDataset(train_ts_set[-args.num_val:],  train_mask_set[-args.num_val:])

    loader_args_1 = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    train_segment_dl = DataLoader(train_seg_set, **loader_args_1)
    val_segment_dl = DataLoader(val_seg_set, **loader_args_1)

    print(f"Length of segmentation input training set {len(train_segment_dl)}")
    print("Start segmentation training!")

    best_acc = 0
    min_loss = np.inf

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(train_segment_dl, model, segment_optimizer, epoch, model_option)
        val_loss = val(val_segment_dl, model, epoch, model_option)

        # saved loss value in list
        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)

        print(f"epoch {epoch+1}: train loss: {train_loss} & val loss: {val_loss}")

        # save check_point
        is_best = val_loss < min_loss

        if is_best:
            min_loss = val_loss

            # save unet segment weights
            save_checkpoint({'epoch': epoch+1,
                            'net': args.net,
                            'state_dict': model.state_dict(),
                            'min_loss': min_loss,
                            'optimizer': segment_optimizer.state_dict()}, 
                            is_best, filename=\
                                os.path.join(model_dir, \
                                    f'{model_option}__planet_4band_polygon_epoch_%s.pth' % str(epoch+1)), keep_all=False)

        
    plt.plot(train_loss_lst, color ="blue")
    plt.plot(val_loss_lst, color = "red")
    plt.savefig(f'/home/geoint/tri/Planet_khuong/output/train_output/{model_option}_train_polygon_loss.png')
    plt.close()

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def train(data_loader, segment_model, optimizer, epoch, model_option):
    losses = AverageMeter()
    segment_model.train()
    global iteration

    for idx, input in enumerate(data_loader):

        input_im = input['ts'].to(cuda, dtype=torch.float32)
        input_mask = input['mask'].to(cuda, dtype=torch.long)

        (B,L,F,H,W) = input_im.shape


        input_mask = input_mask.view(B,H,W)

        # print(f"features shape: {features.shape}")
        # print(f"mask shape: {input_mask.shape}")

        mask_pred = segment_model(input_im)

        # print(f"mask pred shape: {mask_pred.shape}")

        loss = criterion(mask_pred, input_mask)

        losses.update(loss.item(), B)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # print(mask_pred[0])
        # print(input_mask[0])

        # output predictions
        # index_array = torch.argmax(mask_pred, dim=1)

        # plt.figure(figsize=(20,20))
        # plt.subplot(1,3,1)
        # plt.title("Image")
        # x = input['ts']
        # y = input['mask']
        # image = np.transpose(x[0,5,:3,:,:].numpy(), (1,2,0))
        # # image = np.transpose(z_mean[0,:,:,:], (1,2,0))
        # # image = rescale_truncate(image)
        # plt.imshow(image)
        # # plt.savefig(f"{str(data_dir)}{ts_name}-{str(idx)}-input.png")
        # plt.subplot(1,3,2)
        # plt.title("Segmentation Label")
        # image = np.transpose(y[0,:,:], (0,1))
        # plt.imshow(image)
        # # plt.savefig(f"{str(data_dir)}{ts_name}-{str(idx)}-label.png")
        # plt.subplot(1,3,3)
        # plt.title(f"Segmentation Prediction")
        # image = np.transpose(index_array[0,:,:].cpu().numpy(), (0,1))
        # plt.imshow(image)
        # plt.savefig(f"/home/geoint/tri/Planet_khuong/output/training/train-polygon-{str(idx)}-{epoch}-{model_option}-pred.png", \
        #             dpi=300, bbox_inches='tight')
        # plt.close()

    return losses.local_avg


def val(data_loader, segment_model, epoch, model_option):
    losses = AverageMeter()
    segment_model.eval()
    global iteration

    with torch.no_grad():
        for idx, input in tqdm(enumerate(data_loader), total=len(data_loader)):

            input_im = input['ts'].to(cuda, dtype=torch.float32)
            input_mask = input['mask'].to(cuda, dtype=torch.long)

            (B,L,F,H,W) = input_im.shape

            input_mask = input_mask.view(B,H,W)

            # print(f"features shape: {features.shape}")
            # print(f"mask shape: {input_mask.shape}")

            mask_pred = segment_model(input_im)

            # print(f"mask pred shape: {mask_pred.shape}")

            loss = criterion(mask_pred, input_mask)

            losses.update(loss.item(), B)

            # output predictions
            
            # index_array = torch.argmax(mask_pred, dim=1)
            # print(index_array.cpu().numpy().dtype)

            # plt.figure(figsize=(20,20))
            # plt.subplot(1,3,1)
            # plt.title("Image")
            # x = input['ts']
            # y = input['mask']
            # image = np.transpose(x[0,5,:3,:,:].numpy(), (1,2,0))
            # # image = np.transpose(z_mean[0,:,:,:], (1,2,0))
            # # image = rescale_truncate(image)
            # plt.imshow(image)
            # # plt.savefig(f"{str(data_dir)}{ts_name}-{str(idx)}-input.png")
            # plt.subplot(1,3,2)
            # plt.title("Segmentation Label")
            # image = np.transpose(y[0,:,:], (0,1))
            # plt.imshow(image)
            # # plt.savefig(f"{str(data_dir)}{ts_name}-{str(idx)}-label.png")
            # plt.subplot(1,3,3)
            # plt.title(f"Segmentation Prediction")
            # image = np.transpose(index_array[0,:,:].cpu().numpy(), (0,1))
            # plt.imshow(image)
            # plt.savefig(f"/home/geoint/tri/Planet_khuong/output/training/val-{str(idx)}-{epoch}-{model_option}-pred.png", \
            #             dpi=300, bbox_inches='tight')
            # plt.close()

    return losses.local_avg

if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()

    # python models/train_benchmodel.py --model convlstm --dataset Tappan01 --img_dim 64 --epochs 100
    # python models/train_benchmodel.py --model convgru --dataset Tappan01 --img_dim 64 --epochs 100
