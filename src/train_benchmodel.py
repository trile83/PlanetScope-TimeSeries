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
import rioxarray as rxr
from tensorboardX import SummaryWriter
import json
from utils.utils import AverageMeter, save_checkpoint

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='biconvlstm', type=str, help='convlstm, convgru')
parser.add_argument('--dataset', default='tile01', type=str, help='tile01, tile02')
parser.add_argument('--seq_len', default=6, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=4, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=64, type=int)
parser.add_argument('--ts_length', default=10, type=int)
parser.add_argument('--pad_size', default=0, type=int)
parser.add_argument('--num_chips', default=400, type=int)
parser.add_argument('--num_val', default=40, type=int)
parser.add_argument('--num_classes', default=5, type=int)


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

    # print(json_data)

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
    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    # prepare data
    ##### REMEMBER TO CHECK IF THE IMAGE IS CHIPPED IN THE NO-DATA REGION, MAKE SURE IT HAS DATA.
    ts_name=args.dataset
    ts_arr, mask_arr = read_dataset(tile_name=ts_name)

    print('Finish loading time series!')

    input_size = args.img_dim
    total_ts_len = args.ts_length # L

    padding_size = args.pad_size
    
    # print(f'data dict tappan01 ts shape: {ts_arr.shape}')
    # print(f'data dict tappan01 mask shape: {mask_arr.shape}')

    train_ts_set = []
    train_mask_set = []

    ### get RGB image
    # ts_arr = ts_arr[:,1:4,:,:]
    # ts_arr = ts_arr[:,::-1,:,:]

    ## get different chips in the Tappan Square for multiple time series
    num_chips=args.num_chips # I
    num_val=args.num_val

    # h_list_train =[10,20]
    # w_list_train =[15,25]
    h_list_train =[10,20,30,40,50,70,80,90,100,110,200]
    w_list_train =[15,25,35,45,55,75,85,95,105,115,215]

    temp_ts_set = []
    temp_mask_set = []

    # for i in range(len(h_list_train)):
    #     ts, mask = specific_chipper(ts_arr[:total_ts_len,:,:,:], mask_arr,h_list_train[i], w_list_train[i], input_size=input_size)

    i=0
    while i < (num_chips+num_val):
        ts, mask = chipper(ts_arr[:total_ts_len,:,:,:], mask_arr, input_size=input_size)
        # ts = ts.reshape((ts.shape[1],ts.shape[2],ts.shape[3],ts.shape[4]))
        ts = np.squeeze(ts)
        if np.any(ts==0) or np.any(mask==5):
            continue

        print(f"min pixel value: {np.min(ts)} & max pixel value: {np.max(ts)}")

        # t_im = np.transpose(standardize_image(rescale_image(ts[5,:3,:,:]),'local'), (1,2,0))

        # plt.figure(figsize=(10,10))
        # plt.subplot(1,2,1)
        # plt.imshow(t_im)
        # plt.subplot(1,2,2)
        # plt.imshow(np.squeeze(mask))
        # plt.savefig(f'/home/geoint/tri/Planet_khuong/output/train_output/train_input_im-{i}.png',\
        #              dpi=300,bbox_inches='tight')
        # plt.close()

        # del t_im

        ts_old = ts
        for frame in range(ts.shape[0]):
            ts[frame] = standardize_image(rescale_image(ts[frame]),'local')
        # mask = mask.reshape((mask.shape[1],mask.shape[2]))
        mask = np.squeeze(mask)

        print(f"after rescale & standardize: min pixel value: {np.min(ts)} & max pixel value: {np.max(ts)}")

        # ts, mask = padding_ts(ts, mask, padding_size=padding_size)

        temp_ts_set.append(ts)
        temp_mask_set.append(mask)

        i+=1

    train_ts_set = np.stack(temp_ts_set, axis=0)
    train_ts_set = train_ts_set[:,:total_ts_len] # get the first 100 in the time series
    train_mask_set = np.stack(temp_mask_set, axis=0)

    del temp_ts_set
    del temp_mask_set

    del ts_arr
    del mask_arr

    print(f"train ts set shape: {train_ts_set.shape}")
    print(f"train mask set shape: {train_mask_set.shape}")

    ### dpc model ###

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Load model')

    model_option = args.model
    if model_option == "convlstm":
        model = ConvLSTM_Seg(
            num_classes=args.num_classes,
            input_size=(input_size,input_size),
            hidden_dim=160,
            input_dim=4,
            kernel_size=(3, 3)
            )
    elif model_option == "convgru":
            model = ConvGRU_Seg(
                num_classes=args.num_classes,
                input_size=(input_size,input_size),
                input_dim=4,
                kernel_size=(3, 3),
                hidden_dim=180,
            )

    elif model_option == "biconvlstm":
            model = BConvLSTM_Seg(
                num_classes=args.num_classes,
                input_size=(input_size,input_size),
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
    global img_path; img_path, model_path = set_path(args)
    model_dir = "/home/geoint/tri/Planet_khuong/output/checkpoints/"
    
    ### main loop ###
    train_loss_lst = []
    val_loss_lst = []

    # print(f"train mask set shape: {train_mask_set.shape}")

    train_seg_set = tsDataset(train_ts_set[:-num_val],  train_mask_set[:-num_val])
    val_seg_set = tsDataset(train_ts_set[-num_val:],  train_mask_set[-num_val:])
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
                                    f'{model_option}__planet_4band_epoch_%s.pth' % str(epoch+1)), keep_all=False)

        
    plt.plot(train_loss_lst, color ="blue")
    plt.plot(val_loss_lst, color = "red")
    plt.savefig(f'/home/geoint/tri/Planet_khuong/output/train_output/{model_option}_train_loss.png')
    plt.close()

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))

def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def train(data_loader, segment_model, optimizer, epoch, model_option):
    losses = AverageMeter()
    segment_model.train()
    global iteration

    for idx, input in enumerate(data_loader):

        input_im = input['ts'].to(cuda, dtype=torch.float32)
        input_mask = input['mask'].to(cuda, dtype=torch.long)

        (B,L,F,H,W) = input_im.shape
        batch = 1

        input_mask = input_mask.view(batch,H,W)

        # print(f"features shape: {features.shape}")
        # print(f"mask shape: {input_mask.shape}")

        mask_pred = segment_model(input_im)

        # print(f"mask pred shape: {mask_pred.shape}")

        loss = criterion(mask_pred, input_mask)

        losses.update(loss.item(), B)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        # plt.savefig(f"/home/geoint/tri/Planet_khuong/output/training/train-{str(idx)}-{epoch}-{model_option}-pred.png", \
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
            batch = 1

            input_mask = input_mask.view(batch,H,W)

            # print(f"features shape: {features.shape}")
            # print(f"mask shape: {input_mask.shape}")

            mask_pred = segment_model(input_im)

            # print(f"mask pred shape: {mask_pred.shape}")

            loss = criterion(mask_pred, input_mask)

            losses.update(loss.item(), B)

            # # output predictions
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
            # plt.savefig(f"/home/geoint/tri/Planet_khuong/output/training/val-{str(idx)}-{epoch}-{model_option}-pred.png", \
            #             dpi=300, bbox_inches='tight')
            # plt.close()

    return losses.local_avg


def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}'.format(
                    'r%s' % args.net[6::], \
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path

if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()

    # python models/train_benchmodel.py --model convlstm --dataset Tappan01 --img_dim 64 --epochs 100
    # python models/train_benchmodel.py --model convgru --dataset Tappan01 --img_dim 64 --epochs 100
