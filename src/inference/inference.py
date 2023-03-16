import logging
import numpy as np
# import tensorflow as tf
import scipy.signal.windows as w
# from .tiler.tiler import Tiler, Merger
# from .mosaic import from_array
from .data import normalize_image, rescale_image, \
    standardize_batch, standardize_image
from tiler import Tiler, Merger
import torch
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from skimage import exposure


def window2d(window_func, window_size, **kwargs):
    window = np.matrix(window_func(M=window_size, sym=False, **kwargs))
    return window.T.dot(window)

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

def standardize_batch(
    image_batch,
    standardization_type: str,
    mean: list = None,
    std: list = None
):
    """
    Standardize image within parameter, simple scaling of values.
    Loca, Global, and Mixed options.
    """
    for item in range(image_batch.shape[0]):
        image_batch[item, :, :, :] = standardize_image(
            image_batch[item, :, :, :], standardization_type, mean, std)
    return image_batch


def generate_corner_windows(window_func, window_size, **kwargs):
    step = window_size >> 1
    window = window2d(window_func, window_size, **kwargs)
    window_u = np.vstack(
        [np.tile(window[step:step+1, :], (step, 1)), window[step:, :]])
    window_b = np.vstack(
        [window[:step, :], np.tile(window[step:step+1, :], (step, 1))])
    window_l = np.hstack(
        [np.tile(window[:, step:step+1], (1, step)), window[:, step:]])
    window_r = np.hstack(
        [window[:, :step], np.tile(window[:, step:step+1], (1, step))])
    window_ul = np.block([
        [np.ones((step, step)), window_u[:step, step:]],
        [window_l[step:, :step], window_l[step:, step:]]])
    window_ur = np.block([
        [window_u[:step, :step], np.ones((step, step))],
        [window_r[step:, :step], window_r[step:, step:]]])
    window_bl = np.block([
        [window_l[:step, :step], window_l[:step, step:]],
        [np.ones((step, step)), window_b[step:, step:]]])
    window_br = np.block([
        [window_r[:step, :step], window_r[:step, step:]],
        [window_b[step:, :step], np.ones((step, step))]])
    return np.array([
        [window_ul, window_u, window_ur],
        [window_l, window, window_r],
        [window_bl, window_b, window_br],
    ])


def generate_patch_list(
            image_width,
            image_height,
            window_func,
            window_size,
            overlapping=False
        ):
    patch_list = []
    if overlapping:
        step = window_size >> 1
        windows = generate_corner_windows(window_func, window_size)
        # max_height = int(image_height/step - 1) * step
        # max_width = int(image_width/step - 1) * step
        # print("max_height, max_width", max_height, max_width)
    else:
        step = window_size
        windows = np.ones((window_size, window_size))
        # max_height = int(image_height / step) * step
        # max_width = int(image_width / step) * step
        # print("else max_height, max_width", max_height, max_width)

    # for i in range(0, max_height, step):
    #    for j in range(0, max_width, step):
    for i in range(0, image_height-step, step):
        for j in range(0, image_width-step, step):
            if overlapping:
                # Close to border and corner cases
                # Default (1, 1) is regular center window
                border_x, border_y = 1, 1
                if i == 0:
                    border_x = 0
                if j == 0:
                    border_y = 0
                if i == image_height - step:
                    border_x = 2
                if j == image_width - step:
                    border_y = 2
                # Selecting the right window
                current_window = windows[border_x, border_y]
            else:
                current_window = windows

            # The patch is cropped when the patch size is not
            # a multiple of the image size.
            patch_height = window_size
            if i+patch_height > image_height:
                patch_height = image_height - i

            patch_width = window_size
            if j+patch_width > image_width:
                patch_width = image_width - j

            # print(f'i {i} j {j} patch_height {patch_height}
            # patch_width {patch_width}')

            # Adding the patch
            patch_list.append(
                (
                    j,
                    i,
                    patch_width,
                    patch_height,
                    current_window[:patch_width, :patch_height]
                )
            )
    return patch_list

class satDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X):
        'Initialization'
        self.data = X

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]

        return {
            'x': X
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

def train_dpc(input_seq, dpc_model):

    dpc_model.train()

    (B,N,SL,C,H,W) = input_seq.shape
    input_seq = input_seq
    features, _ = dpc_model(input_seq)

    return features.cpu().detach()

def train_dpc_poisson(data_loader, dpc_model):

    dpc_model.train()
    global iteration

    feature_lst = []

    for idx, input in enumerate(data_loader):

        # print(f'id: {idx}')
        input_seq = input["x"]

        input_seq = input_seq.to(cuda, dtype=torch.float32)

        features = dpc_model(input_seq)
        feature_lst.append(features)

    feature_arr = torch.cat(feature_lst, dim=0)

    return feature_arr.cpu().detach()

def get_feature_arr(new_set, train_mask_set, model, num_seq, seq_length):

    test_set = tsDataset(new_set, train_mask_set)
    # Create data loaders
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=False)
    train_dl = DataLoader(test_set, **loader_args)
    val_dl = DataLoader(test_set, **loader_args)
    
    # print(f"length of dpc input training set {len(train_dl)}")

    all_feature_arr = []

    for idx, input in enumerate(train_dl):

        input_ts = input['ts'].to(dtype=torch.float32)
        # input_mask = input['mask']

        (I,L2,N,SL,C,H,W) = input_ts.shape

        input_ts = rearrange(input_ts, "b l2 n sl c h w -> (b l2) n sl c h w")

        # print(f"input ts shape {input_ts.shape}")
        # print(f"input mask shape {input_mask.shape}")

        test_set = satDataset(input_ts)
        loader_args_sat = dict(batch_size=1, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
        train_sat_dl = DataLoader(test_set, **loader_args_sat)

        # for epoch in range(len(train_sat_dl)):

        feature_arr = train_dpc_poisson(train_sat_dl, model)
        feature_arr = rearrange(feature_arr, "(b l2) n sl c h w -> b l2 n sl c h w", l2 = L2)

        # print(f"feature arr shape: {feature_arr.shape}")
        all_feature_arr.append(feature_arr)

    all_feature_arr = torch.cat(all_feature_arr, dim=0)

    # print(f"all feature arr shape: {all_feature_arr.shape}")
    all_feature_arr = reverse_chunks(all_feature_arr, num_seq)
    # print(f'reverse chunk context vector shape: {all_feature_arr.shape}')
    all_feature_arr = reverse_seq(all_feature_arr, seq_length)
    # print(f'reversed windown context vector shape: {all_feature_arr.shape}')

    return all_feature_arr
    
    
def sliding_window_tiler(
            xraster,
            model,
            n_classes: int,
            pad_style: str = 'reflect',
            overlap: float = 0.50,
            constant_value: int = 600,
            batch_size: int = 1024,
            threshold: float = 0.50,
            standardization: str = None,
            dpc_model=None,
            mean=None,
            std=None,
            window: str = 'triang',  # 'overlap-tile',
            normalize=10000.0,
            rescale=None,
            im_type='hls',
            model_option='unet'
        ):
    """
    Sliding window using tiler.
    """
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = \
    #    tf.data.experimental.AutoShardPolicy.OFF
    # batch = tf.data.Dataset.from_tensor_slices(
    #    np.expand_dims(batch, axis=0))
    # batch = batch.with_options(self.options)
    # batch = function(batch, batch_size=batch_size)

    # tile_size = model.layers[0].input_shape[0][1]
    # tile_channels = model.layers[0].input_shape[0][-1]

    tile_size = 64
    tile_channels = xraster.shape[1]
    seq_length = 5
    num_seq = 4
    global cuda; cuda = torch.device('cuda')
    # n_classes = out of the output layer, output_shape

    if model_option == 'unet':

        tiler_image = Tiler(
            data_shape=xraster.shape,
            tile_shape=(tile_channels, tile_size, tile_size),
            channel_dimension=0,
            overlap=overlap,
            mode=pad_style,
            constant_value=constant_value
        )

        # Define the tiler and merger based on the output size of the prediction
        tiler_mask = Tiler(
            data_shape=(n_classes, xraster.shape[1], xraster.shape[2]),
            tile_shape=(n_classes, tile_size, tile_size),
            channel_dimension=0,
            overlap=overlap,
            mode=pad_style,
            constant_value=constant_value
        )

    elif model_option == 'dpc-unet':

        tiler_image = Tiler(
            data_shape=xraster.shape,
            tile_shape=(xraster.shape[0], tile_channels, tile_size, tile_size),
            channel_dimension=1,
            overlap=overlap,
            mode=pad_style,
            constant_value=constant_value
        )

        # Define the tiler and merger based on the output size of the prediction
        tiler_mask = Tiler(
            data_shape=(n_classes, xraster.shape[2], xraster.shape[3]),
            tile_shape=(n_classes, tile_size, tile_size),
            channel_dimension=0,
            overlap=overlap,
            mode=pad_style,
            constant_value=constant_value
        )

    elif model_option == 'dpc-unet-poisson':

        tiler_image = Tiler(
            data_shape=xraster.shape,
            tile_shape=(xraster.shape[0], tile_channels, tile_size, tile_size),
            channel_dimension=1,
            overlap=overlap,
            mode=pad_style,
            constant_value=constant_value
        )

        # Define the tiler and merger based on the output size of the prediction
        tiler_mask = Tiler(
            data_shape=(xraster.shape[0], 128, xraster.shape[2], xraster.shape[3]),
            tile_shape=(xraster.shape[0], 128, tile_size, tile_size),
            channel_dimension=0,
            overlap=overlap,
            mode=pad_style,
            constant_value=constant_value
        )

    else:

        tiler_image = Tiler(
            data_shape=xraster.shape,
            tile_shape=(xraster.shape[0], tile_channels, tile_size, tile_size),
            channel_dimension=1,
            overlap=overlap,
            mode=pad_style,
            constant_value=constant_value
        )

        # Define the tiler and merger based on the output size of the prediction
        tiler_mask = Tiler(
            data_shape=(n_classes, xraster.shape[2], xraster.shape[3]),
            tile_shape=(n_classes, tile_size, tile_size),
            channel_dimension=0,
            overlap=overlap,
            mode=pad_style,
            constant_value=constant_value
        )

    # new_shape_image, padding_image = tiler_image.calculate_padding()
    # new_shape_mask, padding_mask = tiler_mask.calculate_padding()
    # print(xraster.shape, new_shape_image, new_shape_mask)
    # tiler_image.recalculate(data_shape=new_shape_image)
    # tiler_mask.recalculate(data_shape=new_shape_mask)

    # merger = Merger(tiler=tiler_mask, window=window, logits=4)
    merger = Merger(
        tiler=tiler_mask, window=window)  # #logits=4,
    #    tile_shape_merge=(tile_size, tile_size))
    # print(merger)
    # print("WEIGHTS SHAPE", merger.weights_sum.shape)
    # print("WINDOW SHAPE", merger.window.shape)

    # xraster = xraster.pad(
    #    y=padding_image[0], x=padding_image[1],
    #    constant_values=constant_value)
    # print("After pad", xraster.shape)

    # xraster = rescale_image(xraster)

    # Iterate over the data in batches
    if model_option == 'dpc-unet':
        for batch_id, batch in tiler_image(xraster, batch_size=batch_size):

            # print("batch shape", batch.shape)
            batch = rescale_image(batch)

            # for i in range(batch.shape[1]):
            #     batch[:,i,:,:,:] = rescale_image(batch[:,i,:,:,:])

            # Standardize
            # batch = batch / normalize

            # if standardization is not None:
            #     for i in range(batch.shape[1]):
            #         batch[:,i,:,:,:] = standardize_batch(batch[:,i,:,:,:], standardization)

            # DPC:
            im_set = get_seq(batch, seq_length)
            new_set = get_chunks(im_set, num_seq)
            new_set = torch.tensor(new_set).to(cuda, dtype=torch.float32)
            # print('new_set shape: ', new_set.shape)

            (I,L2,N,SL,C,H,W) = new_set.shape

            new_set = rearrange(new_set, "i l2 n sl c h w -> (i l2) n sl c h w")
            output = train_dpc(new_set, model)
            
            # batch = np.moveaxis(batch, -1, 1)
            # print("AFTER PREDICT", batch.shape, batch_id)

            # Merge the updated data in the array
            merger.add_batch(batch_id, batch_size, output.numpy())

    elif model_option == 'dpc-unet-poisson':
        for batch_id, batch in tiler_image(xraster, batch_size=batch_size):

            # print("batch shape", batch.shape)
            # batch = rescale_image(batch)

            # for i in range(batch.shape[1]):
            #     batch[:,i,:,:,:] = rescale_image(batch[:,i,:,:,:])

            # DPC:
            im_set = get_seq(batch, seq_length=6) #(I,L1,SL,C,H,W)
            new_set = get_chunks(im_set, num_seq=4)

            output = get_feature_arr(new_set, batch, model,num_seq=4,seq_length=6)
            # print(f'final output shape: {output.shape}')
            
            # batch = np.moveaxis(batch, -1, 1)
            # print("AFTER PREDICT", batch.shape, batch_id)

            # Merge the updated data in the array
            merger.add_batch(batch_id, batch_size, output)

    elif model_option == 'unet':

        for batch_id, batch in tiler_image(xraster, batch_size=batch_size):

            # Standardize
            batch = rescale_image(batch)

            # if standardization is not None:
                # batch = standardize_batch(batch, standardization)

            # print("AFTER STD", batch.shape)

            batch = torch.Tensor(batch).to(cuda, dtype=torch.float32)

            x = batch

            # Predict
            # batch = model.predict(batch, batch_size=batch_size)
            batch = model(x)[0]
            # batch = np.moveaxis(batch, -1, 1)
            # print("AFTER PREDICT", batch.shape, batch_id)

            # Merge the updated data in the array
            merger.add_batch(batch_id, batch_size, batch.detach().cpu().numpy())

    else:
        for batch_id, batch in tiler_image(xraster, batch_size=batch_size):

            # Rescale and Standardize
            if im_type == 'hls':
                batch = rescale_image(batch)
            elif im_type == 'planet':
                for i in range(batch.shape[0]):
                    for j in range(batch.shape[1]):
                        batch[i,j,:,:,:] = standardize_image(rescale_image(batch[i,j,:,:,:]),'local')

            # print("AFTER STD", batch.shape)

            batch = torch.Tensor(batch).to(cuda, dtype=torch.float32)

            x = batch

            # Predict
            batch = model(x)

            # Merge the updated data in the array
            merger.add_batch(batch_id, batch_size, batch.detach().cpu().numpy())


    # prediction = merger.merge(
    # extra_padding=padding_mask, unpad=True, dtype=xraster.dtype,
    # normalize_by_weights=False)
    prediction = merger.merge(unpad=True)

    print('prediction shape: ', prediction.shape)

    #### prediction = np.transpose(prediction, (1,2,0))


    # if prediction.shape[0] > 1:
    #     prediction = np.argmax(prediction, axis=0)
    # else:
    #     prediction = np.squeeze(
    #         np.where(prediction > threshold, 1, 0).astype(np.int16)
    #     )

    # print('prediction shape after final process: ', prediction.shape)
    
    return prediction


def sliding_window_tiler_multiclass(
            xraster,
            model,
            n_classes: int,
            pad_style: str = 'reflect',
            overlap: float = 0.50,
            constant_value: int = 600,
            batch_size: int = 1024,
            threshold: float = 0.50,
            standardization: str = None,
            mean=None,
            std=None,
            normalize: float = 1.0,
            rescale: str = None,
            window: str = 'triang',  # 'overlap-tile'
            probability_map: bool = False
        ):
    """
    Sliding window using tiler.
    """

    # tile_size = model.layers[0].input_shape[0][1]
    # tile_channels = model.layers[0].input_shape[0][-1]

    tile_size = 512
    tile_channels = 13
    # n_classes = out of the output layer, output_shape

    tiler_image = Tiler(
        # data_shape=xraster.shape,
        data_shape=(xraster.shape[2], xraster.shape[0], xraster.shape[1]),
        # tile_shape=(tile_size, tile_size, tile_channels),
        tile_shape=(tile_channels, tile_size, tile_size),
        channel_dimension=-1,
        overlap=overlap,
        mode=pad_style,
        constant_value=constant_value
    )

    # Define the tiler and merger based on the output size of the prediction
    tiler_mask = Tiler(
        # data_shape=(xraster.shape[0], xraster.shape[1], n_classes),
        data_shape=(n_classes, xraster.shape[0], xraster.shape[1]),
        # tile_shape=(tile_size, tile_size, n_classes),
        tile_shape=(n_classes, tile_size, tile_size, n_classes),
        channel_dimension=-1,
        overlap=overlap,
        mode=pad_style,
        constant_value=constant_value
    )

    # new_shape_image, padding_image = tiler_image.calculate_padding()
    # new_shape_mask, padding_mask = tiler_mask.calculate_padding()
    # print(xraster.shape, new_shape_image, new_shape_mask)

    # tiler_image.recalculate(data_shape=new_shape_image)
    # tiler_mask.recalculate(data_shape=new_shape_mask)

    # merger = Merger(tiler=tiler_mask, window=window, logits=4)
    merger = Merger(tiler=tiler_mask, window=window)  # , #logits=4,
    #    tile_shape_merge=(tile_size, tile_size))
    # print(merger)
    # print("WEIGHTS SHAPE", merger.weights_sum.shape)
    # print("WINDOW SHAPE", merger.window.shape)

    # xraster = xraster.pad(
    #    y=padding_image[0], x=padding_image[1],
    #    constant_values=constant_value)
    # print("After pad", xraster.shape)

    xraster = normalize_image(xraster, normalize)

    if rescale is not None:
        xraster = rescale_image(xraster, rescale)

    # Iterate over the data in batches
    for batch_id, batch_i in tiler_image(xraster, batch_size=batch_size):

        # Standardize
        batch = batch_i.copy()

        if standardization is not None:
            # batch = standardize_batch(batch, standardization, mean, std)
            for item in range(batch.shape[0]):
                batch[item, :, :, :] = standardize_image(
                    batch[item, :, :, :], standardization, mean, std)

        # print("AFTER STD", batch.shape)

        # Predict
        batch = model.predict(batch, batch_size=batch_size)
        # batch = np.moveaxis(batch, -1, 1)
        # print("AFTER PREDICT", batch.shape, batch_id)

        # Merge the updated data in the array
        merger.add_batch(batch_id, batch_size, batch)

    # prediction = merger.merge(
    #    extra_padding=padding_mask, unpad=True, dtype=xraster.dtype,
    # normalize_by_weights=False)
    prediction = merger.merge(unpad=True)
    print("MIN MAX", prediction.min(), prediction.max())

    if not probability_map:
        if prediction.shape[-1] > 1:
            prediction = np.argmax(prediction, axis=-1)
        else:
            prediction = np.squeeze(
                np.where(prediction > threshold, 1, 0).astype(np.int16)
            )
    else:
        prediction = np.squeeze(prediction)
    return prediction