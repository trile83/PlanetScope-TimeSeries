# https://youtu.be/fVeW9a6wItM
"""

@author: Digitalsreeni (Sreenivas Bhattiprolu)

First make sure pytorch and torchcvision are installed, for GPU
In my case: pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install opencv-python matplotlib
pip install 'git+https://github.com/facebookresearch/segment-anything.git'

OR download the repo locally and install
and:  pip install -e .

Download the default trained model: 
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Other models are available:
    https://github.com/facebookresearch/segment-anything#model-checkpoints

"""
# Tested on python 3.9.16

import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
# import rioxarray as rxr
from skimage import exposure
import logging
import tifffile

import sys
sys.path.append("..")
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from samgeo import SamGeo, tms_to_geotiff

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
    if rescale_type == 'per-image':
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    elif rescale_type == 'per-channel':
        for i in range(image.shape[0]):
            image[i, :, :] = (
                image[i, :, :] - np.min(image[i, :, :])) / \
                (np.max(image[i, :, :]) - np.min(image[i, :, :]))
    else:
        logging.info(f'Skipping based on invalid option: {rescale_type}')
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

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        print(np.unique(m))
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        # ax.imshow(np.dstack((img, m*0.35)))
        ax.imshow(np.dstack((img, m*0.70)))

if __name__ == '__main__':

        
    image = cv2.imread('output/20210918.png')  #Try houses.jpg or neurons.jpg
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(image.shape)
    print(image.dtype)

    file_type = 'dakota'

    # try tiff file
    pl_file = \
            '/home/geoint/tri/Planet_khuong/07-21/files/PSOrthoTile/4660870_1459221_2021-07-05_100a/analytic_sr_udm2/4660870_1459221_2021-07-05_100a_BGRN_SR.tif'

    pl_file_1 = '/home/geoint/tri/Planet_khuong/SAM_inputs/4912910_1459321_2021-09-18_242d_BGRN_SR.tif'

    if file_type == 'senegal':
        pl_file = '/home/geoint/tri/nasa_senegal/cassemance/Tappan01_WV02_20110430_M1BS_103001000A27E100_data.tif'
    elif file_type == 'dakota':
        # pl_file = '/home/geoint/tri/Planet_khuong/output/tile01-other.tif'
        pl_file = pl_file_1


    # name = pl_file[-43:-4]
    image = tifffile.imread(pl_file)

    # image = image[:1000,:1000,1:4]

    # image_clone = np.zeros(image.shape)
    # image_clone[:,:,0] = image[:,:,2]
    # image_clone[:,:,1] = image[:,:,0]
    # image_clone[:,:,2] = image[:,:,1]

    # image= image_clone
    
    print(image.dtype)
    print(image.shape)

    # plt.figure(figsize=(10,10))
    # plt.imshow(rescale_truncate(rescale_image(image)))
    # # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    image[image<-9999] = -1

    image = image.astype('int8')

    model_type = "vit_l"

    if model_type == "vit_h":
        sam_checkpoint = "/home/geoint/tri/Planet_khuong/sam_model/sam_vit_h_4b8939.pth"
    elif model_type == "vit_l":
        sam_checkpoint = "/home/geoint/tri/Planet_khuong/sam_model/sam_vit_l_0b3195.pth"
    elif model_type == "vit_b":
        sam_checkpoint = "/home/geoint/tri/Planet_khuong/sam_model/sam_vit_b_01ec64.pth"

    device = "cuda"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam_kwargs = {
        "points_per_side": 32,
        "pred_iou_thresh": 0.86,
        "stability_score_thresh": 0.92,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 100,
    }

    sam = SamGeo(
        checkpoint=sam_checkpoint,
        model_type=model_type,
        device=device,
        erosion_kernel=(3, 3),
        mask_multiplier=255,
        sam_kwargs=sam_kwargs,
    )

    if file_type == 'senegal':
        mask = f'output/sam/senegal/Tappan01_WV02_20110430_M1BS_103001000A27E100_data-segment-{model_type}.tif'
        sam.generate(pl_file, mask)

        shapefile = f'output/sam/senegal/Tappan01_WV02_20110430_M1BS_103001000A27E100_data-segment-{model_type}.shp'
        sam.tiff_to_vector(mask, shapefile)
    else:
        mask = f'output/sam/other-{model_type}.tif'
        sam.generate(pl_file, mask)

        shapefile = f'output/sam/other-{model_type}.shp'
        sam.tiff_to_vector(mask, shapefile)


    #There are several tunable parameters in automatic mask generation that control 
    # how densely points are sampled and what the thresholds are for removing low 
    # quality or duplicate masks. Additionally, generation can be automatically 
    # run on crops of the image to get improved performance on smaller objects, 
    # and post-processing can remove stray pixels and holes. 
    # Here is an example configuration that samples more masks:
    #https://github.com/facebookresearch/segment-anything/blob/9e1eb9fdbc4bca4cd0d948b8ae7fe505d9f4ebc7/segment_anything/automatic_mask_generator.py#L35    

    #Rerun the following with a few settings, ex. 0.86 & 0.9 for iou_thresh
    # and 0.92 and 0.96 for score_thresh

    #############

    # mask_generator_ = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=40,
    #     pred_iou_thresh=0.86,
    #     stability_score_thresh=0.92,
    #     crop_n_layers=1,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=500,  # Requires open-cv to run post-processing
    #     output_mode='binary_mask'
    # )

    # masks = mask_generator_.generate(image)

    # print(len(masks))
    # # print(masks['segmentation'].shape)

    # plt.figure(figsize=(10,10))
    # plt.imshow(rescale_truncate(rescale_image(image)), alpha=0.4)
    # # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.savefig(f'output/test_segment_anything_{model_type}_0417.png', dpi=300, bbox_inches='tight')
    # # plt.show() 
    # plt.close()

    # torch.cuda.empty_cache()

    # """
    # Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:

    # segmentation : the mask
    # area : the area of the mask in pixels
    # bbox : the boundary box of the mask in XYWH format
    # predicted_iou : the model's own prediction for the quality of the mask
    # point_coords : the sampled input point that generated this mask
    # stability_score : an additional measure of mask quality
    # crop_box : the crop of the image used to generate this mask in XYWH format

    # """








