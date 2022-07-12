import argparse
import os
from glob import glob
import math
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10
from skimage.transform import resize
import lpips
import torch

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    #img1 = img1.astype(np.float64)
    #img2 = img2.astype(np.float64)
    #img1=resize(img1,img2.shape)
    
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    #img1=resize(img1,img2.shape)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def np2tensor(array):
    return torch.Tensor(array[:,:,:,np.newaxis].transpose(3,2,0,1))
        
def get_parser():
    parser = argparse.ArgumentParser(description="evaluate final results for SR with 5 metrics")
    parser.add_argument('--original_dir', required=True, type=str, help="original image files location")
    parser.add_argument('--result_dir', required=True, type=str, help="Result image files location")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    if not os.path.exists(opt.original_dir):
        raise ValueError("Cannot find {}".format(opt.original_dir))
    
    if not os.path.exists(opt.result_dir):
        raise ValueError("Cannot find {}".format(opt.result_dir))
        
    original_path_list = []
    original_path_list += glob(opt.original_dir + '/*.jpg')
    original_path_list += glob(opt.original_dir + '/*.JPEG')
    original_path_list += glob(opt.original_dir + '/*.png')
    
    psnr_list = []
    ssim_list = []
    lpips_list = []
    fid_list = []
    idd_list = []
    
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    
    for image_path in tqdm(original_path_list):
        image_name = image_path.split("/")[-1].split(".")[0]
        original_image = Image.open(image_path).convert('RGB')
        original_image = np.array(original_image)
        
        result_image_name = image_name + "_00.png" 
        result_image = Image.open(opt.result_dir + "/" + result_image_name).convert('RGB')
        result_image = np.array(result_image)
                
        psnr_list.append(calculate_psnr(original_image, result_image))
        ssim_list.append(calculate_ssim(original_image, result_image))
        lpips_list.append(loss_fn_vgg(np2tensor(original_image), np2tensor(result_image)))
        
        fid_original_image = preprocess_input(scale_images(original_image, (299,299,3)))
        fid_result_image = preprocess_input(scale_images(result_image, (299,299,3)))
        
        fid_list.append(calculate_fid(fid_model, fid_original_image, fid_result_image))
        
    print("PSNR : {}, SSIM : {}, LPIPS : {}, FID : {}, IDD : {}".format(np.mean(np.array(psnr_list)), np.mean(np.array(ssim_list)), 
                                                                        np.mean(np.array(lpips_list)), np.mean(np.array(fid_list)), np.mean(np.array(idd_list))))
    