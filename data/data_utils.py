"""
Author: Uday Bondi
-------
Functions to obtain image patches from a given image and also create the reconstructed image 
"""
from skimage.measure import block_reduce
import numpy as np
import glob
import scipy.misc as sc_misc
import os
import scipy
import shutil
from matplotlib.image import imread
import math


def get_patches(img, patch_size= (56,56)):
	"""
	Creates patches from a given image
	--------------
	img: Numpy array of the img of interest
	patch_size: Shape of the patch to be extracted
	"""

	img = img.numpy()

	list_patch_startidx = generate_patch_startid_list(img.shape[2:], patch_size, patch_size)
	
	for i, patch_idx in enumerate(list_patch_startidx):
		patch = img[:,:,patch_idx[0]:patch_idx[0]+patch_size[0],patch_idx[1]:patch_idx[1]+patch_size[1]]
		if i==0:
			img_patches = patch
		else:
			img_patches = np.vstack((img_patches, patch))

	return img_patches

def reconstruct_image_from_patches(patches_list, patch_stride):
	"""
	Reconstructs an image from the given image
	---------------
	patches_list: List of patch images as numpy arrays
	patch_stride: The stride at which you have to place them in order to generate the complete image
	"""

	no_of_patches = patches_list.shape[0]
	full_image_shape = (int(((math.sqrt(no_of_patches)-1)*patch_stride[0])+ patches_list.shape[2]), int(((math.sqrt(no_of_patches)-1)*patch_stride[1])+ patches_list.shape[3]))

	assert all(patch[0].shape==patches_list[0,0].shape for patch in patches_list), "All the patches must have the same shape"
	patch_startid_list = generate_patch_startid_list(full_image_shape, patches_list[0,0].shape, patch_stride)
	recon_img = np.zeros(full_image_shape)
	patch_shape = patches_list[0,0].shape

	for i, patch_id in enumerate(patch_startid_list):
		patch = patches_list[i,0,:,:]

		recon_img[patch_id[0]: patch_id[0]+patch_shape[0], patch_id[1]: patch_id[1]+patch_shape[1]] = patch

	return recon_img

def _no_of_patches(in_shape, patch_size, stride_size):
	"""
	Get the number of patches that can be extracted from the image given the patch size and stride size
	-------------------
	Params:
	in_shape - Shape of the input volume as a List
	patch_size - Size of patch as a List
	stride_size - Stride as a list
	-------------------
	Output: no of patches as a list
	"""
	no_of_patches = []

	for i in range(len(in_shape)):
		no_of_patches.append(((in_shape[i] - patch_size[i])//stride_size[i]) + 1)

	return no_of_patches


def generate_patch_startid_list(volume_shape=(224, 224), patch_size=(56,56), stride_size=(56,56)):
	"""
	Given a 2d image, creates a list of patches according to the patch size and the stride size given as an input
	-------------------
	Params:
	volume: tuple indicating the shape of volume
	patch_size: tuple indicating the size of patches to extract
	stride_size: tuple indicating the dim to moven when extracting the pathces
	-------------------
	Output:
	List containing the start ids (tuples) of all possible patches
	"""

	in_shape = list(volume_shape)
	patch_size = list(patch_size)
	stride_size = list(stride_size)

	no_of_patches = _no_of_patches(in_shape, patch_size, stride_size)
	assert no_of_patches[0]!=0, "Cant create any patches"

	list_patch_startidx = []
	for i in range(no_of_patches[0]):
		for j in range(no_of_patches[1]):

				temp_start_idx = (stride_size[0]*i, stride_size[1]*j)
				list_patch_startidx.append(temp_start_idx)
	return list_patch_startidx

def get_list_of_files(folder_path,datatype):
	"""
	Returns the list of files in the given folder
	"""
    list_of_files = glob.glob(folder_path+datatype+'_set/*HC.png')
    return list_of_files

def create_directory(folder_path, directory='downsampled_images/'):
    if not os.path.exists(folder_path+directory):
        os.makedirs(folder_path+directory)


def downsample_img(img, scale=4):
    red_img = block_reduce(img,block_size =(scale,scale,1) , func = np.mean)
    return red_img

def downsample_imgs_to(source_dir, target_dir, scaling_factor, resize='True'):

	list_files = [f for f in os.listdir(source_dir) if os.path.isfile(source_dir+f)]
	for img_name in list_files:
		img = imread(source_dir+img_name)
		dwn_smpled_img = downsample_img(img, scaling_factor)
		#if resize:
		#    dwn_smpled_img= scipy.misc.imresize(dwn_smpled_img, img.shape, interp='bicubic')
		sc_misc.imsave(target_dir+img_name,dwn_smpled_img)


def create_downsampled_imgs(folder_path, scaling_factor,dataset = 'train', resize='True'):
    
    files_list = get_list_of_files(folder_path, dataset)
    for img_path in files_list:
        img = imread(img_path)
        img_file_name = img_path.replace(folder_path+dataset+"_set/","")
        dwn_smpled_img = downsample_img(img, scaling_factor)
        if resize:
               dwn_smpled_img= scipy.misc.imresize(dwn_smpled_img, img.shape, interp='bicubic') 
        create_directory(folder_path, directory = 'downsampled_images_'+dataset+'_'+str(scaling_factor))
        sc_msc.imsave(folder_path+'downsampled_images_'+dataset+'_'+str(scaling_factor)+'/'+dataset+'_r'+str(scaling_factor)+'_'+img_file_name,dwn_smpled_img)     


def prepare_data_training(folder_path, fraction_val = 0.25):

	print("-----Preparing your data folders---------")
	list_files = sorted(get_list_of_files(folder_path,'train'))
	no_train = int((1-fraction_val)*len(list_files))
	create_directory(folder_path,'train/')
	create_directory(folder_path,'val/')
	paths = [folder_path+'train/',folder_path+'val/']
	for i,img in enumerate(list_files):
		if i<=no_train:
			shutil.copy(img, paths[0])
		else:
			shutil.copy(img, paths[1])

	for data_type_path in paths:
		create_directory(data_type_path,'LR/')
		create_directory(data_type_path,'HR/')
		downsample_imgs_to(data_type_path, data_type_path+'LR/', 4, resize='False')
		list_hrfiles = [f for f in os.listdir(data_type_path) if os.path.isfile(data_type_path+f)]
		for img in list_hrfiles:
			shutil.move(data_type_path+img, data_type_path+'HR/')
	print("----- Done ------")



if __name__ =='__main__':
	data_dir = '../data/head_US/'
	prepare_data_training(data_dir) 
