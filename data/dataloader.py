"""
Author: UDAY BONDI
-------------------
Contains the Ultrasound super resolution dataset class. Along with that it can be used for 
getting a dataloader, getting transforms, getting a dataset for training and evaluation
"""

##----------------
## Imports
##----------------
import torch
from PIL import Image
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt 
import time
import os
import copy
import pdb
import glob
import numpy as np
from skimage import io, transform, color
import imageio
import math
from data.data_utils import get_patches, reconstruct_image_from_patches
import torchvision.transforms.functional as TF



##----------------
## Functions to get transforms, dataloaders, datasets
##----------------

def get_dataset(root_dir, opts, mode='train'):
	"""
	Use this function to obtain the dataset of interest based on the options provided
	-----------
	root_dir: (str) Path to the directory with data
	opts: options 
	mode: (str) 'train' or 'val' 
	"""
	if mode=='train':

		data_transforms = get_transforms()
		image_datasets = {x:US_dataset(os.path.join(root_dir, x+'/'),data_transforms[x], opts['Partial_train']) for x in ['train','val']}
		dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
	else:
		data_transforms = get_transforms()
		image_datasets = {x:US_dataset(root_dir,data_transforms[x]) for x in ['val']}
		dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}

	return image_datasets, dataset_sizes

def get_transforms():
	"""
	Function to retrieve transforms to be applied to the images 
	--------------
	returns: pytorch transforms
	"""
	data_mean = [1,1,1]
	data_std = [0,0,0]

	data_transforms = { 'train': (transforms.Compose([
									transforms.Resize((224,224)),
									transforms.ToTensor()
									]),
									transforms.Compose([
											transforms.Resize((224*4,224*4)),
											transforms.ToTensor()
											]),
								  transforms.Compose([
								  		transforms.RandomHorizontalFlip(),
								  		transforms.RandomRotation(90),
								  		transforms.ToTensor()
								  		])
								  ), 
						'val': (transforms.Compose([
									transforms.Resize((224,224)),
									transforms.ToTensor()
									]),
								transforms.Compose([
									transforms.Resize((224*4,224*4)),
									transforms.ToTensor()
									]))
						}

	return data_transforms


def get_data_loader(im_datasets, no_batches, to_shuffle=True):
	"""
	Use this function to obtain dataloaders that provide batches of data and suffle it if required
	------------
	im_datasets: pytorch dataset to be loaded
	no_batches: (int) batch size
	to_shuffle: (bool) Boolean indicating if the data needs to be shuffled
	"""
	#Should the shuffle for validation be true? Check this
	dataloaders = {x:torch.utils.data.DataLoader(im_datasets[x], batch_size=no_batches, shuffle=to_shuffle, num_workers=4) for x in ['train', 'val']}

	return dataloaders

##----------------
## Datasets 
##----------------
class US_dataset(Dataset):


	def __init__(self, root_dir, transform= None, partial_train = False):
		"""
		Use this class for fetal ultrasound(US) data available online. This dataset loads LR and HR images of US data 
		--------------------
		root_dir: (str) path to folder containing US data
		transform: Pytorch transform that would be applied to data
		partial_train: (bool) Can be used to only allow partial data to be trained
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.lr_list_files = glob.glob(root_dir+'LR/'+'*HC.png')
		self.hr_list_files = glob.glob(root_dir+'HR/'+'*HC.png')

		if partial_train:
			print("$$ Loading partial data for training $$")
			self.lr_list_files = self.lr_list_files[:50]
			self.hr_list_files = self.hr_list_files[:50]

		assert len(self.lr_list_files)==len(self.hr_list_files), "Every LR image doesnt have a corresponding HR pair"

	def __len__(self):
		"""
		Returns the number of images in the data folder
		"""
		return len(self.lr_list_files)

	def __getitem__(self, idx):
		"""
		To retrive a sample according to the index
		"""

		lr_img = Image.fromarray(color.gray2rgb(io.imread(self.lr_list_files[idx])))
		hr_img = Image.fromarray(color.gray2rgb(io.imread(self.hr_list_files[idx])))
		img_name = self.lr_list_files[idx].split('/')[-1]
		sample = {'lr': lr_img, 'hr': hr_img, 'name': img_name}

		if self.transform:
			sample['lr'] = self.transform[0](sample['lr'])
			sample['hr'] = self.transform[1](sample['hr'])
		return sample


##----------------
## Transforms 
##----------------

class paired_transform(object):

	def __init__(self, data_opts):
		"""
		Transform class that can be used for problems that require the same transform on input and target. 
		Ex: Segmentation 
		-----------
		data_opts: options 
		"""
		self.to_flip = False
		self.angle = 0
		self.Normalize = None
		if data_opts.get('use_flip'):
			self.to_flip = random.choice([True, False])
		if data_opts.get('use_rot'):
			self.angle = random.randint(-45, 45)
		if data_opts.get('Normalize'):
			self.norm_mean = data_opts['Normalize']['mean']
			self.norm_std =  data_opts['Normalize']['std']

	def __call__(self, sample):
		for img_type in ['lr', 'hr']:
			if self.angle:
				sample[img_type] = TF.rotate(sample[img_type], self.angle)
			if self.to_flip:
				sample[img_type] = TF.hflip(sample[img_type])
			sample[img_type]= TF.to_tensor(sample[img_type])
			if self.Normalize:
				sample[img_type] = TF.normalize(sample[img_type],self.norm_mean,self.norm_std)
		return sample

##----------------
## Test functions
##----------------

def test_us_dataset(root_dir, opts):

	us_dataset, us_data_sizes = get_dataset(root_dir, opts)
	dataloader = get_data_loader(us_datasets, 1)

	sample = next(iter(dataloader['train']))
	print("The pixel range for LR is: ({l}, {u})".format(l=sample['lr'][0].min(),u=sample['lr'].max()))
	print("The pixel range for HR is: ({l}, {u})".format(l=sample['hr'][0].min(),u=sample['hr'].max()))
	inputs, labels = sample['lr'], sample['hr']
	result = transforms.ToPILImage()(inputs[0])
	result.show(title='LR img with trans')

	result = transforms.ToPILImage()(labels[0])
	result.show(title='HR img with trans')

	patches = get_patches(inputs, patch_size=(56,56))
	recon_input = reconstruct_image_from_patches(patches, (56,56))

	io.imshow(patches[5,0])
	io.show()

	io.imshow(recon_input)
	io.show()



if __name__=='__main__':

	class options():
	    def __init__(self):
	    	self.partial_data_to_train = False

	data_dir = '../../Data/head_US/' 
	opts = options()
	us_datasets, us_data_sizes = get_dataset(data_dir, opts)
	print("The size of US datasets: ",us_data_sizes)
	us_dataloader = get_data_loader(us_datasets, 4)

	test_us_dataset(data_dir, opts)

	for i_batch, sample_batched in enumerate(us_dataloader['train']):
		print(i_batch, sample_batched['lr'].size(), sample_batched['hr'].size())

		if i_batch==2:
			break






