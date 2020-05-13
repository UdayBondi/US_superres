"""
Author: Uday Bondi
-------
Functions to apply transfer learning to train the model and test it. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt 
import time
import os
import copy
import pdb
from tqdm import tqdm
from data.data_utils import get_patches, reconstruct_image_from_patches, create_directory 
import math
from evaluation_utils import calculate_psnr, log2file
from skimage import io
import scipy.misc as sc_msc
import cv2
from data.dataloader import test_us_dataset
import pathlib


def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, scheduler, opts):
	"""
	Function to train a model 
	-------------
	Model:         Pytorch network 
	dataloaders:   Pytorch Dataloaders
	dataset_sizes: The number of samples provided in the dataset
	device:        Device where the training needs to be done
	criterion:     Loss function being used 
	optimizer:     Pytorch optimization function
	scheduler:     Pytorch scheduling to change the learning rate
	opts:          Options file
	"""
	create_directory(opts['path']['save_path'],'train_'+opts['name'])
	progress = log2file(opts['path']['save_path']+'train_'+opts['name'])
	#progress._log_settings(opts)
	since = time.time()
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(opts['training_params']['num_epochs']):
		progress._log('Epoch {}/{}'.format(epoch, opts['training_params']['num_epochs']-1))
		progress._log('-'*10)
		progress._log("==> Learning rate: ")
		for param_group in optimizer.param_groups:
			progress._log(str(param_group['lr']))

		for phase in ['train', 'val']:

			if phase == 'train':
				scheduler.step()
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_acc = 0

			for sample in tqdm(dataloaders[phase]):
				inputs, labels = sample['lr'], sample['hr']
				to_save = inputs
				inputs = torch.from_numpy(get_patches(inputs, patch_size=(56,56)))

				labels_patches = torch.from_numpy(get_patches(labels, patch_size= (56*4,56*4)))
				inputs = inputs.to(device)
				labels_patches = labels_patches.to(device)

				# zero the parameter grads
				optimizer.zero_grad()

				#forward
				with torch.set_grad_enabled(phase=='train'):

					outputs = model(inputs)
					loss = criterion(outputs, labels_patches)
					recon_out = reconstruct_image_from_patches(outputs.cpu().detach().numpy(), (56*4,56*4))

					if phase == 'train':
						loss.backward()
						optimizer.step()

				#Statistics
				running_loss += loss.item()
				assert loss.item()!=math.inf, "Loss is shooting to infinity"
				running_acc += calculate_psnr(recon_out*255, labels.cpu().detach().numpy()[0,0]*255)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_acc/ dataset_sizes[phase]
			io.imsave(opts['path']['save_path']+'train_'+opts['name']+'/'+'test_result.png', recon_out)
			io.imsave(opts['path']['save_path']+'train_'+opts['name']+'/'+'input.png', to_save[0,0])

			progress._log('{} Loss: {:.4f} psnr: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			if phase =='val' and epoch_acc>best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
				progress._log("$$$$$$----Saving the best model-----$$$$$")
				torch.save(model.state_dict(), opts['path']['save_path']+'train_'+opts['name']+'/'+opts['name']+'_best_model.pt')

	time_elapsed = time.time() -since

	progress._log('Training complete in {:.0f}m {:.0f}s'.format(
	        time_elapsed // 60, time_elapsed % 60))
	progress._log('Best val psnr: {:4f}'.format(best_acc))

	model.load_state_dict(best_model_wts)
	return model

def test_model(model, dataloaders, dataset_sizes, device, opts):
	"""
	Function to test a given model 
	-------------
	Model: Pytorch network 
	dataloaders: Pytorch Dataloaders
	dataset_sizes: The number of samples provided in the dataset
	device: Device where the training needs to be done
	opts: Options file
	"""
	model.eval()
	create_directory(opts.save_path, '')
	progress = log2file(opts['path']['results_path'])
	running_acc = 0

	for sample in tqdm(dataloaders['val']):
		
		_inputs, labels , img_name= sample['lr'], sample['hr'], sample['name'][0].split('.')[0]

		inputs = torch.from_numpy(get_patches(_inputs, patch_size=(56,56))).to(device)
		outputs = model(inputs)
		recon_out = reconstruct_image_from_patches(outputs.cpu().detach().numpy(), (56*4,56*4))
		running_acc = calculate_psnr(recon_out*255, labels.numpy()[0,0]*255)

		io.imsave(opts.save_path+img_name+'.png', recon_out)
		progress._log("PSNR: {}".format(running_acc))
		progress._log("Name: {} \n ------------".format(img_name))

def fine_tune_edsr(model, us_dataloader, dataset_sizes, device, opts):
    """
	Function to perform transfer learning on an EDSR model. Specifically, trainining diffirent parts of the network with different learning rates. 
	----------------
	model: Pytorch network
	us_dataloader: Pytorch Dataloaders
	dataset_sizes: The number of samples provided in the dataset
	device: Device where the training needs to be done
	opts: Options file
    """
    model = model.to(device)
    criterion = nn.L1Loss()

    param_lr_list = [{'params': model.head.parameters(), 'lr': opts['training_params']['lr_head']},
                     {'params': model.body[0:4].parameters(), 'lr': opts['training_params']['lr_body1']},
                     {'params': model.body[4:].parameters(), 'lr': opts['training_params']['lr_body2']},
                     {'params': model.tail.parameters(), 'lr': opts['training_params']['lr_tail']} ]

    optimizer_ft = optim.Adam(param_lr_list, lr = 0.0001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.6)
    model = train_model(model, us_dataloader, dataset_sizes, device, criterion, optimizer_ft, exp_lr_scheduler,opts)

    return model

def get_lr(optim):
    for param_group in optim.param_groups:
        return param_group['lr']

