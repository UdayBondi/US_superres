"""
Author: Uday Bondi
-------
Test the model 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data.dataloader import *
from data.data_utils import prepare_data_training
from model.edsr import make_model
from trainer import test_model
from skimage import io, color
import argparse
from options.options import get_options, save_options


def get_args():
    parser = argparse.ArgumentParser(description="Run this to test the model")
    parser.add_argument("opath", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    return args

if __name__ == '__main__':

	args = get_args()
	opts = get_options(args.opath)

	device = torch.device("cuda:"+str(opts['gpu_ids']) if torch.cuda.is_available() else "cpu")
	print("The device",device)
    ##----------------
    ## Data loading
    ##----------------
	test_opts = opts['datasets']['test']
	us_datasets, us_data_sizes = get_dataset(test_opts['data_folder'], test_opts, mode='test')
	print("The size of US datasets: ", us_data_sizes)
	dataloaders = {x:torch.utils.data.DataLoader(us_datasets[x], batch_size=test_opts['batch_size'], shuffle=False, num_workers=4) for x in ['val']}
	model = make_model(opts['network']).to(device)
	##----------------
    ## Model 
    ##----------------
	print('---------Testing the model-------')
	model.load_state_dict(torch.load(opts['path']['best_model']))
	test_model(model, dataloaders, us_data_sizes, device, opts)