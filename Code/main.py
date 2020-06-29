"""
Author: Uday Bondi
-------
Train the model
"""

import torch
from data.dataloader import *
from data.data_utils import prepare_data_training
from model.edsr import make_model
from trainer import train_model, fine_tune_edsr
from options.options import get_options, save_options
import argparse
    

def show_model(model):
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def get_args():
    parser = argparse.ArgumentParser(description="Run this to train a model")
    parser.add_argument("opath", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_args()
    opts = get_options(args.opath)
    print("$$ Experiment: ", opts['name'])
    device = torch.device("cuda:"+str(opts['gpu_ids']) if torch.cuda.is_available() else "cpu")
    print("$$ GPU being used:",device)
    print("$$ GPU Memory allocated:", torch.cuda.memory_allocated(device))
    ##----------------
    ## Data loading
    ##----------------
    train_opts = opts['datasets']['train']
    us_datasets, us_data_sizes = get_dataset(train_opts['data_folder'], train_opts)
    print("The size of data being used: ", us_data_sizes)
    us_dataloader = get_data_loader(us_datasets, train_opts['batch_size'])

    ##----------------
    ## Model 
    ##----------------
    model = make_model(opts['network'])
    print('---------Applying Transfer Learning-------')
    model.load_state_dict(torch.load(opts['path']['pretrain_model']+'EDSR_x'+str(opts['scale'])+'.pt'))
    ft_model = fine_tune_edsr(model, us_dataloader, us_data_sizes ,device, opts)

    save_options(args.opath, opts['path']['save_path']+'train_'+opts['name'])




