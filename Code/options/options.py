import yaml
from shutil import copyfile
def add_defaults(opts):

	opts['mode']= 'train'

def get_options(opts_path):

	with open(opts_path) as f:
		opts = yaml.load(f, Loader=yaml.FullLoader)
	
	add_defaults(opts)

	
	return opts

def save_options(opts_path, to_save_path):

	copyfile(opts_path, to_save_path+'/train_options.yml')


	
