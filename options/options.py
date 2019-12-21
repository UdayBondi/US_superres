import yaml

def add_defaults(opts):

	opts['mode']= 'train'

def get_options(opts_path):

	with open(opts_path) as f:
    	opts = yaml.load(f, Loader=yaml.FullLoader)
    
    add_defaults(opts)

    return opts
