"""
Author: Uday Bondi
-------
Run this script to build a reproducible dataset for super resolution. Please go through the command line arguments for use. 

"""

import glob 
import os
import requests
from tqdm import tqdm
import zipfile
import cv2
import random
import shutil 
import pdb
import argparse


def get_args():
	parser = argparse.ArgumentParser(description="Prepares Dataset for SR")
	parser.add_argument("data", type=str, help="Name of the dataset that needs to be prepared: [fhead / liver]")
	parser.add_argument("dir", type=str, help="Path to target directory for dataset creation")
	parser.add_argument("-m", "--mode", default='bicubic', type=str, help="Interpolation technique: [bicubic / nearest/ area/ linear]")
	parser.add_argument("-s","--scale", default='0.25',type=float,help="Define the scaling factor. Ex: 0.25 means x4 downsampling")
	args = parser.parse_args()

	return args

def download_from_url(file_name, download_path, download_url):
	"""
	Downloads the file from a url if it does not exist in the mentioned path.
	-----------------
	file_name: Name of the file to be downloaded
	download_path: Path where the file needs to be downloaded
	download_url: Url for the file to be downloaded

	"""
	if os.path.exists(download_path+file_name):
		print("$$ Already exists")

	else:
		r = requests.get(download_url, stream=True)
	
		total_file_size = int(r.headers.get('content-length', 0))
		block_size = 1024
		t = tqdm(total = total_file_size, unit='iB', unit_scale=True)
	
		with open(file_name, 'wb') as f:
			for data in r.iter_content(block_size):
				t.update(len(data))
				f.write(data)
			
		t.close()
	
		if total_file_size !=0 and t.n!=total_file_size:
			print("$$ Error downloading, please try again")


def unzip_file(file, path, remove=True):
	"""
	Unzip the mentioned file
	-----------------
	file_name: Name of the file to be downloaded
	path: Path where the file needs to be extracted
	remove: if true, removes the zip file
	"""
	with zipfile.ZipFile(file,"r") as zip_ref:
		zip_ref.extractall(path)
	print("$$ Unzipped")
	
	if remove:
		os.remove(file)
		print(file+" removed")

def get_list_of_files(folder_path,datatype, reg_ex):
	"""
	Returns a list of file paths(complete path)
	"""
	list_of_files = glob.glob(folder_path+datatype+'/'+reg_ex)
	return list_of_files

def create_directory(folder_path, directory):
	"""
	Creates a directory if it does not exist
	
	-----------------
	folder_path: The absolute path to the folder where the dir is to be created
	directory: Name of the directory to be created

	"""
	if not os.path.exists(folder_path+directory):
		os.makedirs(folder_path+directory)

def move_to(list_file_path, destination_path):
	"""
	Moves a set of files to the destination
	
	-----------------
	list_file_path: List containing the complete paths of files
	destination_path: Path where the files can be moved to 
	"""
	for file in list_file_path:
		src = file
		file_name = src.split('/')[-1]
		shutil.move(file, destination_path+file_name)

def downsample_img(img, method, scale):
	"""
	Reduce the resolution of a given image
	-----------------
	img: Numpy array (preferrably loaded using cv2)
	method: Interpolation technique to be used. select from [bicubic / nearest/ area/ linear]
	scale: Downsampling factor
	"""
	if method=='bicubic':
		red_img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
	
	if method=='nearest':
		red_img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)
		
	if method=='area':
		red_img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
		
	if method=='linear':
		red_img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR)

		
	return red_img

def create_LRHR_pairs(folder_path, method, scale):
	"""
	In a folder with images, this function creates two folders HR and LR
	LR folder would have the downsampled version of images in the folder as per the
	specified method of downsampling and scale. 
	-------------------
	folder_path: Path of folder containing images. LR and HR folders are created in this path 
	method: Interpolation technique to be used. select from [bicubic / nearest/ area/ linear]
	scale: scale: Downsampling factor
	"""
	
	create_directory(folder_path,'LR/')
	create_directory(folder_path, 'HR/')
	list_files = [f for f in os.listdir(folder_path) if os.path.isfile(folder_path+f)]
	#pdb.set_trace()
	for filename in list_files:
		
		img = cv2.imread(folder_path+filename, cv2.IMREAD_COLOR)
		red_img = downsample_img(img, method, scale)
		cv2.imwrite(folder_path+'LR/'+filename, red_img)
		shutil.move(folder_path+filename, folder_path+'HR/'+filename)

def prepare_fetal_head_data(data_path, method, scale, data_split={'train': 0.8, 'val': 0.1, 'test': 0.1}):

	"""
	Use this fucntion to prepare fetal head US data for Super resolution. 
	Folders created:
	----fetal_head_data:
		   --train:
			  --HR
			  --LR
		   --val:
			  --HR
			  --LR
		   --test:
			  --HR
			  --LR
	-------------------
	data_path: Path of folder where the data should be prepared
	method: Interpolation technique to be used. select from [bicubic / nearest/ area/ linear]
	scale: scale: Downsampling factor
	data_split: Split for train, test, val. It is a dictionary with train, val and test keys. 
	"""

	
	
	print("-----Downloading data---------")
	trainSet_url = 'https://zenodo.org/record/1327317/files/training_set.zip?download=1'
	testSet_url = 'https://zenodo.org/record/1327317/files/test_set.zip?download=1'
	download_from_url('trainset.zip', data_path, trainSet_url)
	download_from_url('testset.zip', data_path, testSet_url)
	unzip_file('trainset.zip',data_path, remove=True)
	unzip_file('testset.zip',data_path, remove=True)
	
	print("------Preparing Dataset-------")
	create_directory(data_path,'fetal_head_data_'+method+'/')
	new_data_path = data_path + 'fetal_head_data_'+method+'/'
	train_set_list = sorted(get_list_of_files(data_path, 'training_set', '*HC.png'))
	labels_list = sorted(get_list_of_files(data_path, 'training_set', '*Annotation.png'))
	test_set_list = sorted(get_list_of_files(data_path, 'test_set', '*HC.png')) 	# Move this down
	
	#Remove labels:
	create_directory(new_data_path,'labels/')
	move_to(labels_list, new_data_path+'labels/')
	shutil.rmtree(new_data_path+'labels/')             # Comment this if you need the labels in a folder
	
	total_file_list = train_set_list+test_set_list
	modes = ['train/','val/','test/']
	
	#Randomise:
	random.seed(230)
	random.shuffle(total_file_list)
	#Split data 
	data_list = {'train':total_file_list[:int(data_split['train']*len(total_file_list))],
				 'val':total_file_list[int(data_split['train']*len(total_file_list)):int((data_split['train']+data_split['val'])*len(total_file_list))],
				 'test':total_file_list[int((data_split['train']+data_split['val'])*len(total_file_list)):]  }
	#Move data to respective folders
	idx = 0
	for mode in modes:
		create_directory(new_data_path,mode)
		for file_path in data_list[mode.replace('/','')]:
			shutil.move(file_path, new_data_path+mode+str(idx)+'_HC.png')
			idx = idx+1
		
		create_LRHR_pairs(new_data_path+mode, method, scale)
	#remove the previous folders
	shutil.rmtree(data_path+'training_set/')
	shutil.rmtree(data_path+'test_set/')
	print("$$ Done")
	print("$$ Train: "+ str(len(data_list['train']))+" Val: "+str(len(data_list['val']))+" Test: "+str(len(data_list['test'])))

def prepare_liver_us_data(data_path, method, scale, data_split={'train': 0.8, 'val': 0.1, 'test': 0.1}):
	"""
	Use this fucntion to prepare liver US data for Super resolution. 
	Folders created:
	----liver_us_data:
		--Data_ori:
		   --train:
			  --HR
			  --LR
		   --val:
			  --HR
			  --LR
		   --test:
			  --HR
			  --LR
		--Label_ori: (same as data_ori)
	-------------------
	data_path: Path of folder where the liver US data is present
	method: Interpolation technique to be used. select from [bicubic / nearest/ area/ linear]
	scale: scale: Downsampling factor
	data_split: Split for train, test, val. It is a dictionary with train, val and test keys. 
	-------------------
	Note: 1) Make sure the folders inside data_path are named: Data_ori and Label_ori. 
	2) The file extension for data_ori is expected to be .bmp and file extension for label_ori is .png. 
	"""

	print("------Preparing Dataset-------")
	create_directory(data_path,'liver_us_data_'+method+'/')
	new_data_path = data_path + 'liver_us_data_'+method+'/'

	total_file_list = sorted(get_list_of_files(data_path, 'Data_ori', '*.bmp'))
	labels_file_list = sorted(get_list_of_files(data_path, 'Label_ori', '*png'))
	assert len(total_file_list) == len(labels_file_list), "Every image must have a corresponding label"
	modes = ['train/','val/','test/']

	#Randomise:
	random.seed(230)
	group_list = list(zip(total_file_list,labels_file_list))
	random.shuffle(group_list)
	total_file_list, labels_file_list = zip(*group_list)
	#Split data 
	data_list = {'train':total_file_list[:int(data_split['train']*len(total_file_list))], 
				 'val':total_file_list[int(data_split['train']*len(total_file_list)):int((data_split['train']+data_split['val'])*len(total_file_list))],
				 'test':total_file_list[int((data_split['train']+data_split['val'])*len(total_file_list)):]}
	#Move data to respective folders
	for mode in modes:
		create_directory(new_data_path+'Data_ori/',mode)
		create_directory(new_data_path+'Label_ori/',mode)
		for file_path in data_list[mode.replace('/','')]:
			file_name = (file_path.split('/')[-1]).replace('.bmp','')
			shutil.copy(file_path, new_data_path+'Data_ori/'+mode+file_name+'.bmp')
			shutil.copy(data_path+'Label_ori/'+file_name+'.png', new_data_path+'Label_ori/'+mode+file_name+'.png')
		
		create_LRHR_pairs(new_data_path+'Data_ori/'+mode, method, scale)
		create_LRHR_pairs(new_data_path+'Label_ori/'+mode, method, scale)

	print("$$ Done")
	print("$$ Train: "+ str(len(data_list['train']))+" Val: "+str(len(data_list['val']))+" Test: "+str(len(data_list['test'])))



if __name__ == '__main__':
	args = get_args()
	if args.data=='fhead':
		prepare_fetal_head_data(args.dir, args.mode, args.scale)
	if args.data=='liver':
		prepare_liver_us_data(args.dir, args.mode, args.scale)
