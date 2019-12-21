"""
Author: Uday Bondi
-------
Accuracy metrics 
Logging options
This script can also be used to measure the PSNR between two images when run on its own
"""


import skimage 
import scipy
from matplotlib.image import imread
import scipy.misc as sc_msc
import pdb
import numpy as np
import math
import argparse
import os

	
def psnr(img1, img2):

	#img1 = img1/(np.max(img1)-np.min(img1))
	#img2 = img2/(np.max(img2)-np.min(img2))

	mse = np.mean((img1 - img2)** 2)
	if mse == 0:
		return 100
	#PIXEL_MAX = np.max(img2)
	PIXEL_MAX = 255
	return 20 * math.log10(PIXEL_MAX/math.sqrt(mse))

def get_args():
	parser = argparse.ArgumentParser(description="Calulates PSNR between two images")
	parser.add_argument("hr_img_dir", help="Path to high res image")
	parser.add_argument("lr_img_dir", help="Path to low res image")
	parser.add_argument("-s","--save", help="Save the resized image", action="store_true")
	args = parser.parse_args()

	return args

def get_single_channel(img):

	if len(img.shape) == 3:
		print("\n---Only channel zero selected---\n")
		img = img[:,:,0]

	return img

def calculate_psnr(lr_img, hr_img):
	hr_img = get_single_channel(hr_img)
	lr_img = get_single_channel(lr_img)

	assert hr_img.shape==lr_img.shape, "Both the images dont have the same dimensions"

	return psnr(hr_img, lr_img)

class log2file():

    def __init__(self, file_dir, log_name= 'log.txt'):

        self.dir = file_dir
        self.file = os.path.join(file_dir, log_name)
        f = open(self.file, 'w')
        f.close()

    def _log(self,string):

        print(string)
        with open(self.file, 'a') as f:

            f.write(string)
            f.write('\n')


def main():

	args = get_args()
	hr_img = get_single_channel(imread(args.hr_img_dir))
	lr_img = get_single_channel(imread(args.lr_img_dir))

	if hr_img.shape!=lr_img.shape:
		print("\n----Both the images dont have equal dimensions! Resizing LR image to have the same size as HR image---\n")

		lr_img = scipy.misc.imresize(lr_img, hr_img.shape, interp='nearest')
		if args.save:
			sc_msc.imsave('./resized_image.png',lr_img)
			print("\nResized image has been saved in the same directory as the code\n")

	psnr_v = psnr(hr_img, lr_img)
	print("\n$Result: The PSNR is ",psnr_v)

if __name__=='__main__':
	main()






