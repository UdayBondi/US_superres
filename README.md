# Ultrasound Image Super Resolution

This repository consists of code that can be used to run *image super resolution algorithms on ultrasound data*. The code is modular in such a way that you can add other networks into the *models* folder and use them. 
Please take a look at the comments in the code for specific details. 


Details about important folders and files in the repo
* Data: Contains dataloading scripts and data trasnforms 
* Model: Image SR Network scripts
* Options: YAML files containing training and test options
* build_dataset.py: Reproducible script to download and partition different US datasets
* trainer.py: Contains the training code
* **main.py: Run this file to train the model**
* **test.py: Run this to test .pt model**

## Training Instructions: 

* Edit the train YAML file given in the Options folder to provide training options.
  * YAML file should contain details about data path, experiment name, network to be used, network design params, learning rate, epochs 
  * Ensure the Batch size is 1 to avoid memory errors. 
  * Path to save the results of training must be provided. 
* Ensure that the data folder contains train and val folders in it. The path to data folder must be specified in the YAML file. 
* Run the main.py file with the path to YAML file as an argument (-h for help)
  * The training progress in printed on the terminal and is logged into a text file. 
  * A directory with the results of training(log, best model, images) is created at the location specified by the YAML file. 
  
## Testing Instructions: 

* Edit the test YAML file given in the Options folder to provide test options.
  * YAML file should contain the test folder path, model path, network structure similar to the model you are testing. 
  * Path to save the results of training must be provided. 
* Run the test.py file with the path to YAML file as an argument (-h for help)
  * Every test image's PSNR and SSIM is printed on the terminal and is logged into a text file. 
  * The results directory specified in the YAML will contain a folder with the experiment name. This folder will have the model predictions for every test image along with a log file. 
  
For queries, contact udaybondi007@gmail.com
  
