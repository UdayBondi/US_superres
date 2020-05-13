# Ultrasound Image Super Resolution

This repository consists of code that can be used to run *image super resolution algorithms on ultrasound data*. The code is modular in such a way that you can add other networks into the *models* folder and use them. 
Please take a look at the comments in the code for specific details. 


Details about important folders and files in the repo
* Data: Contains dataloading scripts and data trasnforms 
* Model: Image SR Network scripts
* Options: YAML files containing training and test options
* build_dataset.py: Reproducible script to download and partition different US datasets
* trainer.py: Contains the training code
* **main.py: Run this file for train/test**
  
## Training Instructions: 

* Edit a YAML file given in the Options folder to provide training options.
  * YAML file should contain details about data path, experiment name, network to be used, network design params, learning rate, epochs 
  * Path to save the results of training must be provided. 
* Run the main.py file with the path to YAML file as an argument (-h for help)
  * The training progress in printed in the terminal and is logged into a text file. 
  * A directory with the results of training(log, best model, images) is created at the location specified by the YAML file. 
  

For queries, contact udaybondi007@gmail.com
  
