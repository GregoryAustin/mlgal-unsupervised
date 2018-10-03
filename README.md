# Code for honours project at UCT 
To run the code, you need the datasets. For both the unlabeled and labeled dataset contact the author of this project. 

Once you have the datasets usage is as follows: 
For training the autoencoder: 
python conv_ae_og.py # for training on unlabeled galaxy dataset 
python conv_ae_og.py --ds 2 # for training on labeled galaxy dataset 

For the galaxy dataset: 
python classifier.py --experiment 1 # for experiment 1 
python classifier.py --experiment 2 # for experiment 2
python classifier.py --experiment 3 # for experiment 3
python classifier.py --experiment 4 # for experiment 4

To use classifier on CIFAR 10 add the following flag: 
--ds 2


Author:
Gregory Austin
ASTGRE002
u14039712@tuks.co.za
