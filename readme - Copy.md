Project: SVHN classifier based on cnn (using tensorflow) 
Highlight:
This is a convolution neural network based on tensorflow and written in python 3.5. 
Load net work framework from JSON files and save network model.
Automatically save train and test abstract.
Designed to work with the SVHN dataset, but also can deal with other dataset by just changing 'importdata' function and configs.
Data:
SVHN(Street View House Number), must be .mat. Can download from http://ufldl.stanford.edu/housenumbers/).
Data should be put into the data folder with default name 'extra_32x32.mat', 'train_32x32.mat',  'test_32x32.mat'.
Install:
This project requires Python 3.5 and the following Python libraries installed:
TensorFlow
NumPy
openpyxl
xlsxwriter
Code:
Controller.py
import_data_with_color.py
network_framework_auto.py
nn_op.py
nn_test.py
nn_train.py
Record_result.py
Set_configs.py
Configs and setting:
Configs.json
With framework settings in 'framework'
Usage:
Run the code 'Controller.py', using default settings will train a net work by using the framework 'VGG-C' which was based on an model from Visual Geometry Group(http://www.robots.ox.ac.uk/~vgg/research/very_deep/). And it will automatically save train and test time cost, accuracy, error list and predicted labels in .xlsx files. Also support pickup and continue train after stop with auto find pre-trained models in model folder.
Based on default settings£¬ using 'extra_32x32.mat' and 'train_32x32.mat' each for twice can get a result greater than 95% predict accuracy.
Reference:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist
https://github.com/aymericdamien/TensorFlow-Examples
https://github.com/dennybritz/cnn-text-classification-tf
https://github.com/indiejoseph/cnn-text-classification-tf-chinese
https://github.com/conceptacid/parking239-vgg-tf
https://deeplearning.net/tutorial/lenet.html
 



