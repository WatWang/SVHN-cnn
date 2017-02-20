This folder contains the network framework files.
These files contrains the struct of the network and the code 'network_framework_auto' can build nerual network from these files. By change setting in the 'Contraller.py' can choose different framework files.
Framework files must obey following rules:
Written in JSON.
layer name must be "Layer_" + number
For every layer, should be written like this(words following '#' is comment and should not appear in these files):
Convolutional layer example(pooling will be auto done after the convolutional layer):
"Layer_1":{
     "layer_type":"conv",
     # Necessary, layer type can be 'conv' or 'linear'
     "output_size": 64,   
     # Necessary, output size, for conv layer can be [64,64,64],     
     # which mean use conv2d three times and have three 
     # convolutional sub-layer.
     "conv_filter_size":7,
     # Optional, default 3, convolution filter size, for          
     # multi-sub-layer convolution layer exp. filter:[64,
     # 64,64], conv filter size can be[3,5,7]
     "conv_strides":2,
     # Optional, default 1, convolution strides
     "conv_padding":'SAME', 
     # Optional, default 'SAME', convolution padding method, 
     # can be 'SAME' or 'VALID'
     "pool_type":"MAX",
     # Optional, default 'MAX', pooling type, 
     # can be 'MAX' for max pool or 'AVG' for average pool
     "pool_padding":'SAME', 
     # Optional, default 'SAME', pooling padding method, 
     # can be 'SAME' or 'VALID'
     "pool_strides":[1,2,2,1],
     # Optional, default [1,2,2,1], pooling strides
     "pool_k_size":[1,2,2,1],
     # Optional, default [1,2,2,1], pooling kernel size
},
Full connection layer example:
"Layer_2":{
     "layer_type":"linear",
     # Necessary, layer type can be 'conv' or 'linear'
     "output_size": 64,   
     # Necessary, output size, set depend on image size and
     # network model
     	"activation":"relu",
     # Optional, default "relu",  activation function in the 
     # linear layer, can be "relu", "relu6", "elu"
     # more function can be found at:  
     # https://www.tensorflow.org/versions/r0.10
     # /api_docs/python/nn/activation_functions_
     "dropout_switch":True,
     # Optional, default False, set if apply dropout
     "keep_prob": 0.7,   
     # Optional, default 0.8, when dropout switch is on
     # this can set the probility for dropout
}
For the output layer, program will auto set output size as number of classes, and set activation as False.




