import scipy.io
import numpy as np

def importdata( data_name ):
    # This is a function for import local .mat data files, from \data
    # Automatically take the RGB picture to gray
    # Return appended image and label, data size = (num of image, height of image, weight of image, num of color)

    # Set files direction
    file_direction = 'data/' + data_name

    # Load data
    matdata = scipy.io.loadmat(file_direction)

    # Get label and image
    labelnum = np.array(matdata['y'])
    data = np.array(matdata['X'])

    # Reshape the data into set size
    data = np.transpose(data,(3,0,1,2))

    # Turn label into bool type
    label = np.zeros((data.shape[0],10))

    for i in range(data.shape[0]):
        label[i,labelnum[i]-1] = 1

    return (data,label)
