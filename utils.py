import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import h5py


def show_image(x):
        fig = plt.figure()
        plt.imshow(x.cpu().numpy())
        plt.show()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD
    
### write a load_mat code here (you can ask an LLM)
def load_mat(filename):
    with h5py.File(filename, 'r') as f:
        # Assumes the main dataset is named 'ShapeSpace' and stored as a MATLAB-style column-major array
        shape_space_data = f['ShapeSpace'][:]  # extract the dataset
        shape_space_data = np.array(shape_space_data).T  # MATLAB stores data column-wise; transpose to match Python
        return {'ShapeSpace': shape_space_data}

