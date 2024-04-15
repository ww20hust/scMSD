import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np

def MixDataset(data_matrix_list,noise_percentage,ratio):
    cell_type_num = len(data_matrix_list)
    data_matrix_list_noise = [None]*cell_type_num 
    for i in range(cell_type_num):
        noise = np.random.uniform(-0.1, 0.1, data_matrix_list[i].shape)
        data_matrix_list_noise[i] = data_matrix_list[i] + noise_percentage*noise*data_matrix_list[i]
    
    mix_sample_spec = np.empty((cell_type_num,data_matrix_list_noise[0].shape[-1]))
    for i in range(cell_type_num):
        if ratio[i] == 0:
            mix_sample_spec[i] = np.zeros(data_matrix_list_noise[0].shape[-1])
        else:
            random_indices = np.random.choice(data_matrix_list_noise[i].shape[0], ratio[i], replace=False)
            random_samples = data_matrix_list_noise[i][random_indices]
            mix_sample_spec[i] = np.sum(random_samples, axis=0)
    mix_sample_spec = np.sum(mix_sample_spec, axis=0)
    mix_sample_spec = mix_sample_spec + np.random.uniform(-0.1, 0.1, mix_sample_spec.shape)*noise_percentage*mix_sample_spec
    return mix_sample_spec