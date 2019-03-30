from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from Misc import binvox_rw

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

def voxels_to_3d_numpy_array(voxels):
    np_array = voxels.data
    #print ("np_array_shape: ",np_array.shape)
    np_array = np_array.astype(np.int8) #not sure this conversion to 8 bits helps because the tensorflow example step it gets converted to 64
    #print ("np_array_shape: ",np_array.shape)
    return np_array

def read_file(file_path):
    with open(file_path, 'rb') as f:
        voxels = binvox_rw.read_as_3d_array(f)
    array=voxels_to_3d_numpy_array(voxels)
    return array

def get_dir_names_for_lables(parent_dir):
    dir_names=[]
    for f in os.listdir(parent_dir):
        if not f.startswith('.'):  #remove hidden files
            dir_names.append(f)
    print (dir_names)
    return dir_names

class Shapenet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, parent_dir, transform=None,file_suffix="surface.binvox"):
        self.parent_dir = parent_dir
        self.transform = transform
        self.file_suffix=file_suffix
        self.labels = get_dir_names_for_lables(parent_dir)
        self.list_label_rows = []
        counter = 0
        df = pd.DataFrame(columns=['label', 'file_to_binvox'])
        for dir_label in self.labels:
            print("label ", dir_label)
            input_dir = parent_dir + "/" + dir_label
            print("input_dir ", input_dir)
            for subdir, dirs, files in os.walk(input_dir):
                for file in files:
                    filepath = subdir + os.sep + file
                    if filepath.endswith(self.file_suffix):
                        counter = counter + 1
                        #print(counter, ' -->', filepath)
                        # voxels=read_file(filepath)
                        self.list_label_rows.append([dir_label, filepath])  # store in a dict

    def __len__(self):
        return len(self.list_label_rows)

    def __getitem__(self, idx):
        label_file_path=self.list_label_rows[idx]
        label=label_file_path[0]
        voxel_array=read_file(label_file_path[1])
        sample = {'label': label, 'voxel_array': voxel_array}

        if self.transform:
            sample = self.transform(sample)

        return sample

parent_dir='/Users/Stephen/work/data/ShapeNet'

dataset = Shapenet(parent_dir)
print ("len(dataset)",len(dataset))
for i in range(len(dataset)):
    sample = dataset[i]
    print(i, sample['label'], sample['voxel_array'])