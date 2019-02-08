import numpy as np
from scipy import ndimage
from functools import reduce
from math import sqrt
import math


array=np.random.rand(128, 128,128)

def reshape_voxel_to_image_fmt(voxel_array,shrink_to=0.25):
    compressed_tensor = ndimage.zoom(voxel_array, shrink_to).astype(np.float32)  # 0.5 = new size is 64,64,64, 0.25=32,32,32
    voxel_cnt = np.prod(compressed_tensor.shape) #total voxels
    squared_rounded = sqrt(voxel_cnt / 3) #first dimension needs to be three
    even_sides = int(math.ceil(squared_rounded)) #get even sides for image
    zeroes_to_add = (3 * even_sides * even_sides) - voxel_cnt
    compressed_tensor = compressed_tensor.reshape(-1)  #make a row to add zeroes too
    compressed_tensor = np.pad(compressed_tensor, (0, zeroes_to_add), 'constant')  #add zeroes
    compressed_tensor_reshaped = compressed_tensor.reshape((3, even_sides, even_sides))
    return compressed_tensor_reshaped

result=reshape_voxel_to_image_fmt(array,.5)
print ("End result:" , result.shape)


