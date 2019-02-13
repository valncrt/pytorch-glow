import numpy as np
import sparse

x = np.random.random((100, 100, 100))
x[x < 0.9] = 0  # fill most of the array with zeros
print("total size:", np.prod(x.shape) ,"Shape: ",x.shape,"Non-zero count: ",np.count_nonzero(x),"bytes: ",x.nbytes)

s = sparse.COO(x)  # convert to sparse array

#print("Sparsely:","total size:", np.prod(s.shape) ,"Shape: ",s.shape,"Non-zero count: ",np.count_nonzero(s),"bytes: ",s.nbytes, "Percent reduction: ",(1- s.nbytes/x.nbytes)*100)

print ("\n #####")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
# create a float sparse tensor in CPU
indices = torch.LongTensor([[0, 0, 1], [0, 0, 1]])
#values = torch.FloatTensor([2, 3, 4])
values=np.random.rand(3,3,3)
print ("values: ",values.nbytes,"@@@@@@",values.shape ,"\n",values)
#sizes = [2, 2]
arr=torch.sparse_coo_tensor(indices, values)

#print ("arr: ",arr.nbytes,"@@@@@@",arr.shape ,"\n",arr)

sparse_tensor=torch.sparse.FloatTensor(128, 128,128)
x = np.random.random((128, 128, 128))

print("spare size: ",sparse_tensor.shape)
sparse_tensor.add_(x)