import torch
import numpy as np


data = []
x = np.random.random((3, 128, 128))
x[x < 0.9] = 0
x1= np.random.random((3, 128, 128))
x1[x1 < 0.9] = 0
data.append(np.asarray(x,dtype=np.float32))
data.append(np.asarray(x1,dtype=np.float32))


x_i=np.transpose(np.nonzero(data))
#print ("x_I: ",x_i)
x_v=np.array(data)
x_v=x_v[x_v > 0.9]

#print("x_v: ",x_v, len(x_v))

x_i=torch.LongTensor(x_i)
x_v=torch.FloatTensor(x_v)
print("x_i: ",x_i)
print("x_v: ",x_v)

size_x_i=x_i.size()
print("Size x_i: ", size_x_i[0])
size_x_v=x_v.size()
print("Size x_v: ", size_x_v[0])
#sparse=torch.sparse.FloatTensor(x_i, x_v, torch.Size([size_x_i[0] + 1, 100, 100]))
device = 'cpu' #'cpu'
#print ("x_v shape: ".x_v.shape)
#sparse=torch.sparse.FloatTensor(x_i.t(), x_v, torch.Size([size_x_i[0] + 1,128,128,128]),device =device)
#print("size ", sparse.size,"Contents :",sparse)




def convert_list_of_tensors_to_sparse_array(data_in):

    index= np.transpose(np.nonzero(data_in))
    index = torch.LongTensor(index)
    index_size=index.size()
    data = np.array(data_in)
    data = data[data > 0.9]  #Special case testing only!!
    data = torch.FloatTensor(data)
    print ("Data shape in method: ", data.shape)
    print ("data shape ", data.shape,"index_size[0]: ",index_size[0])

    sparse_matrix = torch.sparse.FloatTensor(index.t(), data, torch.Size([index_size[0] + 1, 3, 128, 128]))
    return sparse_matrix

test=convert_list_of_tensors_to_sparse_array(data)
print (test)

