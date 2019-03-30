import torch
import numpy as np

#make data
x = np.random.random((100, 100, 100))
x[x < 0.9] = 0  # fill most of the array with zeros
#print("total size:", np.prod(x.shape) ,"Shape: ",x.shape,"Non-zero count: ",np.count_nonzero(x),"bytes: ",x.nbytes)

i=np.transpose(np.nonzero(x))
i = torch.LongTensor(i)
#i=np.nonzero(x)
indices = torch.tensor([0, 2])
i=torch.index_select(i, 0, indices)
print (i.shape, i)

v=x[x >0.9]
v=torch.FloatTensor(v)

print(v)

t=torch.sparse.FloatTensor(i,v,[100,100,100])


#i = torch.LongTensor([[0, 1, 1],[2, 0, 2]])
#v = torch.FloatTensor([3, 4, 5])
#x = np.random.random((128, 128, 128))
#sparse=torch.sparse_coo_tensor(i, v, torch.Size([100,100,100]))

import torch
import numpy as np
from scipy.sparse import csr_matrix
import numpy as np
import sparse

def spy_sparse2torch_sparse(data):
    """
    :param data: a scipy sparse csr matrix
    :return: a sparse torch tensor
    """
    samples=data.shape[0]
    print(samples)
    features=data.shape[1]
    print(features)
    values=data.data
    print(values)
    z_dim=data.shape[2]
    #coo_data=data.tocoo()
    coo_data = np.transpose(np.nonzero(x))
    coo_data = torch.LongTensor(i)

    indices=torch.LongTensor([coo_data[:],coo_data[:]])
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features,z_dim])
    return t


x = np.random.random((100, 100, 100))
x[x < 0.9] = 0  # fill most of the array with zeros

s = sparse.COO(x)  # convert to sparse array

#sparese=spy_sparse2torch_sparse(s)
