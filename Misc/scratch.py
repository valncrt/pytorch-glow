import math
import numpy as np

arr=np.zeros((3,3,3))
print(arr.shape)

arr[0,0,0]=1
arr[1,0,0]=1
arr[0,1,0]=1
arr[0,0,1]=1
arr[2,1,1]=1
print(arr)
arr2=arr
#print(arr.compress([True, True, True], arr))
print("#######")
arr = arr[~np.all(arr == 0, axis=0)]
arr = arr[~np.all(arr == 0, axis=1)]
#arr = arr[~np.all(arr == 0, axis=2)]
#arr= np.delete(arr,np.where(~arr.any(axis=0))[0], axis=0)

#arr=arr[np.all(arr == 0, axis=2)]
print(arr)
print(arr.shape)

from numba import jit

@jit(nopython=True)
def trim_enum_nb(A):
    for idx in range(A.shape[0]):
        if (A[idx]==0).all():
            break
    return A[:idx]

#arr1=trim_enum_nb(arr)
#print(arr1)
#print(arr1.shape)

print("$$$$$$$$$$$")
import numpy, scipy.sparse
Asp = scipy.sparse.csr_matrix(arr2)
print(Asp)