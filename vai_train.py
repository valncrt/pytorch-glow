from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils

import numpy as np
import os
import pdb
import argparse
import time

from invertible_layers import * 
from utils import *
#from pytorch_get_data import *
from torch.utils.data.sampler import SubsetRandomSampler


from scipy import ndimage
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import binvox_rw

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# training
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--depth', type=int, default=10) 
parser.add_argument('--n_levels', type=int, default=3) 
parser.add_argument('--norm', type=str, default='actnorm')
parser.add_argument('--permutation', type=str, default='shuffle')
parser.add_argument('--coupling', type=str, default='affine')
parser.add_argument('--n_bits_x', type=int, default=8)
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--learntop', action='store_true')
parser.add_argument('--n_warmup', type=int, default=20, help='number of warmup epochs')
parser.add_argument('--lr', type=float, default=1e-3)
# logging
parser.add_argument('--print_every', type=int, default=500, help='print NLL every _ minibatches')
parser.add_argument('--test_every', type=int, default=5, help='test on valid every _ epochs')
parser.add_argument('--save_every', type=int, default=5, help='save model every _ epochs')
parser.add_argument('--data_dir', type=str, default='../pixelcnn-pp')
parser.add_argument('--save_dir', type=str, default='/home/ubuntu/work/pytorch-glow/samples/models', help='directory for log / saving')
parser.add_argument('--load_dir', type=str, default=None, help='directory from which to load existing model')
parser.add_argument('--sample_dir', type=str, default=None, help='save samples here')
parser.add_argument('--n_active_dims', type=int, default=10, help='adding because the fork being used had it')

top_level_shapenet_dir='/home/ubuntu/work/data/shapenet/ShapeNetCore.v2/testing'
args = parser.parse_args()
args.n_bins = 2 ** args.n_bits_x

if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# loading / dataset preprocessing
#tf = transforms.Compose([transforms.ToTensor(),
#                         lambda x: x + torch.zeros_like(x).uniform_(0., 1./args.n_bins)])

#train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True,
#    download=True, transform=tf), batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)

#test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False,
#    transform=tf), batch_size=args.batch_size, shuffle=False, num_workers=10)

global_dim_size=32 #based on compression done in voxels_to_3d_numpy_array, 1=128, 0.5=64

plt.ion()   # interactive mode
#parent_dir='/Users/Stephen/work/data/ShapeNet'

def voxels_to_3d_numpy_array(voxels):
    np_array = voxels.data
    #print ("np_array_shape: ",np_array.shape)
    #np_array = np_array.astype(float)
    compressed_tensor=ndimage.zoom(np_array, 0.25).astype(np.float32) #0.5 = new size is 64,64,64, 0.25=32,32,32
    #print (compressed_tensor)
    global_dim_size=compressed_tensor.shape[0]
    #print ("compressed_tensor shape ", compressed_tensor.shape,global_dim_size)

    return compressed_tensor

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
                        self.list_label_rows.append([float(dir_label), filepath])  # store in a dict

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


dataset = Shapenet(top_level_shapenet_dir)

#splitting code from https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)


# construct model and ship to GPU
model = Glow_((args.batch_size, global_dim_size, global_dim_size, global_dim_size), args).cuda()
#model = Glow_((args.batch_size, 128, 128, 128), args)
#print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))

# set up the optimizer
optim = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=45, gamma=0.1)

# data dependant init
#init_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True,
#    download=True, transform=tf), batch_size=512, shuffle=True, num_workers=1)

init_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=1)

#with torch.no_grad():
    #model.eval()
    #for ( _,voxel_array) in init_loader:
        #img = img.cuda()
        #print ("Img  ",voxel_array)
        #objective = torch.zeros_like(img[:, 0, 0, 0])
        #_ = model(img, objective)
        #break

with torch.no_grad():
    model.eval()
    for row in init_loader:
        #print ("Row .. ",row['voxel_array'])
        #row = init_loader[i]
        #print( row['label'], row['voxel_array'])
        voxels=row['voxel_array'].cuda()
        objective = torch.zeros_like(voxels[:, 0, 0, 0])
        #print ("--> type(voxels,type(objective) ",type(voxels),type(objective))
        _ = model(voxels, objective)
        break


# once init is done, we leverage Data Parallel
model = nn.DataParallel(model).cuda()
start_epoch = 0

# load trained model if necessary (must be done after DataParallel)
if args.load_dir is not None: 
    model, optim, start_epoch = load_session(model, optim, args)

# training loop
# ------------------------------------------------------------------------------
for epoch in range(start_epoch, args.n_epochs):
    print('epoch %s' % epoch)
    model.train()
    avg_train_bits_x = 0.
    num_batches = len(train_loader)
    for i, row in enumerate(train_loader):
        t = time.time()
        voxels=row['voxel_array'].cuda()
        objective = torch.zeros_like(voxels[:, 0, 0, 0])
       
        # discretizing cost 
        objective += float(-np.log(args.n_bins) * np.prod(voxels.shape[1:]))
        #print ("voxels.shape[1:]", voxels.shape[1:]) 
        
        # log_det_jacobian cost (and some prior from Split OP)
        z, objective = model(voxels, objective)

        nll = (-objective) / float(np.log(2.) * np.prod(voxels.shape[1:]))
        
        # Generative loss
        nobj = torch.mean(nll)

        optim.zero_grad()
        nobj.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        optim.step()
        avg_train_bits_x += nobj.item()

        # update learning rate
        new_lr = float(args.lr * min(1., (i + epoch * num_batches) / (args.n_warmup * num_batches)))
        for pg in optim.param_groups: pg['lr'] = new_lr

        if (i + 1) % args.print_every == 0: 
            print('avg train bits per pixel {:.4f}'.format(avg_train_bits_x / args.print_every))
            #avg_train_bits_x = 0.
            #sample = model.module.sample()
            #grid = utils.make_grid(sample)
            #utils.save_image(grid, '{}/cifar_Test_{}_{}.png'.format(args.sample_dir, epoch, i // args.print_every))
            sample = model.module.sample()
            print ("sample type ... ", type(sample.cpu().numpy() )) 
            numpy_sample=sample.cpu().numpy()
            from numpy import inf
            numpy_sample[numpy_sample == -inf] = 0
            print ("numpy_sample shape .. ",numpy_sample.shape,numpy_sample.shape[0])
            file_to_save='/home/ubuntu/work/pytorch-glow/samples/train_' + str(epoch) + '_output.numpy'
            np.save (file_to_save,numpy_sample)
        
        print('iteration took {:.4f}'.format(time.time() - t))
        
    # test loop
    # --------------------------------------------------------------------------
    if (epoch + 1) % args.test_every == 0:
        model.eval()
        avg_test_bits_x = 0.
        with torch.no_grad():
            for i, row in enumerate(test_loader): 
                voxels=row['voxel_array'].cuda()
                objective = torch.zeros_like(voxels[:, 0, 0, 0])
               
                # discretizing cost 
                objective += float(-np.log(args.n_bins) * np.prod(voxels.shape[1:]))
                
                # log_det_jacobian cost (and some prior from Split OP)
                z, objective = model(voxels, objective)

                nll = (-objective) / float(np.log(2.) * np.prod(voxels.shape[1:]))
                
                # Generative loss
                nobj = torch.mean(nll)
                avg_test_bits_x += nobj

            print('avg test bits per pixel {:.4f}'.format(avg_test_bits_x.item() / i))

        sample = model.module.sample()
        print ("sample type ... ", type(sample.cpu().numpy() )) 
        numpy_sample=sample.cpu().numpy()
        
        from mayavi import mlab
        import numpy as np

        #s = np.random.rand(20, 20, 20)
        from numpy import inf
        numpy_sample[numpy_sample == -inf] = 0
        print ("numpy_sample shape .. ",numpy_sample.shape,numpy_sample.shape[0])
        file_to_save='/home/ubuntu/work/pytorch-glow/samples/test_' + str(epoch) + '_output.numpy'
        np.save (file_to_save,numpy_sample)
        
        #grid = utils.make_grid(sample)
        #utils.save_image(grid, '{}/cifar_Test_{}.png'.format(args.sample_dir, epoch))
    
    if (epoch + 1) % args.save_every == 0: 
        save_session(model, optim, args, epoch)

        
