# https://github.com/ecreager/pytorch-glow
# CUDA_VISIBLE_DEVICES=0 python vai_train_2.py --depth 10 --coupling affine --batch_size 64 --print_every 100 --permutation conv

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
# from pytorch_get_data import *
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
from Misc import binvox_rw
from PIL import Image
import math

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# training
parser.add_argument('--batch_size', type=int, default=64)
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
parser.add_argument('--save_dir', type=str, default='exps', help='directory for log / saving')
parser.add_argument('--load_dir', type=str, default=None, help='directory from which to load existing model')
args = parser.parse_args()
args.n_bins = 2 ** args.n_bits_x

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# loading / dataset preprocessing
# Do I need this transform??
tf = transforms.Compose([transforms.ToTensor(), lambda x: x + torch.zeros_like(x).uniform_(0., 1. / args.n_bins)])

top_level_shapenet_dir = '/home/ubuntu/work/data/shapenet/ShapeNetCore.v2/test_small'  # test_small  testing

image_side_cnt = 1024  # no shrinking... #128 #256 dependency to shrink_to

device = 'cuda'  # 'cpu'


def voxels_to_3d_numpy_array(voxels):
    np_array = voxels.data
    # print ( "array in non-zero ...",np.count_nonzero(np_array))

    return np_array


def read_file(file_path):
    with open(file_path, 'rb') as f:
        voxels = binvox_rw.read_as_3d_array(f)
    array = voxels_to_3d_numpy_array(voxels)
    return array


def get_dir_names_for_lables(parent_dir):
    dir_names = []
    for f in os.listdir(parent_dir):
        if not f.startswith('.'):  # remove hidden files
            dir_names.append(f)
    print(dir_names)
    return dir_names


def convert_list_of_tensors_to_sparse_array(data_in):
    index = np.transpose(np.nonzero(data_in))
    index = torch.LongTensor(index)
    index_size = index.size()
    data = np.array(data_in)
    data = data[data > 0.9]  # Special case testing only!!
    data = torch.FloatTensor(data)
    sparse_matrix = torch.sparse.FloatTensor(index.t(), data,torch.Size([index_size[0] + 1, 128, 128, 128]), device = device)
    return sparse_matrix


def reshape_voxel_to_image_fmt(voxel_array, shrink_to=0.25):
    # voxel_array=np.compress([False, False, False],voxel_array)
    compressed_tensor = ndimage.zoom(voxel_array,
                                     zoom=shrink_to)  # .astype(np.float32)  # 0.5 = new size is 64,64,64, 0.25=32,32,32
    print("reshape_voxel_to_image_fmt in non-zero ...", np.count_nonzero(voxel_array),
          np.count_nonzero(compressed_tensor))
    voxel_cnt = np.prod(compressed_tensor.shape)  # total voxels
    # squared_rounded = math.sqrt(voxel_cnt / 3) #first dimension needs to be three
    # even_sides = int(math.ceil(squared_rounded)) #get even sides for image, will be larger than actually needed
    # even_sides = round_up_to_even(even_sides) #round to even because of stack trace in 'def squeeze_bchw' when sides are not divisable by 2 evenly
    even_sides = image_side_cnt
    zeroes_to_add = (3 * even_sides * even_sides) - voxel_cnt
    compressed_tensor = compressed_tensor.reshape(-1)  # make a row to add zeroes too
    compressed_tensor = np.pad(compressed_tensor, (0, zeroes_to_add), 'constant')  # add zeroes
    compressed_tensor_reshaped = compressed_tensor.reshape((3, even_sides, even_sides))
    return compressed_tensor_reshaped


class Shapenet(Dataset):

    def __init__(self, parent_dir, transform=None, file_suffix="surface.binvox", target_transform=None):
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        self.parent_dir = parent_dir
        self.transform = transform
        self.file_suffix = file_suffix
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
                        # print(counter, ' -->', filepath)
                        # voxels=read_file(filepath)
                        self.list_label_rows.append([float(dir_label), filepath])  # store in a dict
                        self.targets.append(float(dir_label))
                        voxel_array = read_file(filepath)
                        # print ( "voxel_array1 in non-zero ...",np.count_nonzero(voxel_array))
                        voxel_array = reshape_voxel_to_image_fmt(voxel_array, 1)  # 1= no reshape , 0.25 =105, 0.5 =296

                        voxel_array = np.asarray(voxel_array, dtype=np.float32)
                        self.data.append(voxel_array)

        self.data = convert_list_of_tensors_to_sparse_array(self.data)  # sparse array
        self.data = np.vstack(self.data).reshape(-1, 3, image_side_cnt, image_side_cnt)

    def __len__(self):
        return len(self.list_label_rows)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        # print ("IMG: ", type(img), img.shape)
        # img = Image.fromarray(img)
        img = Image.frombytes("RGB", (img.shape[0], img.shape[1]), img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, target = self.data[idx], self.targets[idx]

        return img, target


dataset = Shapenet(top_level_shapenet_dir, transform=tf)

# splitting code from https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

#####################
# construct model and ship to GPU
model = Glow_((args.batch_size, 3, image_side_cnt, image_side_cnt), args).cuda()  # dependency to __getitem__
# print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))

# set up the optimizer
optim = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=45, gamma=0.1)

# data dependant init
# init_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True,download=True, transform=tf), batch_size=512, shuffle=True,num_workers=1)
init_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

with torch.no_grad():
    model.eval()
    for (img, _) in init_loader:
        print("img: ", type(img), img.shape)
        print("_: ", _)
        img = img.cuda()
        objective = torch.zeros_like(img[:, 0, 0, 0])
        _ = model(img, objective)
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
    for i, (img, label) in enumerate(train_loader):
        t = time.time()
        img = img.cuda()
        objective = torch.zeros_like(img[:, 0, 0, 0])

        # discretizing cost
        objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
        # print ("objective: ", objective)

        # log_det_jacobian cost (and some prior from Split OP)
        z, objective = model(img, objective)

        nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))

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
            avg_train_bits_x = 0.
            sample = model.module.sample()
            grid = utils.make_grid(sample)
            utils.save_image(grid, './samples/cifar_Test_{}_{}.png'.format(epoch, i // args.print_every))

        print('iteration took {:.4f}'.format(time.time() - t))

    # test loop
    # --------------------------------------------------------------------------
    if (epoch + 1) % args.test_every == 0:
        model.eval()
        avg_test_bits_x = 0.
        with torch.no_grad():
            for i, (img, label) in enumerate(test_loader):
                img = img.cuda()
                objective = torch.zeros_like(img[:, 0, 0, 0])

                # discretizing cost
                objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))

                # log_det_jacobian cost (and some prior from Split OP)
                z, objective = model(img, objective)

                nll = (-objective) / float(np.log(2.) * np.prod(img.shape[1:]))

                # Generative loss
                nobj = torch.mean(nll)
                avg_test_bits_x += nobj
                print("Generative loss default data: ", avg_test_bits_x)

            print('avg test bits per pixel {:.4f}'.format(avg_test_bits_x.item() / i))

        sample = model.module.sample()
        grid = utils.make_grid(sample)
        utils.save_image(grid, './samples/cifar_Test_{}.png'.format(epoch))

    if (epoch + 1) % args.save_every == 0:
        save_session(model, optim, args, epoch)



