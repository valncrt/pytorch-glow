
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch.nn.modules.batchnorm import _BatchNorm

import numpy as np
import pdb
import os
from invertible_layers import *
import torch.optim as optim
import argparse
import time

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

def save_session(model, optim, args, epoch):
    path = os.path.join(args.save_dir, str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)

    # save the model and optimizer state
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    torch.save(optim.state_dict(), os.path.join(path, 'optim.pth'))
    print('Successfully saved model')

def load_session(model, optim, epoch, model_dir):
    epoch=str(epoch)
    #print ("os.path.join(model,epoch, 'model.pth')",os.path.join(model,epoch, 'model.pth'))
    model.load_state_dict(torch.load(os.path.join(model_dir, epoch, 'model.pth')))
    optim.load_state_dict(torch.load(os.path.join(model_dir, epoch, 'optim.pth')))
    try:
        model.load_state_dict(torch.load(os.path.join(model,epoch, 'model.pth')))
        optim.load_state_dict(torch.load(os.path.join(model,epoch, 'optim.pth')))
        print('Successfully loaded model')
    except Exception as e:
        pdb.set_trace()
        print('Could not restore session properly')

    return model, optim, start_epoch

global_dim_size=32 #based on compression done in voxels_to_3d_numpy_array, 1=128, 0.5=64
model = Glow_((args.batch_size, global_dim_size, global_dim_size, global_dim_size), args).cuda()
optim = optim.Adam(model.parameters(), lr=1e-3)

model_dir='/Users/Stephen/work/pytorch-glow/samples/models/'
optim_dir='/Users/Stephen/work/pytorch-glow/samples/models/'

model,optim,start_epoch=load_session(model,optim,39,model_dir)