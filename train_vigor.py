#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# from logging import _Level
import os

import torchvision.utils

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataLoader.Vigor_dataset import load_vigor_data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.io as scio

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

from model_vigor import ModelVIGOR

import numpy as np
import os
import argparse


def train(net, args):

    for epoch in range(args.resume, args.epochs):
        net.train()

        trainloader, valloader = load_vigor_data(args.batch_size, area=args.area, rotation_range=args.rotation_range,
                                                 train=True, weak_supervise=True)

        for Loop, Data in enumerate(trainloader, 0):
            # get the inputs

            # sat_map, grd_left_imgs, gt_shift_u, gt_shift_v = [item.to(device) for item in Data]
            grd, sat, gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel = [item.to(device) for item in Data]
            # city = Data[-1]

            net.forward_projImg(sat, grd, meter_per_pixel, gt_shift_u, gt_shift_v, gt_rot, mode='train')

            break

        break

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')

    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')

    parser.add_argument('--rotation_range', type=float, default=180., help='degree')

    # parser.add_argument('--coe_shift_lat', type=float, default=0., help='meters')
    # parser.add_argument('--coe_shift_lon', type=float, default=0., help='meters')
    # parser.add_argument('--coe_heading', type=float, default=0., help='degree')

    parser.add_argument('--coe_triplet', type=float, default=1., help='degree')

    parser.add_argument('--batch_size', type=int, default=4, help='batch size')

    parser.add_argument('--N_iters', type=int, default=2, help='any integer')

    parser.add_argument('--direction', type=str, default='G2SP', help='G2SP' or 'S2GP')

    parser.add_argument('--Optimizer', type=str, default='TransV1', help='LM or SGD')

    parser.add_argument('--proj', type=str, default='CrossAttn', help='geo, polar, nn, CrossAttn')

    parser.add_argument('--use_uncertainty', type=int, default=1, help='0 or 1')

    parser.add_argument('--area', type=str, default='cross', help='same or cross')
    parser.add_argument('--multi_gpu', type=int, default=0, help='0 or 1')

    args = parser.parse_args()

    return args


def getSavePath(args):
    save_path = './ModelsVigor/Corr2D_' + str(args.direction) \
                + '_' + str(args.proj) + '_' + args.area

    if args.use_uncertainty:
        save_path = save_path + '_Uncertainty'

    print('save_path:', save_path)

    return save_path


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    np.random.seed(2022)

    args = parse_args()

    mini_batch = args.batch_size

    save_path = getSavePath(args)

    net = ModelVIGOR(args)
    if args.multi_gpu:
        net = nn.DataParallel(net, dim=0)

    ### cudaargs.epochs, args.debug)
    net.to(device)
    ###########################

    train(net, args)

