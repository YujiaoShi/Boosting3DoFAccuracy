#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# from logging import _Level
import os

import torchvision.utils

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

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

from model_vigor import ModelVigor

import numpy as np
import os
import argparse

from utils import gps2distance
import time


def test(net_test, args, save_path, epoch):
    ### net evaluation state
    net_test.eval()

    dataloader = load_vigor_data(args.batch_size, area=args.area, rotation_range=args.rotation_range,
                                 train=False)

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    start_time = time.time()
    with torch.no_grad():
        for i, Data in enumerate(dataloader, 0):

            grd, sat, gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel = [item.to(device) for item in Data]

            pred_u, pred_v = net_test(sat, grd, meter_per_pixel, gt_rot, mode='test')

            pred_u = pred_u * meter_per_pixel
            pred_v = pred_v * meter_per_pixel

            pred_us.append(pred_u.data.cpu().numpy())
            pred_vs.append(pred_v.data.cpu().numpy())

            gt_shift_u = gt_shift_u * meter_per_pixel * 512 / 4
            gt_shift_v = gt_shift_v * meter_per_pixel * 512 / 4

            gt_us.append(gt_shift_u.data.cpu().numpy())
            gt_vs.append(gt_shift_v.data.cpu().numpy())

            if i % 20 == 0:
                print(i)
    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size

    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    scio.savemat(os.path.join(save_path, 'result.mat'), {'gt_us': gt_us, 'gt_vs': gt_vs,
                                                         'pred_us': pred_us, 'pred_vs': pred_vs,
                                                         })

    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)  # [N]
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2)


    metrics = [1, 3, 5]

    f = open(os.path.join(save_path, 'results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)
    f.write(line)

    line = 'Distance average: (init, pred)' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance))
    print(line)
    f.write(line + '\n')
    line = 'Distance median: (init, pred)' + str(np.median(init_dis)) + ' ' + str(np.median(distance))
    print(line)
    f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    result = np.mean(distance)

    net_test.train()

    return


def val(dataloader, net_test, args, save_path, epoch, best=0.0):
    ### net evaluation state
    net_test.eval()

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    start_time = time.time()
    with torch.no_grad():
        for i, Data in enumerate(dataloader, 0):

            grd, sat, gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel = [item.to(device) for item in Data]

            pred_u, pred_v = net_test(sat, grd, meter_per_pixel, gt_rot, mode='test')

            pred_u = pred_u * meter_per_pixel
            pred_v = pred_v * meter_per_pixel

            pred_us.append(pred_u.data.cpu().numpy())
            pred_vs.append(pred_v.data.cpu().numpy())

            gt_shift_u = gt_shift_u * meter_per_pixel * 512 / 4
            gt_shift_v = gt_shift_v * meter_per_pixel * 512 / 4

            gt_us.append(gt_shift_u.data.cpu().numpy())
            gt_vs.append(gt_shift_v.data.cpu().numpy())

            if i % 20 == 0:
                print(i)
    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / args.batch_size

    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    scio.savemat(os.path.join(save_path, 'result.mat'), {'gt_us': gt_us, 'gt_vs': gt_vs,
                                                         'pred_us': pred_us, 'pred_vs': pred_vs,
                                                         })

    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)  # [N]
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2)

    metrics = [1, 3, 5]

    f = open(os.path.join(save_path, 'val_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    line = 'Time per image (second): ' + str(duration) + '\n'
    print(line)
    f.write(line)

    line = 'Distance average: (init, pred)' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance))
    print(line)
    f.write(line + '\n')
    line = 'Distance median: (init, pred)' + str(np.median(init_dis)) + ' ' + str(np.median(distance))
    print(line)
    f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    result = np.mean(distance)

    net_test.train()

    if (result < best):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(net.state_dict(), os.path.join(save_path, 'Model_best.pth'))

    print('Finished Val')
    return result


def triplet_loss(corr_maps, gt_shift_u, gt_shift_v):
    
   
    losses = []
    for level in range(len(corr_maps)):

        corr = corr_maps[level]
        B, corr_H, corr_W = corr.shape

        w = torch.round(corr_W / 2 - 0.5 + gt_shift_u * 512 / np.power(2, 3 - level) / 4).reshape(-1)
        h = torch.round(corr_H / 2 - 0.5 + gt_shift_v * 512 / np.power(2, 3 - level) / 4).reshape(-1)

        pos = corr[range(B), h.long(), w.long()]  # [B]
        # print(pos.shape)
        pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
        losses.append(loss)

    return torch.sum(torch.stack(losses, dim=0))


def train(net, args, save_path):
    bestResult = 0.0

    time_start = time.time()
    for epoch in range(args.resume, args.epochs):
        net.train()

        base_lr = 1e-4

        optimizer = optim.Adam(net.parameters(), lr=base_lr)
        optimizer.zero_grad()

        trainloader, valloader = load_vigor_data(args.batch_size, area=args.area, rotation_range=args.rotation_range,
                                                 train=True)

        print('batch_size:', args.batch_size, '\n num of batches:', len(trainloader))

        for Loop, Data in enumerate(trainloader, 0):
            grd, sat, gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel = [item.to(device) for item in Data]

            corr_maps0, corr_maps1, corr_maps2 = net(sat, grd, meter_per_pixel, gt_rot, mode='train')
            loss = triplet_loss([corr_maps0, corr_maps1, corr_maps2], gt_shift_u, gt_shift_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # This step is responsible for updating weights

            if Loop % 10 == 9:
                time_end = time.time()
                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                      ' triplet loss: ' + str(np.round(loss.item(), decimals=4)) +
                      ' Time: ' + str(time_end - time_start)
                      )
                time_start = time_end

        print('Save Model ...')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pth'))

        bestResult = val(valloader, net, args, save_path, epoch, best=bestResult)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=1, help='test with trained model')
    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')

    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')

    parser.add_argument('--rotation_range', type=float, default=0., help='degree')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    parser.add_argument('--proj', type=str, default='CrossAttn', help='geo, polar, nn, CrossAttn')

    parser.add_argument('--use_uncertainty', type=int, default=1, help='0 or 1')
    
    parser.add_argument('--area', type=str, default='cross', help='same or cross')
    parser.add_argument('--multi_gpu', type=int, default=1, help='0 or 1')

    args = parser.parse_args()

    return args


def getSavePath(args):
    save_path = './ModelsVigor/' \
                + str(args.proj) + '_' + args.area

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

    net = ModelVigor(args)
    if args.multi_gpu:
        # net = MultiGPU(net, dim=0)
        net = nn.DataParallel(net, dim=0)

    ### cudaargs.epochs, args.debug)
    net.to(device)
    ###########################

    if args.test:
        net.load_state_dict(torch.load(os.path.join(save_path, 'model_9.pth')))
        current = test(net, args, save_path, epoch=0)

    else:

        if args.resume:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth')))
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')

        train(net, args, save_path)

