import os

import torchvision.utils

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataLoader.Oxford_dataset import load_train_data, load_val_data, load_test_data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.io as scio

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

from model_oxford import ModelOxford

import numpy as np
import os
import argparse


class MultiGPU(nn.DataParallel):
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)


def val(net_test, args, save_path, best_rank_result, epoch):
    ### net evaluation state
    net_test.eval()

    dataloader = load_val_data(mini_batch, args.rotation_range, ori_sat_res=args.sat_ori_res)

    pred_us = []
    pred_vs = []
    # pred_oriens = []

    gt_us = []
    gt_vs = []


    for i, data in enumerate(dataloader, 0):

        sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in
                                                                                     data]
        if args.proj == 'CrossAttn':
            pred_u, pred_v = net_test(sat_map, grd_left_imgs, left_camera_k,
                                                                  gt_heading=gt_heading, mode='test')
        else:
            pred_u, pred_v = net_test.corr(sat_map, grd_left_imgs, left_camera_k, gt_heading=gt_heading,
                                                           mode='test')

        pred_us.append(pred_u.data.cpu().numpy())
        pred_vs.append(pred_v.data.cpu().numpy())

        gt_us.append(gt_shift_u[:, 0].data.cpu().numpy())
        gt_vs.append(gt_shift_v[:, 0].data.cpu().numpy())

        if i % 20 == 0:
            print(i)

    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)
    # pred_oriens = np.concatenate(pred_oriens, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)
    # gt_oriens = np.concatenate(gt_oriens, axis=0)

    scio.savemat(os.path.join(save_path, 'result.mat'), {'gt_us': gt_us, 'gt_vs': gt_vs,
                                                         'pred_us': pred_us, 'pred_vs': pred_vs,
                                                         })
    meter_per_pixel = 0.0924 * args.sat_ori_res / 512
    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2) * meter_per_pixel  # [N]
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2) * meter_per_pixel

    # angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    # idx0 = angle_diff > 180
    # angle_diff[idx0] = 360 - angle_diff[idx0]
    #
    # init_angle = np.abs(gt_oriens)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print('Test1 results:')

    line = 'Distance average: (init, pred)' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance))
    print(line)
    f.write(line + '\n')
    
    line ='Distance median: (init, pred)' + str(np.median(init_dis)) + ' ' + str(np.median(distance))
    print(line)
    f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    # print('-------------------------')
    # f.write('------------------------\n')
    #
    # for idx in range(len(angles)):
    #     pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
    #     init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
    #     line = 'angle within ' + str(angles[idx]) + ' degrees (init, pred): ' + str(init) + ' ' + str(pred)
    #     print(line)
    #     f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    result = np.mean(distance)

    net_test.train()

    ### save the best params
    if (result < best_rank_result):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(net_test.state_dict(), os.path.join(save_path, 'Model_best.pth'))

    return result


def test(net_test, args, save_path, epoch, test_set):

    net_test.eval()

    dataloader = load_test_data(mini_batch, args.rotation_range, test_set, ori_sat_res=args.sat_ori_res)

    pred_us = []
    pred_vs = []

    gt_us = []
    gt_vs = []

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in
                                                                                         data]
            if args.proj == 'CrossAttn':
                pred_u, pred_v = net_test(sat_map, grd_left_imgs, left_camera_k, gt_heading=gt_heading, mode='test')
            else:
                pred_u, pred_v = net_test.corr(sat_map, grd_left_imgs, left_camera_k, gt_heading=gt_heading,
                                                               mode='test')

            pred_us.append(pred_u.data.cpu().numpy())
            pred_vs.append(pred_v.data.cpu().numpy())

            gt_us.append(gt_shift_u[:, 0].data.cpu().numpy() )
            gt_vs.append(gt_shift_v[:, 0].data.cpu().numpy() )

            if i % 20 == 0:
                print(i)

    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)

    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)

    scio.savemat(os.path.join(save_path, 'Sat_ori_res' + str(args.sat_ori_res) + 'test' + str(test_set) + '_result.mat'), {'gt_us': gt_us, 'gt_vs': gt_vs,
                                                         'pred_us': pred_us, 'pred_vs': pred_vs,
                                                         })

    meter_per_pixel = 0.0924 * args.sat_ori_res / 512
    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2) * meter_per_pixel  # [N]
    init_dis = np.sqrt(gt_us ** 2 + gt_vs ** 2) * meter_per_pixel

    diff_lats = np.abs((pred_us - gt_us)) * meter_per_pixel
    diff_lons = np.abs((pred_vs - gt_vs)) * meter_per_pixel

    metrics = [1, 3, 5]

    f = open(os.path.join(save_path, 'Sat_ori_res' + str(args.sat_ori_res) + 'test' + str(test_set) + '_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print('Test', test_set, ' results:')

    line = 'Distance average: (init, pred) ' + str(np.mean(init_dis)) + ' ' + str(np.mean(distance))
    print(line)
    f.write(line + '\n')

    line = 'Distance average: (init, pred) ' + str(np.median(init_dis)) + ' ' + str(np.median(distance))
    print(line)
    f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(metrics)):
        pred = np.sum(diff_lats < metrics[idx]) / diff_lats.shape[0] * 100
        init = np.sum(np.abs(gt_us) < metrics[idx]) / gt_us.shape[0] * 100

        line = 'lateral      within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

        pred = np.sum(diff_lons < metrics[idx]) / diff_lons.shape[0] * 100
        init = np.sum(np.abs(gt_vs) < metrics[idx]) / gt_vs.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')


    print('====================================')
    f.write('====================================\n')
    f.close()

    net_test.train()

    return np.mean(distance), np.median(distance)


def triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading):
    
   
    losses = []
    for level in range(len(corr_maps)):

        corr = corr_maps[level]
        B, corr_H, corr_W = corr.shape

        w = torch.round(corr_W / 2 - 0.5 + gt_shift_u / np.power(2, 3 - level)).reshape(-1)
        h = torch.round(corr_H / 2 - 0.5 + gt_shift_v / np.power(2, 3 - level)).reshape(-1)

        pos = corr[range(B), h.long(), w.long()]  # [B]

        pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))

        losses.append(loss)

    return torch.sum(torch.stack(losses, dim=0))


def train(net, args, save_path):
    bestRankResult = 0.0

    for epoch in range(args.resume, args.epochs):
        net.train()

        base_lr = 1e-4

        optimizer = optim.Adam(net.parameters(), lr=base_lr)
        optimizer.zero_grad()

        trainloader = load_train_data(mini_batch, args.rotation_range, ori_sat_res=args.sat_ori_res)

        loss_vec = []

        print('batch_size:', mini_batch, '\n num of batches:', len(trainloader))

        for Loop, Data in enumerate(trainloader, 0):
            # get the inputs

            sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in
                                                                                         Data]


            if args.proj == 'CrossAttn':

                corr_maps0, corr_maps1, corr_maps2 = net(sat_map, grd_left_imgs, left_camera_k, gt_shift_u, gt_shift_v,
                                                gt_heading, mode='train', epoch=epoch)
                loss = triplet_loss([corr_maps0, corr_maps1, corr_maps2], gt_shift_u, gt_shift_v, gt_heading)
            else:
                loss = \
                    net.corr(sat_map, grd_left_imgs, left_camera_k, gt_shift_u, gt_shift_v, gt_heading,
                                 mode='train', epoch=epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # This step is responsible for updating weights


            loss_vec.append(loss.item())

            if Loop % 10 == 9:  #

                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                      ' triplet loss: ' + str(np.round(loss.item(), decimals=4))
                      )

        print('Save Model ...')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pth'))

        current = val(net, args, save_path, bestRankResult, epoch)
        if (current > bestRankResult):
            bestRankResult = current

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')

    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')

    parser.add_argument('--rotation_range', type=float, default=0., help='degree')

    parser.add_argument('--batch_size', type=int, default=14, help='batch size')

    parser.add_argument('--N_iters', type=int, default=2, help='any integer')

    parser.add_argument('--Optimizer', type=str, default='TransV1', help='LM or SGD')

    parser.add_argument('--proj', type=str, default='CrossAttn', help='geo, polar, nn, CrossAttn')

    parser.add_argument('--use_uncertainty', type=int, default=1, help='0 or 1')
    parser.add_argument('--multi_gpu', type=int, default=1, help='0 or 1')

    parser.add_argument('--sat_ori_res', type=int, default=800, help='original satellite image resolution, default is 800')
    parser.add_argument('--grd_rand_shift_pixels', type=int, default=200,
                        help='random shift pixel of the ground camera with respect to its satellite image, default is 200')
    parser.add_argument('--ori_meter_per_pixel', type=float, default=0.0924,
                        help='meter per pixel of the original satellite image provided by the dataset, fixed, plz do not change')

    parser.add_argument('--test_epoch', type=int, default=19, help='19')

    args = parser.parse_args()

    return args


def getSavePath(args):
    save_path = './ModelsOxford/2DoF/' + str(args.proj)

    if args.use_uncertainty:
        save_path = save_path + '_Uncertainty'

    if args.sat_ori_res != 800:
        save_path = save_path + '_SatOriRes' + str(args.sat_ori_res)

    if args.grd_rand_shift_pixels != 800:
        save_path = save_path + '_RandShift' + str(args.grd_rand_shift_pixels)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('save_path:', save_path)

    return save_path


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    np.random.seed(2022)
    
    print(torch.cuda.device_count())

    args = parse_args()

    mini_batch = args.batch_size

    save_path = getSavePath(args)

    net = ModelOxford(args)
    
    if args.multi_gpu:
        net = nn.DataParallel(net, dim=0)
    net.to(device)

    if args.test:
        net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.test_epoch) + '.pth')))

        # mean_dis1, median_dis1 = test(net, args, save_path, epoch=args.test_epoch, test_set=0)
        mean_dis1, median_dis1 = test(net, args, save_path, epoch=args.test_epoch, test_set=1)
        # mean_dis2, median_dis2 = test(net, args, save_path, epoch=args.test_epoch, test_set=2)
        # mean_dis3, median_dis3 = test(net, args, save_path, epoch=args.test_epoch, test_set=3)

    else:

        if args.resume:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth')))
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')

        train(net, args, save_path)

