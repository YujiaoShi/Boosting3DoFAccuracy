
import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.optim as optim
from dataLoader.KITTI_dataset import load_train_data, load_test1_data, load_test2_data
from models_kitti import Model
import scipy.io as scio

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

import numpy as np
import os
import argparse

import time


def test1(net_test, args, save_path, epoch):
    net_test.eval()

    dataloader = load_test1_data(mini_batch, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    
    pred_lons = []
    pred_lats = []
    pred_oriens = []
    
    gt_lons = []
    gt_lats = []
    gt_oriens = []

    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):

            sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in data[:-1]]

            if args.proj == 'CrossAttn':
                pred_u, pred_v, pred_orien = net_test.CVattn_rot_corr(sat_map, grd_left_imgs, left_camera_k, gt_heading=gt_heading, mode='test')
            else:
                pred_u, pred_v, pred_orien = net_test.rot_corr(sat_map, grd_left_imgs, left_camera_k, gt_heading=gt_heading, mode='test')

            pred_lons.append(pred_u.data.cpu().numpy())
            pred_lats.append(pred_v.data.cpu().numpy())
            pred_oriens.append(pred_orien.data.cpu().numpy())

            gt_lons.append(gt_shift_u[:, 0].data.cpu().numpy() * args.shift_range_lon)
            gt_lats.append(gt_shift_v[:, 0].data.cpu().numpy() * args.shift_range_lat)
            gt_oriens.append(gt_heading[:, 0].data.cpu().numpy() * args.rotation_range)

            # import pdb; pdb.set_trace()

            if i % 20 == 0:
                print(i)
    end_time = time.time()
    duration = (end_time - start_time) / len(dataloader) / mini_batch

    pred_lons = np.concatenate(pred_lons, axis=0)
    pred_lats = np.concatenate(pred_lats, axis=0)
    pred_oriens = np.concatenate(pred_oriens, axis=0)
    
    gt_lons = np.concatenate(gt_lons, axis=0)
    gt_lats = np.concatenate(gt_lats, axis=0)
    gt_oriens = np.concatenate(gt_oriens, axis=0)

    scio.savemat(os.path.join(save_path, 'test1_result.mat'), {'gt_lons': gt_lons, 'gt_lats': gt_lats, 'gt_oriens': gt_oriens,
                                                         'pred_lats': pred_lats, 'pred_lons': pred_lons, 'pred_oriens': pred_oriens})

    distance = np.sqrt((pred_lons - gt_lons) ** 2 + (pred_lats - gt_lats) ** 2)  # [N]

    init_dis = np.sqrt(gt_lats ** 2 + gt_lons ** 2)
    
    diff_lats = np.abs(pred_lats - gt_lats)
    diff_lons = np.abs(pred_lons - gt_lons)
   
    angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]
  
    init_angle = np.abs(gt_oriens)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print('Time per image (second): ' + str(duration) + '\n')
    print('Test1 results:')
    
    print('Distance average: (init, pred)', np.mean(init_dis), np.mean(distance))
    print('Distance median: (init, pred)', np.median(init_dis), np.median(distance))

    print('Lateral average: (init, pred)', np.mean(np.abs(gt_lats)), np.mean(diff_lats))
    print('Lateral median: (init, pred)', np.median(np.abs(gt_lats)), np.median(diff_lats))

    print('Longitudinal average: (init, pred)', np.mean(np.abs(gt_lons)), np.mean(diff_lons))
    print('Longitudinal median: (init, pred)', np.median(np.abs(gt_lons)), np.median(diff_lons))

    print('Angle average (init, pred): ', np.mean(np.abs(gt_oriens)), np.mean(angle_diff))
    print('Angle median (init, pred): ', np.median(np.abs(gt_oriens)), np.median(angle_diff))

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
        init = np.sum(np.abs(gt_lats) < metrics[idx]) / gt_lats.shape[0] * 100

        line = 'lateral within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    for idx in range(len(metrics)):
        pred = np.sum(diff_lons < metrics[idx]) / diff_lons.shape[0] * 100
        init = np.sum(np.abs(gt_lons) < metrics[idx]) / gt_lons.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()

    net_test.train()
    return


def test2(net_test, args, save_path, epoch):
    ### net evaluation state
    net_test.eval()

    dataloader = load_test2_data(mini_batch, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    
    pred_lons = []
    pred_lats = []
    pred_oriens = []
    
    gt_lons = []
    gt_lats = []
    gt_oriens = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):

            sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in data[:-1]]

            if args.proj == 'CrossAttn':
                pred_u, pred_v, pred_orien = net_test.CVattn_rot_corr(sat_map, grd_left_imgs, left_camera_k, gt_heading=gt_heading, mode='test')
            else:
                pred_u, pred_v, pred_orien = net_test.rot_corr(sat_map, grd_left_imgs, left_camera_k, gt_heading=gt_heading, mode='test')

            pred_lons.append(pred_u.data.cpu().numpy())
            pred_lats.append(pred_v.data.cpu().numpy())
            pred_oriens.append(pred_orien.data.cpu().numpy())

            gt_lons.append(gt_shift_u[:, 0].data.cpu().numpy() * args.shift_range_lon)
            gt_lats.append(gt_shift_v[:, 0].data.cpu().numpy() * args.shift_range_lat)
            gt_oriens.append(gt_heading[:, 0].data.cpu().numpy() * args.rotation_range)

            if i % 20 == 0:
                print(i)

    pred_lons = np.concatenate(pred_lons, axis=0)
    pred_lats = np.concatenate(pred_lats, axis=0)
    pred_oriens = np.concatenate(pred_oriens, axis=0)
    
    gt_lons = np.concatenate(gt_lons, axis=0)
    gt_lats = np.concatenate(gt_lats, axis=0)
    gt_oriens = np.concatenate(gt_oriens, axis=0)

    scio.savemat(os.path.join(save_path, 'test2_result.mat'), {'gt_lons': gt_lons, 'gt_lats': gt_lats, 'gt_oriens': gt_oriens,
                                                         'pred_lats': pred_lats, 'pred_lons': pred_lons, 'pred_oriens': pred_oriens})

    distance = np.sqrt((pred_lons - gt_lons) ** 2 + (pred_lats - gt_lats) ** 2)  # [N]

    init_dis = np.sqrt(gt_lats ** 2 + gt_lons ** 2)
    
    diff_lats = np.abs(pred_lats - gt_lats)
    diff_lons = np.abs(pred_lons - gt_lons)
   
    angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]
  
    init_angle = np.abs(gt_oriens)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print('Test2 results:')
    
    print('Distance average: (init, pred)', np.mean(init_dis), np.mean(distance))
    print('Distance median: (init, pred)', np.median(init_dis), np.median(distance))

    print('Lateral average: (init, pred)', np.mean(np.abs(gt_lats)), np.mean(diff_lats))
    print('Lateral median: (init, pred)', np.median(np.abs(gt_lats)), np.median(diff_lats))

    print('Longitudinal average: (init, pred)', np.mean(np.abs(gt_lons)), np.mean(diff_lons))
    print('Longitudinal median: (init, pred)', np.median(np.abs(gt_lons)), np.median(diff_lons))

    print('Angle average (init, pred): ', np.mean(np.abs(gt_oriens)), np.mean(angle_diff))
    print('Angle median (init, pred): ', np.median(np.abs(gt_oriens)), np.median(angle_diff))

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
        init = np.sum(np.abs(gt_lats) < metrics[idx]) / gt_lats.shape[0] * 100

        line = 'lateral within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    for idx in range(len(metrics)):

        pred = np.sum(diff_lons < metrics[idx]) / diff_lons.shape[0] * 100
        init = np.sum(np.abs(gt_lons) < metrics[idx]) / gt_lons.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    # f.write('------------------------\n')
    
    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (init, pred): ' + str(init) + ' ' + str(pred)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    result = np.sum((diff_lats < metrics[0])) / diff_lats.shape[0] * 100

    net_test.train()


    return result


def train(net, lr, args, save_path):

    for epoch in range(args.resume, args.epochs):
        net.train()

        base_lr = lr

        if epoch > 0:
            base_lr = 1e-5

        optimizer = optim.Adam(net.parameters(), lr=base_lr)
        optimizer.zero_grad()

        trainloader = load_train_data(mini_batch, args.shift_range_lat, args.shift_range_lon, args.rotation_range)

        loss_vec = []

        print('batch_size:', mini_batch, '\n num of batches:', len(trainloader))

        for Loop, Data in enumerate(trainloader, 0):
            # get the inputs

            sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in Data[:-1]]
            file_name = Data[-1]

            if args.proj == 'CrossAttn':
                opt_loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                shift_lat_last, shift_lon_last, theta_last, \
                corr_loss = net.CVattn_rot_corr(sat_map, grd_left_imgs, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, mode='train')
            else:
                opt_loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                shift_lat_last, shift_lon_last, theta_last, \
                grd_conf_list, corr_loss = \
                    net.rot_corr(sat_map, grd_left_imgs, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, mode='train')

            loss = opt_loss + corr_loss * torch.exp(-net.coe_T) + net.coe_T + net.coe_R

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # This step is responsible for updating weights

            loss_vec.append(loss.item())

            if Loop % 10 == 9:  #

                level = 2
                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Delta: Level-' + str(level) +
                      ' loss: ' + str(np.round(loss_decrease[level].item(), decimals=4)) +
                      ' lat: ' + str(np.round(shift_lat_decrease[level].item(), decimals=2)) +
                      ' lon: ' + str(np.round(shift_lon_decrease[level].item(), decimals=2)) +
                      ' rot: ' + str(np.round(thetas_decrease[level].item(), decimals=2)))

                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Last : Level-' + str(level) +
                        ' loss: ' + str(np.round(loss_last[level].item(), decimals=4)) +
                        ' lat: ' + str(np.round(shift_lat_last[level].item(), decimals=2)) +
                        ' lon: ' + str(np.round(shift_lon_last[level].item(), decimals=2)) +
                        ' rot: ' + str(np.round(theta_last[level].item(), decimals=2))
                        )

                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) +
                        ' triplet loss: ' + str(np.round(corr_loss.item(), decimals=4)) +
                        ' coe_R: ' + str(np.round(net.coe_R.item(), decimals=2)) +
                        ' coe_T: ' + str(np.round(net.coe_T.item(), decimals=2))
                        )

        print('Save Model ...')

        torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(epoch) + '.pth'))

        test1(net, args, save_path, epoch)
        test2(net, args, save_path, epoch)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 1e-2

    parser.add_argument('--rotation_range', type=float, default=10., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')

    parser.add_argument('--batch_size', type=int, default=3, help='batch size')

    parser.add_argument('--level', type=int, default=3, help='2, 3, 4, -1, -2, -3, -4')
    parser.add_argument('--N_iters', type=int, default=2, help='any integer')

    parser.add_argument('--Optimizer', type=str, default='TransV1G2SP', help='')

    parser.add_argument('--proj', type=str, default='CrossAttn', help='geo, CrossAttn')

    parser.add_argument('--use_uncertainty', type=int, default=1, help='0 or 1')

    args = parser.parse_args()

    return args



def getSavePath(args):
    save_path = './ModelsKitti/3DoF/'\
                + 'lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(
        args.rotation_range) \
                + '_Nit' + str(args.N_iters) + '_' + str(args.Optimizer) + '_' + str(args.proj)

    if args.use_uncertainty:
        save_path = save_path + '_Uncertainty'

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

    args = parse_args()

    mini_batch = args.batch_size

    save_path = getSavePath(args)
    net = Model(args)
    net.to(device)

    if args.test:
        net.load_state_dict(torch.load(os.path.join(save_path, 'model_4.pth')), strict=False)

        test1(net, args, save_path, epoch=0)
        test2(net, args, save_path, epoch=0)

    else:
        if args.resume:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth')))
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')

        lr = args.lr
        train(net, lr, args, save_path)

