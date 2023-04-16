import random

import numpy as np
import os
from PIL import Image
import PIL
from torch.utils.data import Dataset, Subset

import torch
# import pandas as pd
# import utils
# import torchvision.transforms.functional as TF
# from torchvision import transforms
# import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
# from KITTI_dataset import DistanceBatchSampler


import cv2
import math
#
#
# GrdImg_H = 154  # 256 # original: 375 #224, 256
# GrdImg_W = 231  # 1024 # original:1242 #1248, 1024
# GrdOriImg_H = 800
# GrdOriImg_W = 1200
num_thread_workers = 2
root = '../../dataset/VIGOR'

class VIGORDataset(Dataset):
    def __init__(self, root, rotation_range, label_root='splits_new', split='same', train=True, transform=None, pos_only=True):
        self.root = root
        self.rotation_range = rotation_range
        self.label_root = label_root
        self.split = split
        self.train = train
        self.pos_only = pos_only
        if transform != None:
            self.grdimage_transform = transform[0]
            self.satimage_transform = transform[1]

        if self.split == 'same':
            self.city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        elif self.split == 'cross':
            if self.train:
                self.city_list = ['NewYork', 'Seattle']
            else:
                self.city_list = ['SanFrancisco', 'Chicago']

        self.meter_per_pixel_dict = {'NewYork': 0.113248 * 640 / 512,
                                     'Seattle': 0.100817 * 640 / 512,
                                     'SanFrancisco': 0.118141 * 640 / 512,
                                     'Chicago': 0.111262 * 640 / 512}

        # load sat list
        self.sat_list = []
        self.sat_index_dict = {}

        idx = 0
        for city in self.city_list:
            sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', sat_list_fname, idx)
        self.sat_list = np.array(self.sat_list)
        self.sat_data_size = len(self.sat_list)
        print('Sat loaded, data size:{}'.format(self.sat_data_size))

        # load grd list
        self.grd_list = []
        self.label = []
        self.sat_cover_dict = {}
        self.delta = []
        idx = 0
        for city in self.city_list:
            # load grd panorama list
            if self.split == 'same':
                if self.train:
                    label_fname = os.path.join(self.root, self.label_root, city, 'same_area_balanced_train__corrected.txt')
                else:
                    label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test__corrected.txt')
            elif self.split == 'cross':
                label_fname = os.path.join(self.root, self.label_root, city, 'pano_label_balanced__corrected.txt')

            with open(label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.sat_index_dict[data[i]])
                    label = np.array(label).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.label.append(label)
                    self.delta.append(delta)
                    if not label[0] in self.sat_cover_dict:
                        self.sat_cover_dict[label[0]] = [idx]
                    else:
                        self.sat_cover_dict[label[0]].append(idx)
                    idx += 1
            print('InputData::__init__: load ', label_fname, idx)
        self.data_size = len(self.grd_list)
        print('Grd loaded, data size:{}'.format(self.data_size))
        self.label = np.array(self.label)
        self.delta = np.array(self.delta)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # full ground panorama
        try:
            grd = PIL.Image.open(os.path.join(self.grd_list[idx]))
            grd = grd.convert('RGB')
        except:
            print('unreadable image')
            grd = PIL.Image.new('RGB', (320, 640))  # if the image is unreadable, use a blank image
        grd = self.grdimage_transform(grd)

        # generate a random rotation
        rotation = np.random.uniform(low=-1.0, high=1.0)  #
        rotation_angle = rotation * self.rotation_range
        grd = torch.roll(grd, (torch.round(torch.as_tensor(rotation_angle/180) * grd.size()[2]/2).int()).item(), dims=2)

        # satellite
        if self.pos_only:  # load positives only
            pos_index = 0
            sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
            [row_offset, col_offset] = self.delta[idx, pos_index]  # delta = [delta_lat, delta_lon]
        else:  # load positives and semi-positives
            col_offset = 320
            row_offset = 320
            while (np.abs(col_offset) >= 320 or np.abs(
                    row_offset) >= 320):  # do not use the semi-positives where GT location is outside the image
                pos_index = random.randint(0, 3)
                sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
                [row_offset, col_offset] = self.delta[idx, pos_index]  # delta = [delta_lat, delta_lon]

        sat = sat.convert('RGB')
        width_raw, height_raw = sat.size

        sat = self.satimage_transform(sat)
        _, height, width = sat.size()
        row_offset = np.round(row_offset / height_raw * height)
        col_offset = np.round(col_offset / width_raw * width)

        # groundtruth location on the aerial image
        gt_shift_y = row_offset
        gt_shift_x = -col_offset

        if 'NewYork' in self.grd_list[idx]:
            city = 'NewYork'
        elif 'Seattle' in self.grd_list[idx]:
            city = 'Seattle'
        elif 'SanFrancisco' in self.grd_list[idx]:
            city = 'SanFrancisco'
        elif 'Chicago' in self.grd_list[idx]:
            city = 'Chicago'

        return grd, sat, gt_shift_x, gt_shift_y, rotation, self.meter_per_pixel_dict[city]


# ---------------------------------------------------------------------------------
class DistanceBatchSampler:
    def __init__(self, sampler, batch_size, drop_last, train_label):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        # self.required_dis = required_dis
        self.backup = []
        # self.backup_location = torch.tensor([])
        # self.file_name = file_name
        self.train_label = train_label

    def check_add(self, id_list, idx):
        '''
        id_list: a list containing grd image indexes we currently have in a batch
        idx: the grd image index to be determined where or not add to the current batch
        '''

        sat_idx = self.train_label[idx]
        for id in id_list:
            sat_id = self.train_label[id]
            for i in sat_id:
                if i in sat_idx:
                    return False

        return True

    def __iter__(self):
        batch = []

        for idx in self.sampler:

            if self.check_add(batch, idx):
                # add to batch
                batch.append(idx)

            else:
                # add to back up
                self.backup.append(idx)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

                remove = []
                for i in range(len(self.backup)):
                    idx = self.backup[i]

                    if self.check_add(batch, idx):
                        batch.append(idx)
                        remove.append(i)

                for i in sorted(remove, reverse=True):
                    self.backup.remove(self.backup[i])

        if len(batch) > 0 and not self.drop_last:
            yield batch
            print('batched all, left in backup:', len(self.backup))

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore



def load_vigor_data(batch_size, area="same", rotation_range=10, train=True, weak_supervise=True):
    """

    Args:
        batch_size: B
        area: same | cross
    """

    # satmap_transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    #
    # grdimage_transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    transform_grd = transforms.Compose([
        transforms.Resize([320, 640]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    transform_sat = transforms.Compose([
        # resize
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    # torch.manual_seed(202)
    # np.random.seed(202)
    vigor = VIGORDataset(root, rotation_range, split=area, train=train,transform=(transform_grd, transform_sat))

    if train is True:
        index_list = np.arange(vigor.__len__())
        # np.random.shuffle(index_list)
        train_indices = index_list[0: int(len(index_list) * 0.8)]
        val_indices = index_list[int(len(index_list) * 0.8):]
        training_set = Subset(vigor, train_indices)
        val_set = Subset(vigor, val_indices)
        if weak_supervise:
            train_bs = DistanceBatchSampler(torch.utils.data.RandomSampler(training_set), batch_size, True,
                                            vigor.label[train_indices])
            train_dataloader = DataLoader(training_set, batch_sampler=train_bs, num_workers=num_thread_workers)

        else:
            train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader

    else:
        test_dataloader = DataLoader(vigor, batch_size=batch_size, shuffle=False)

        return test_dataloader


# loader, _ = load_vigor_data(1)
#
# for i, data in enumerate(loader):
#     grd, sat, gt_shift_x, gt_shift_y, rotation, city = data
#     print(city)