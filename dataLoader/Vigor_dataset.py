import random

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
# import pandas as pd
# import utils
# import torchvision.transforms.functional as TF
# from torchvision import transforms
# import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import math

import PIL
from PIL import Image
from torch.utils.data import Dataset, Subset

num_thread_workers = 2
# root = '/backup/dataset/VIGOR'
root = '/home/yujiao/dataset/VIGOR'


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

        from sklearn.utils import shuffle
        for rand_state in range(20):
            self.grd_list, self.label, self.delta = shuffle(self.grd_list, self.label, self.delta, random_state=rand_state)

        self.data_size = int(len(self.grd_list))
        self.grd_list = self.grd_list[: self.data_size]
        self.label = self.label[: self.data_size]
        self.delta = self.delta[: self.data_size]
        print('Grd loaded, data size:{}'.format(self.data_size))
        self.label = np.array(self.label)
        self.delta = np.array(self.delta)

    def __len__(self):
        return self.data_size

    def get_grd_sat_img_pair(self, idx):

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
        grd = torch.roll(grd, (torch.round(torch.as_tensor(rotation_angle / 180) * grd.size()[2] / 2).int()).item(),
                         dims=2)

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
        gt_shift_y = row_offset / height * 4  # -L/4 ~ L/4  -1 ~ 1
        gt_shift_x = -col_offset / width * 4  #

        if 'NewYork' in self.grd_list[idx]:
            city = 'NewYork'
        elif 'Seattle' in self.grd_list[idx]:
            city = 'Seattle'
        elif 'SanFrancisco' in self.grd_list[idx]:
            city = 'SanFrancisco'
        elif 'Chicago' in self.grd_list[idx]:
            city = 'Chicago'

        return grd, sat, \
            torch.tensor(gt_shift_x, dtype=torch.float32), \
            torch.tensor(gt_shift_y, dtype=torch.float32), \
            torch.tensor(rotation, dtype=torch.float32), \
            torch.tensor(self.meter_per_pixel_dict[city], dtype=torch.float32)

    def __getitem__(self, idx):

        return self.get_grd_sat_img_pair(idx)


def load_vigor_data(batch_size, area="same", rotation_range=10, train=True):
    """

    Args:
        batch_size: B
        area: same | cross
    """

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
    vigor = VIGORDataset(root, rotation_range, split=area, train=train, transform=(transform_grd, transform_sat))

    if train is True:
        index_list = np.arange(vigor.__len__())
        # np.random.shuffle(index_list)
        train_indices = index_list[0: int(len(index_list) * 0.8)]
        val_indices = index_list[int(len(index_list) * 0.8):]
        training_set = Subset(vigor, train_indices)
        val_set = Subset(vigor, val_indices)

        train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader

    else:
        test_dataloader = DataLoader(vigor, batch_size=batch_size, shuffle=False)

        return test_dataloader



#
# class SatGrdDataset(Dataset):
#     def __init__(self, area, train_test,transform=None,mode="val"):
#         self.root= '../../dataset/VIGOR'
#         self.mode=mode
#         if transform != None:
#             self.satmap_transform = transform[0]
#             self.grdimage_transform = transform[1]
#         self.area = area
#         self.train_test = train_test
#         self.sat_size = [512, 512]  # [320, 320] or [512, 512]
#         self.grd_size = [320, 640]  # [320, 640]  # [224, 1232]
#         label_root = 'splits'
#
#         if self.area == 'same':
#             self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
#             self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
#         elif self.area == 'cross':
#             self.train_city_list = ['NewYork', 'Seattle']
#             if self.train_test == 'train':
#                 self.test_city_list = ['NewYork', 'Seattle']
#             elif self.train_test == 'test':
#                 self.test_city_list = ['SanFrancisco', 'Chicago']
#
#                 # load sat list, the training and test set both contain all satellite images
#         self.train_sat_list = []
#         self.train_sat_index_dict = {}
#         idx = 0
#         for city in self.train_city_list:
#             train_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
#             with open(train_sat_list_fname, 'r') as file:
#                 for line in file.readlines():
#                     self.train_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
#                     self.train_sat_index_dict[line.replace('\n', '')] = idx
#                     idx += 1
#             print('InputData::__init__: load', train_sat_list_fname, idx)
#         self.train_sat_list = np.array(self.train_sat_list)
#         self.train_sat_data_size = len(self.train_sat_list)
#         print('Train sat loaded, data size:{}'.format(self.train_sat_data_size))
#
#         self.test_sat_list = []
#         self.test_sat_index_dict = {}
#         self.__cur_sat_id = 0  # for test
#         idx = 0
#         for city in self.test_city_list:
#             test_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
#             with open(test_sat_list_fname, 'r') as file:
#                 for line in file.readlines():
#                     self.test_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
#                     self.test_sat_index_dict[line.replace('\n', '')] = idx
#                     idx += 1
#             print('InputData::__init__: load', test_sat_list_fname, idx)
#         self.test_sat_list = np.array(self.test_sat_list)
#         self.test_sat_data_size = len(self.test_sat_list)
#         print('Test sat loaded, data size:{}'.format(self.test_sat_data_size))
#
#         # load grd training list and test list.
#         self.train_list = []
#         self.train_label = []
#         self.train_sat_cover_dict = {}
#         self.train_delta = []
#         idx = 0
#         for city in self.train_city_list:
#             # load train panorama list
#             if self.area == 'same':
#                 train_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_train.txt')
#             if self.area == 'cross':
#                 train_label_fname = os.path.join(self.root, label_root, city, 'pano_label_balanced.txt')
#             with open(train_label_fname, 'r') as file:
#                 for line in file.readlines():
#                     data = np.array(line.split(' '))
#                     label = []
#                     for i in [1, 4, 7, 10]:
#                         label.append(self.train_sat_index_dict[data[i]])
#                     label = np.array(label).astype(np.int)
#                     delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
#                     self.train_list.append(os.path.join(self.root, city, 'panorama', data[0]))
#                     self.train_label.append(label)
#                     self.train_delta.append(delta)
#                     if not label[0] in self.train_sat_cover_dict:
#                         self.train_sat_cover_dict[label[0]] = [idx]
#                     else:
#                         self.train_sat_cover_dict[label[0]].append(idx)
#                     idx += 1
#             print('InputData::__init__: load ', train_label_fname, idx)
#
#         # split the original training set into training and validation sets
#         self.train_list, self.val_list, self.train_label, self.val_label, self.train_delta, self.val_delta = train_test_split(
#                 self.train_list, self.train_label, self.train_delta, test_size=0.2, random_state=42)
#
#         self.train_label = np.array(self.train_label)
#         self.train_delta = np.array(self.train_delta)
#         self.val_label = np.array(self.val_label)
#         self.val_delta = np.array(self.val_delta)
#         self.train_data_size = len(self.train_list)
#         self.val_data_size = len(self.val_list)
#         self.trainIdList = [*range(0, self.train_data_size, 1)]
#         self.valIdList = [*range(0, self.val_data_size, 1)]
#
#
#     def __len__(self):
#         if self.mode=="train":
#             return len(self.train_list)
#         else:
#             return len(self.val_list)
#
#     def get_file_list(self):
#         if self.mode=="train":
#             return self.train_list
#         else:
#             return self.val_list
#
#     def __getitem__(self, idx):
#         if self.mode=="train":
#             img_idx=idx
#             # print("processing train set")
#             # img = cv2.imread(self.train_list[img_idx])
#             # img = img.astype(np.float32)
#             # grd_img = cv2.resize(img, (self.grd_size[1], self.grd_size[0]), interpolation=cv2.INTER_AREA)
#             # grd_img = (Image.fromarray(cv2.cvtColor((grd_img).astype(np.uint8), cv2.COLOR_BGR2RGB)))
#
#             grd_img = Image.open(self.train_list[img_idx]).resize((self.grd_size[1], self.grd_size[0]))
#             grd_img = self.grdimage_transform(grd_img)
#
#             pos_index = random.randint(0, 3)
#             img = cv2.imread(self.train_sat_list[self.train_label[img_idx][pos_index]])
#             sat_img1 = cv2.resize(img, (self.sat_size[1], self.sat_size[0]), interpolation=cv2.INTER_AREA)
#             sat_img = (Image.fromarray(cv2.cvtColor((sat_img1).astype(np.uint8), cv2.COLOR_BGR2RGB)))
#
#             # sat_img = Image.open(self.train_sat_list[self.train_label[img_idx][pos_index]]).crop((20, 20, 620, 620)).resize((self.sat_size[1], self.sat_size[0]))
#             sat_img = self.satmap_transform(sat_img)
#
#             [col_offset, row_offset] = self.train_delta[img_idx, pos_index]  # delta = [delta_lat, delta_lon]
#             row_offset_resized = (row_offset / 640 * self.sat_size[0]).astype(np.int32)
#             col_offset_resized = (col_offset / 640 * self.sat_size[0]).astype(np.int32)
#             x, y = np.meshgrid(
#                 np.linspace(-self.sat_size[0] / 2 + row_offset_resized, self.sat_size[0] / 2 + row_offset_resized,
#                             self.sat_size[0]),
#                 np.linspace(-self.sat_size[0] / 2 - col_offset_resized, self.sat_size[0] / 2 - col_offset_resized,
#                             self.sat_size[0]))
#             d = np.sqrt(x * x + y * y)
#             sigma, mu = 4, 0.0
#             img0 = 1000*np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
#             # cv2.imwrite("./GT_vigor.png ",img0)
#             # print("********** pos: ",img0.shape,np.where(img0==img0.max()))
#             a=np.where(img0==img0.max())
#             y=a[1]
#             x=a[0]
#         ####################################################
#
#             gt_shift_y = (x[0] - 256)
#             gt_shift_x = (y[0] - 256)
#
#             # print("************* yx: ",gt_shift_x,gt_shift_y)
#             # cv2.imwrite("./GT.png",img0)
#             #
#             # ###debug
#             # img1=cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
#             # cv2.circle(sat_img1, ( 256, 256), 1, (0, 255, 255), 8)
#             # cv2.circle(sat_img1, (y[0], x[0]), 1, (255, 255, 255), 8)
#             # cv2.imwrite("./offset_vigor.png",sat_img1)
#             return sat_img, grd_img, \
#                torch.tensor(gt_shift_x, dtype=torch.float32).reshape(1), \
#                torch.tensor(gt_shift_y, dtype=torch.float32).reshape(1)
#         else:
#             img_idx = idx
#             # img = cv2.imread(self.val_list[img_idx])
#             # img = img.astype(np.float32)
#             # grd_img = cv2.resize(img, (self.grd_size[1], self.grd_size[0]), interpolation=cv2.INTER_AREA)
#             # grd_img = (Image.fromarray(cv2.cvtColor((grd_img).astype(np.uint8), cv2.COLOR_BGR2RGB)))
#
#             grd_img = Image.open(self.val_list[img_idx]).resize((self.grd_size[1], self.grd_size[0]))
#             grd_img = self.grdimage_transform(grd_img)
#
#             # satellite
#             pos_index = 0  # we use the positive (no semi-positive) satellite images during testing
#             # img = cv2.imread(self.test_sat_list[self.val_label[img_idx][pos_index]])
#             # img = img.astype(np.float32)
#             # sat_img1 = cv2.resize(img, (self.sat_size[1], self.sat_size[0]), interpolation=cv2.INTER_AREA)
#             # sat_img = (Image.fromarray(cv2.cvtColor((sat_img1).astype(np.uint8), cv2.COLOR_BGR2RGB)))
#
#             sat_img = Image.open(self.test_sat_list[self.val_label[img_idx][pos_index]]).resize((self.sat_size[1], self.sat_size[0]))
#             sat_img= self.satmap_transform(sat_img)
#
#
#             # get groundtruth location on the satellite map
#             [col_offset, row_offset] = self.val_delta[img_idx, pos_index]  # delta = [delta_lat, delta_lon]
#             row_offset_resized = (row_offset / 640 * self.sat_size[0]).astype(np.int32)
#             col_offset_resized = (col_offset / 640 * self.sat_size[0]).astype(np.int32)
#             # Gaussian GT
#             x, y = np.meshgrid(
#                 np.linspace(-self.sat_size[0] / 2 + row_offset_resized, self.sat_size[0] / 2 + row_offset_resized,
#                             self.sat_size[0]),
#                 np.linspace(-self.sat_size[0] / 2 - col_offset_resized, self.sat_size[0] / 2 - col_offset_resized,
#                             self.sat_size[0]))
#             d = np.sqrt(x * x + y * y)
#             sigma, mu = 4, 0.0
#             img0 = 1000*np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
#             # cv2.imwrite("./GT_vigor.png ", img0)
#             # print("********** pos: ",img0.shape,np.where(img0==img0.max()))
#             a = np.where(img0 == img0.max())
#             y = a[1]
#             x = a[0]
#             ####################################################
#
#             gt_shift_x = (x[0] - 256)
#             gt_shift_y = (y[0] - 256)
#
#             # print("************* yx: ",gt_shift_x,gt_shift_y)
#             # cv2.imwrite("./GT.png",img0)
#             #
#             # ###debug
#             # img1=cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
#             # cv2.circle(sat_img1, (256, 256), 1, (0, 255, 255), 8)
#             # cv2.circle(sat_img1, (y[0], x[0]), 1, (255, 255, 0), 8)
#             # cv2.imwrite("./offset_vigor.png", sat_img1)
#             # print("debug")
#             # print("debug")
#
#
#             return sat_img, grd_img, \
#                torch.tensor(gt_shift_x, dtype=torch.float32).reshape(1), \
#                torch.tensor(gt_shift_y, dtype=torch.float32).reshape(1),\
#
# class SatGrdDatasetTest(Dataset):
#     def __init__(self, area, train_test,transform=None):
#         self.root = '../../dataset/VIGOR'
#         if transform != None:
#             self.satmap_transform = transform[0]
#             self.grdimage_transform = transform[1]
#         self.train_test=train_test
#         self.area=area
#         self.sat_size = [512, 512]  # [320, 320] or [512, 512]
#         self.grd_size = [320, 640]  # [320, 640]  # [224, 1232]
#         label_root = 'splits'
#
#         if self.area == 'same':
#             self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
#             self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
#         elif self.area == 'cross':
#             self.train_city_list = ['NewYork', 'Seattle']
#             if self.train_test == 'train':
#                 self.test_city_list = ['NewYork', 'Seattle']
#             elif self.train_test == 'test':
#                 self.test_city_list = ['SanFrancisco', 'Chicago']
#
#                 # load sat list, the training and test set both contain all satellite images
#         idx = 0
#
#         self.test_sat_list = []
#         self.test_sat_index_dict = {}
#         for city in self.test_city_list:
#             test_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
#             with open(test_sat_list_fname, 'r') as file:
#                 for line in file.readlines():
#                     self.test_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
#                     self.test_sat_index_dict[line.replace('\n', '')] = idx
#                     idx += 1
#             print('InputData::__init__: load', test_sat_list_fname, idx)
#         self.test_sat_list = np.array(self.test_sat_list)
#         self.test_sat_data_size = len(self.test_sat_list)
#         print('Test sat loaded, data size:{}'.format(self.test_sat_data_size))
#
#         if self.train_test == 'test':
#             self.val_list = []
#             self.val_label = []
#             self.test_sat_cover_dict = {}
#             self.val_delta = []
#             idx = 0
#             for city in self.test_city_list:
#                 # load test panorama list
#                 if self.area == 'same':
#                     test_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test.txt')
#                 if self.area == 'cross':
#                     test_label_fname = os.path.join(self.root, label_root, city, 'pano_label_balanced.txt')
#                 with open(test_label_fname, 'r') as file:
#                     for line in file.readlines():
#                         data = np.array(line.split(' '))
#                         label = []
#                         for i in [1, 4, 7, 10]:
#                             label.append(self.test_sat_index_dict[data[i]])
#                         label = np.array(label).astype(np.int)
#                         delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
#                         self.val_list.append(os.path.join(self.root, city, 'panorama', data[0]))
#                         self.val_label.append(label)
#                         self.val_delta.append(delta)
#                         if not label[0] in self.test_sat_cover_dict:
#                             self.test_sat_cover_dict[label[0]] = [idx]
#                         else:
#                             self.test_sat_cover_dict[label[0]].append(idx)
#                         idx += 1
#                 print('InputData::__init__: load ', test_label_fname, idx)
#
#         self.val_label = np.array(self.val_label)
#         self.val_delta = np.array(self.val_delta)
#         self.val_data_size = len(self.val_list)
#         self.valIdList = [*range(0, self.val_data_size, 1)]
#
#     def __len__(self):
#             return len(self.val_list)
#
#     def get_file_list(self):
#             return self.val_list
#
#     def __getitem__(self, idx):
#         img_idx = idx
#         # print("processing validation set")
#         # img = cv2.imread(self.val_list[img_idx])
#         # img = img.astype(np.float32)
#         # grd_img = cv2.resize(img, (self.grd_size[1], self.grd_size[0]), interpolation=cv2.INTER_AREA)
#         # grd_img = (Image.fromarray(cv2.cvtColor((grd_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)))
#         grd_img = Image.open(self.val_list[img_idx]).resize((self.grd_size[1], self.grd_size[0]))
#         grd_img = self.grdimage_transform(grd_img)
#
#         # satellite
#         pos_index = 0  # we use the positive (no semi-positive) satellite images during testing
#         img = cv2.imread(self.test_sat_list[self.val_label[img_idx][pos_index]])
#         img = img.astype(np.float32)
#         sat_img1 = cv2.resize(img, (self.sat_size[1], self.sat_size[0]), interpolation=cv2.INTER_AREA)
#         sat_img = (Image.fromarray(cv2.cvtColor((sat_img1 * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)))
#         # sat_img = Image.open(self.test_sat_list[self.val_label[img_idx][pos_index]]).resize((self.sat_size[1], self.sat_size[0]))
#
#         sat_img = self.satmap_transform(sat_img)
#
#         # get groundtruth location on the satellite map
#         [col_offset, row_offset] = self.val_delta[img_idx, pos_index]  # delta = [delta_lat, delta_lon]
#         row_offset_resized = (row_offset / 640 * self.sat_size[0]).astype(np.int32)
#         col_offset_resized = (col_offset / 640 * self.sat_size[0]).astype(np.int32)
#         # Gaussian GT
#         x, y = np.meshgrid(
#             np.linspace(-self.sat_size[0] / 2 + row_offset_resized, self.sat_size[0] / 2 + row_offset_resized,
#                         self.sat_size[0]),
#             np.linspace(-self.sat_size[0] / 2 - col_offset_resized, self.sat_size[0] / 2 - col_offset_resized,
#                         self.sat_size[0]))
#         d = np.sqrt(x * x + y * y)
#         sigma, mu = 4, 0.0
#         img0 = 1000 * np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
#
#         a = np.where(img0 == img0.max())
#         y = a[1]
#         x = a[0]
#         ####################################################
#
#         gt_shift_y = (x[0] - 256)
#         gt_shift_x = (y[0] - 256)
#
#
#
#         return sat_img, grd_img, \
#                torch.tensor(gt_shift_x, dtype=torch.float32).reshape(1), \
#                torch.tensor(gt_shift_y, dtype=torch.float32).reshape(1)
#
#
# def load_test_data(batch_size, area="same"):
#     """
#
#     Args:
#         batch_size: B
#         area: same | cross
#
#     Returns: sat_img, grd_img, gt_x ,gt_y  (pixel)
#
#     """
#
#     satmap_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#
#     grdimage_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#
#
#     test_set = SatGrdDatasetTest(area,"test",transform=(satmap_transform, grdimage_transform))
#
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True,
#                             num_workers=num_thread_workers, drop_last=False)
#     return test_loader
#
#
# def load_train_data(batch_size, area="cross",mode="train"):
#     """
#     Args:
#         batch_size:
#         area: cross | same
#         mode: train | val
#
#     Returns: sat_img, grd_img, gt_x ,gt_y  (pixel)
#
#     """
#     satmap_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#
#     grdimage_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#
#     train_set = SatGrdDataset(area,"train",transform=(satmap_transform, grdimage_transform),mode=mode)
#
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
#                               num_workers=num_thread_workers, drop_last=False)
#     return train_loader

# device = torch.device("cuda:0")
# mini_batch=1
# trainloader =load_train_data(mini_batch,area="cross",mode="train")
# # trainloader =load_test_data(mini_batch,area="cross")
# for Loop, Data in enumerate(trainloader, 0):
#     # get the inputs
#     sat_map, grd_left_imgs, gt_shift_u, gt_shift_v= [item.to(device) for item in Data]
#     break
#
# print("sat_map shape: ",sat_map.shape)
# print("grd image shape: ",grd_left_imgs.shape)
# sat_map shape:  torch.Size([2, 3, 512, 512])
# grd image shape:  torch.Size([2, 3, 154, 231])