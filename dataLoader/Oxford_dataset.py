import random

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
import pandas as pd
import utils
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import math

# root_dir = '/media/yujiao/6TB/FeiWu/Oxford_dataset/Oxford_ground/' # '../../data/Kitti' # '../Data' #'..\\Data' #
root_dir = '/home/users/u6293587/Oxford_dataset/Oxford_ground/'
# root_dir = '/home/yujiao/dataset/Oxford_dataset/Oxford_ground/'

GrdImg_H = 154  # 256 # original: 375 #224, 256
GrdImg_W = 231  # 1024 # original:1242 #1248, 1024
GrdOriImg_H = 800
GrdOriImg_W = 1200
num_thread_workers = 2

# train_file = './dataLoader/train_files.txt'
train_file = './dataLoader/oxford/training.txt'
test1_file = './dataLoader/oxford/test1_j.txt'
test2_file = './dataLoader/oxford/test2_j.txt'
test3_file = './dataLoader/oxford/test3_j.txt'
val_file = "./oxford/validation.txt"


class SatGrdDataset(Dataset):
    """
    output:
    sat img
    left_camera_k
    grd img
    gt_shift_x       pixel
    gt_shift_y       pixel
    theta            (-1,1)
    0.1235 meter_per_pixel
    sat_map shape:  torch.Size([1, 3, 512, 512])
    grd image shape:  torch.Size([1, 3, 154, 231])
    """
    def __init__(self, root, file, transform=None, rotation_range=0, ori_sat_res=800):

        self.root = root
        self.ori_sat_res = ori_sat_res

        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]
        self.rotation_range=rotation_range

        #broken ground image idx
        self.yaws = np.load("./dataLoader/oxford/train_yaw.npy")

        broken = [937, 9050, 11811, 12388, 16584]
        self.train_yaw = []
        for i in range(len(self.yaws)):
            if i not in broken:
                self.train_yaw.append(self.yaws[i])  #loading yaws

        primary = np.array([[619400., 5736195.],
                            [619400., 5734600.],
                            [620795., 5736195.],
                            [620795., 5734600.],
                            [620100., 5735400.]])
        secondary = np.array([[900., 900.],  # tl
                              [492., 18168.],  # bl
                              [15966., 1260.],  # tr
                              [15553., 18528.],  # br
                              [8255., 9688.]])  # c
        n = primary.shape[0]
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        X = pad(primary)
        Y = pad(secondary)
        A, res, rank, s = np.linalg.lstsq(X, Y)

        self.transform = lambda x: unpad(np.dot(pad(x), A))
        self.sat_map = cv2.imread(
            "./dataLoader/oxford/satellite_map_new.png")  # read whole over-view map

        print(self.sat_map.shape)

        with open(file, 'r') as f:
            self.file_name = f.readlines()    # read training file

        trainlist = []
        with open("./dataLoader/oxford/"+'training.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                trainlist.append(content.split(" "))
        self.trainList = trainlist
        self.trainNum = len(trainlist)
        trainarray = np.array(trainlist)
        self.trainUTM = np.transpose(trainarray[:, 2:].astype(np.float64))

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        img_idx=idx
        # =================== read camera intrinsice for left and right cameras ====================

        fx = float(964.828979) * GrdImg_W / GrdOriImg_W
        cx = float(643.788025) * GrdImg_W / GrdOriImg_W
        fy = float(964.828979) * GrdImg_H / GrdOriImg_H
        cy = float(484.407990) * GrdImg_H / GrdOriImg_H

        left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
        # if not self.stereo:
        # =================== read ground img ===================================
        line = self.file_name[idx]
        grdimg=line.split(" ")[0]
        grdimg=root_dir+grdimg

        left_img_name = os.path.join(grdimg)
        grd_img = cv2.imread(left_img_name)

        grd_img = Image.fromarray(cv2.cvtColor(grd_img, cv2.COLOR_BGR2RGB))
        grd_img= self.grdimage_transform(grd_img)

        # =================== position in satellite map ===================================
        image_coord = np.round(self.transform(np.array([[self.trainUTM[0, img_idx], self.trainUTM[1, img_idx]]]))[0])

        # =================== set random offset ===================================
        alpha = 2 * math.pi * random.random()
        r = 200 * np.sqrt(2) * random.random()
        row_offset = int(r * math.cos(alpha))
        col_offset = int(r * math.sin(alpha))

        sat_coord_row = int(image_coord[1] + row_offset)  # sat center location
        sat_coord_col = int(image_coord[0] + col_offset)

        # print(sat_coord_row, sat_coord_col)
        # =================== crop satellite map ===================================
        img = self.sat_map[sat_coord_row - int(self.ori_sat_res//2) - int(self.ori_sat_res//4):sat_coord_row + int(self.ori_sat_res//2) + int(self.ori_sat_res//4),
              sat_coord_col - int(self.ori_sat_res//2) - int(self.ori_sat_res//4):sat_coord_col + int(self.ori_sat_res//2) + int(self.ori_sat_res//4),
              :]  # load at each side extra 200 pixels to avoid blank after rotation

        #=================== set rotation random ===================================
        theta = np.random.uniform(-1, 1)
        # rdm= theta * self.rotation_range/ np.pi * 180        # radian  ground truth
        rdm = theta * self.rotation_range                      #degree
        #======================================================================

        # rotate_angle = self.train_yaw[img_idx] / np.pi * 180-90 +rdm*180/np.pi   # degree
        rotate_angle = self.train_yaw[img_idx] / np.pi * 180 - 90 + rdm            # degree
        H, W = img.shape[:2]
        assert H == W == self.ori_sat_res // 2 * 3
        rot_matrix = cv2.getRotationMatrix2D((int(H // 2), int(W // 2)), rotate_angle, 1)  # rotate satellite image
        # NOT SURE THE ORDER OF H * W IN ABOVE AND BELOW IS CORRECT OR NOT. HERE IT DOES NOT MATTER, BECAUSE H==W.
        img = cv2.warpAffine(img, rot_matrix, (H, W))
        # rot_matrix = cv2.getRotationMatrix2D((600, 600), rotate_angle, 1)
        # img = cv2.warpAffine(img, rot_matrix, (1200, 1200))
        img = img[int(self.ori_sat_res//4):-int(self.ori_sat_res//4), int(self.ori_sat_res//4):-int(self.ori_sat_res//4), :]
        sat_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)# 0.1235
        # img[:, :, 0] -= 103.939  # Blue
        # img[:, :, 1] -= 116.779  # Green
        # img[:, :, 2] -= 123.6  # Red
        sat_img = Image.fromarray(cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB))
        sat_img = self.satmap_transform(sat_img)
        row_offset_resized = int(np.round((int(self.ori_sat_res//2) + row_offset) / self.ori_sat_res * 512 - 256))
        col_offset_resized = int(np.round((int(self.ori_sat_res//2) + col_offset) / self.ori_sat_res * 512 - 256))

        #=================== set ground truth ===================================
        x, y = np.meshgrid(np.linspace(-256 + col_offset_resized, 256 + col_offset_resized, 512),
                           np.linspace(-256 + row_offset_resized, 256 + row_offset_resized, 512))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 4, 0.0
        img0 = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        rot_matrix = cv2.getRotationMatrix2D((256, 256), rotate_angle, 1)
        img0 = cv2.warpAffine(img0, rot_matrix, (512, 512))
        a = np.where(img0 == img0.max())
        y = a[0]
        x = a[1]

        gt_shift_x = (x[0]-256)   # pixel  right positive parallel heading
        gt_shift_y = (y[0]-256)   # pixel  down positive vertical heading         0.1235 meter_per_pixel

        return sat_img, left_camera_k, grd_img, \
               torch.tensor(gt_shift_x, dtype=torch.float32).reshape(1), \
               torch.tensor(gt_shift_y, dtype=torch.float32).reshape(1), \
               torch.tensor(theta, dtype=torch.float32).reshape(1) \


class SatGrdDatasetVal(Dataset):

    def __init__(self, root, transform=None, rotation_range=0, ori_sat_res=800):

        self.root = root
        self.ori_sat_res = ori_sat_res

        if transform is not None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]
        self.rotation_range=rotation_range

        #broken ground image idx
        self.yaws = np.load("./dataLoader/oxford/val_yaw.npy") # for debug

        primary = np.array([[619400., 5736195.],
                            [619400., 5734600.],
                            [620795., 5736195.],
                            [620795., 5734600.],
                            [620100., 5735400.]])
        secondary = np.array([[900., 900.],  # tl
                              [492., 18168.],  # bl
                              [15966., 1260.],  # tr
                              [15553., 18528.],  # br
                              [8255., 9688.]])  # c
        n = primary.shape[0]
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        X = pad(primary)
        Y = pad(secondary)
        A, res, rank, s = np.linalg.lstsq(X, Y)

        self.transform = lambda x: unpad(np.dot(pad(x), A))
        self.sat_map = cv2.imread("./dataLoader/oxford/satellite_map_new.png")  # read whole over-view map
        vallist = []
        with open("./dataLoader/oxford/"+'validation.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                vallist.append(content.split(" "))
        self.valList = vallist
        self.valNum = len(vallist)
        valarray = np.array(vallist)
        self.valUTM = np.transpose(valarray[:, 2:].astype(np.float64))

    def __len__(self):
        return self.valNum

    def get_file_list(self):
        return self.valList

    def __getitem__(self, idx):
        img_idx=idx
        # =================== read camera intrinsice for left and right cameras ====================

        fx = float(964.828979) * GrdImg_W / GrdOriImg_W
        cx = float(643.788025) * GrdImg_W / GrdOriImg_W
        fy = float(964.828979) * GrdImg_H / GrdOriImg_H
        cy = float(484.407990) * GrdImg_H / GrdOriImg_H

        left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))

        # =================== read ground img ===================================
        line = self.valList[idx]
        grdimg=root_dir+line[0]

        left_img_name = os.path.join(grdimg)
        grd_img = cv2.imread(left_img_name)

        grd_img = Image.fromarray(cv2.cvtColor(grd_img, cv2.COLOR_BGR2RGB))
        grd_img= self.grdimage_transform(grd_img)

        # =================== position in satellite map ===================================
        image_coord = np.round(self.transform(np.array([[self.valUTM[0, img_idx], self.valUTM[1, img_idx]]]))[0])
        col_split = int((image_coord[0]) // (self.ori_sat_res//2))
        if np.round(image_coord[0] - (self.ori_sat_res//2) * col_split) < (self.ori_sat_res//4):
            col_split -= 1
        col_pixel = int(np.round(image_coord[0] - (self.ori_sat_res//2) * col_split))

        row_split = int((image_coord[1]) // (self.ori_sat_res//2))
        if np.round(image_coord[1] - (self.ori_sat_res//2) * row_split) < (self.ori_sat_res//4):
            row_split -= 1
        row_pixel = int(np.round(image_coord[1] - (self.ori_sat_res//2) * row_split))

        img = self.sat_map[row_split * int(self.ori_sat_res//2) - int(self.ori_sat_res//4):row_split * int(self.ori_sat_res//2) + self.ori_sat_res + int(self.ori_sat_res//4),
              col_split * int(self.ori_sat_res//2) - int(self.ori_sat_res//4):col_split * int(self.ori_sat_res//2) + self.ori_sat_res + int(self.ori_sat_res//4),
              :]  # read extra 200 pixels at each side to avoid blank after rotation

        # =================== set rotation random ===================================
        theta = np.random.uniform(-1, 1)
        # rdm= theta * self.rotation_range/ np.pi * 180        # radian  ground truth
        rdm = theta * self.rotation_range  # degree
        # ======================================================================
        rotate_angle = self.yaws[img_idx] / np.pi * 180-90 +rdm  # degree
        H, W = img.shape[:2]
        assert H == W == (self.ori_sat_res//2) * 3
        rot_matrix = cv2.getRotationMatrix2D((int(H//2), int(W//2)), rotate_angle, 1)  # rotate satellite image
        # NOT SURE THE ORDER OF H * W IN ABOVE AND BELOW IS CORRECT OR NOT. HERE IT DOES NOT MATTER, BECAUSE H==W.
        img = cv2.warpAffine(img, rot_matrix, (H, W))
        img = img[int(self.ori_sat_res//4):-int(self.ori_sat_res//4), int(self.ori_sat_res//4):-int(self.ori_sat_res//4), :]
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        # img[:, :, 0] -= 103.939  # Blue
        # img[:, :, 1] -= 116.779  # Green
        # img[:, :, 2] -= 123.6  # Red
        sat_img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        sat_img=self.satmap_transform(sat_img)

        row_offset_resized = int(-(row_pixel / self.ori_sat_res * 512 - 256))
        col_offset_resized = int(-(col_pixel / self.ori_sat_res * 512 - 256))
    ################################################
        x, y = np.meshgrid(np.linspace(-256 + col_offset_resized, 256 + col_offset_resized, 512),
                           np.linspace(-256 + row_offset_resized, 256 + row_offset_resized, 512))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 4, 0.0
        img0 = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        rot_matrix = cv2.getRotationMatrix2D((256, 256), rotate_angle, 1)
        img0 = cv2.warpAffine(img0, rot_matrix, (512, 512))
        # print("********** pos: ",img0.shape,np.where(img0==img0.max()))
        a=np.where(img0==img0.max())
        y=a[0]
        x=a[1]
    ####################################################

        gt_shift_x=(x[0]-256)   # right positive parallel heading
        gt_shift_y=(y[0]-256)   #down positive vertical heading

        return sat_img, left_camera_k, grd_img, \
               torch.tensor(gt_shift_x, dtype=torch.float32).reshape(1), \
               torch.tensor(gt_shift_y, dtype=torch.float32).reshape(1), \
               torch.tensor(theta, dtype=torch.float32).reshape(1) \


class SatGrdDatasetTest(Dataset):

    def __init__(self, root, transform=None, rotation_range=0, test=0, ori_sat_res=800):
        self.root = root
        self.ori_sat_res = ori_sat_res
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]
        self.rotation_range = rotation_range

        # broken ground image idx

        # self.yaws = np.load("./dataLoader/oxford/train_yaw.npy")

        with open('./dataLoader/oxford/test_yaw.npy', 'rb') as f:
            self.val_yaw = np.load(f)
        if test == 2:
                self.val_yaw=self.val_yaw[1672+1:1672+1708+1]
        elif test == 3:
                self.val_yaw=self.val_yaw[1672+1708+1:1672+1708+1708+1]

        primary = np.array([[619400., 5736195.],
                            [619400., 5734600.],
                            [620795., 5736195.],
                            [620795., 5734600.],
                            [620100., 5735400.]])
        secondary = np.array([[900., 900.],  # tl
                              [492., 18168.],  # bl
                              [15966., 1260.],  # tr
                              [15553., 18528.],  # br
                              [8255., 9688.]])  # c
        n = primary.shape[0]
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        X = pad(primary)
        Y = pad(secondary)
        A, res, rank, s = np.linalg.lstsq(X, Y)

        self.transform = lambda x: unpad(np.dot(pad(x), A))
        # self.sat_map = cv2.imread(
        #     "./dataLoader/oxford/satellite_map_new.png")  # read whole over-view map

        self.sat_map = cv2.imread(
            "./dataLoader/oxford/satellite_map_new.png")  # for debug

        test_2015_08_14_14_54_57 = []
        with open('./dataLoader/oxford/test1_j.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                test_2015_08_14_14_54_57.append(content.split(" "))
        test_2015_08_12_15_04_18 = []
        with open('./dataLoader/oxford/test2_j.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                test_2015_08_12_15_04_18.append(content.split(" "))
        test_2015_02_10_11_58_05 = []
        with open('./dataLoader/oxford/test3_j.txt', 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                content = line[:-1]
                test_2015_02_10_11_58_05.append(content.split(" "))

        if test==0:
            testlist = test_2015_08_14_14_54_57 + test_2015_08_12_15_04_18 + test_2015_02_10_11_58_05
        elif test==1:
            testlist = test_2015_08_14_14_54_57
        elif test ==2:
            testlist = test_2015_08_12_15_04_18
        else:
            testlist = test_2015_02_10_11_58_05
        self.valList = testlist
        self.valNum = len(testlist)
        valarray = np.array(testlist)
        self.valUTM = np.transpose(valarray[:,2:].astype(np.float64))
        print("len.........: ",self.valNum )

    def __len__(self):
        return self.valNum

    def get_file_list(self):
        return self.valList

    def __getitem__(self, idx):
        img_idx = idx
        # =================== read camera intrinsice for left and right cameras ====================
        fx = float(964.828979) * GrdImg_W / GrdOriImg_W
        cx = float(643.788025) * GrdImg_W / GrdOriImg_W
        fy = float(964.828979) * GrdImg_H / GrdOriImg_H
        cy = float(484.407990) * GrdImg_H / GrdOriImg_H

        left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
        # =================== read ground img ===================================
        line = self.valList[idx]
        grdimg = root_dir + line[0]

        left_img_name = os.path.join(grdimg)
        grd_img = cv2.imread(left_img_name)

        grd_img = Image.fromarray(cv2.cvtColor(grd_img, cv2.COLOR_BGR2RGB))
        grd_img = self.grdimage_transform(grd_img)

        # grd_img= grd_img.astype(np.float32)
        # grd_img[:, :, 0] -= 103.939  # Blue
        # grd_img[:, :, 1] -= 116.779  # Green
        # grd_img[:, :, 2] -= 123.6  # Red

        # =================== position in satellite map ===================================
        image_coord = np.round(self.transform(np.array([[self.valUTM[0, img_idx], self.valUTM[1, img_idx]]]))[0])
        col_split = int((image_coord[0]) // (self.ori_sat_res//2))
        if np.round(image_coord[0] - (self.ori_sat_res//2) * col_split) < (self.ori_sat_res//4):
            col_split -= 1
        col_pixel = int(np.round(image_coord[0] - (self.ori_sat_res//2) * col_split))

        row_split = int((image_coord[1]) // (self.ori_sat_res//2))
        if np.round(image_coord[1] - (self.ori_sat_res//2) * row_split) < (self.ori_sat_res//4):
            row_split -= 1
        row_pixel = int(np.round(image_coord[1] - (self.ori_sat_res//2) * row_split))

        img = self.sat_map[int(row_split * (self.ori_sat_res//2) - (self.ori_sat_res//4)):int(row_split * (self.ori_sat_res//2) + self.ori_sat_res + (self.ori_sat_res//4)),
              col_split * (self.ori_sat_res//2) - (self.ori_sat_res//4):col_split * (self.ori_sat_res//2) + self.ori_sat_res + (self.ori_sat_res//4),
              :]  # read extra 200 pixels at each side to avoid blank after rotation

        # =================== set rotation random ===================================
        theta = np.random.uniform(-1, 1)
        rdm = theta * self.rotation_range  # degree
        # ======================================================================

        rotate_angle = self.val_yaw[img_idx] / np.pi * 180 - 90+rdm  # degree
        H, W = img.shape[:2]
        assert H == W == self.ori_sat_res//2*3
        rot_matrix = cv2.getRotationMatrix2D((int(H//2), int(W//2)), rotate_angle, 1)  # rotate satellite image
        img = cv2.warpAffine(img, rot_matrix, (H, W))
        # CANNOT gaurantee the order of H & W in above two lines are correct, but its fine here, because H==W
        img = img[int(self.ori_sat_res//4):-int(self.ori_sat_res//4), int(self.ori_sat_res//4):-int(self.ori_sat_res//4), :]
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        # img[:, :, 0] -= 103.939  # Blue
        # img[:, :, 1] -= 116.779  # Green
        # img[:, :, 2] -= 123.6  # Red
        sat_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        sat_img = self.satmap_transform(sat_img)

        row_offset_resized = int(-(row_pixel / self.ori_sat_res * 512 - 256))
        col_offset_resized = int(-(col_pixel / self.ori_sat_res * 512 - 256))

        ################################################
        x, y = np.meshgrid(np.linspace(-256 + col_offset_resized, 256 + col_offset_resized, 512),
                           np.linspace(-256 + row_offset_resized, 256 + row_offset_resized, 512))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 4, 0.0
        img0 = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        rot_matrix = cv2.getRotationMatrix2D((256, 256), rotate_angle, 1)
        img0 = cv2.warpAffine(img0, rot_matrix, (512, 512))
        # print("********** pos: ",img0.shape,np.where(img0==img0.max()))
        a = np.where(img0 == img0.max())
        y = a[0]
        x = a[1]
        ####################################################
        gt_shift_x = (x[0] - 256)  # right positive parallel heading
        gt_shift_y = (y[0] - 256)  # down positive vertical heading
        # print("************* yx: ",gt_shift_x,gt_shift_y)
        # cv2.imwrite("./GT.png", img0)

        return sat_img, left_camera_k, grd_img, \
               torch.tensor(gt_shift_x, dtype=torch.float32).reshape(1), \
               torch.tensor(gt_shift_y, dtype=torch.float32).reshape(1), \
               torch.tensor(theta, dtype=torch.float32).reshape(1) \




"""
load dataset, shuffle=False, for load_test_data, testNum=0, test all test datasets
"""



def load_val_data(batch_size, rotation_range=0, ori_sat_res=800):

    print("loding validation dataset..............")
    satmap_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    val_set = SatGrdDatasetVal(root=root_dir,
                               transform=(satmap_transform, grdimage_transform),
                               rotation_range=rotation_range,
                               ori_sat_res=ori_sat_res)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_thread_workers, drop_last=False)
    return val_loader


def load_test_data(batch_size, rotation_range=0, testNum=0, ori_sat_res=800):
    print("loading test dataset..............")
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    test_set = SatGrdDatasetTest(root=root_dir,
                                 transform=(satmap_transform, grdimage_transform),
                                 rotation_range=rotation_range, test=testNum,
                                 ori_sat_res=ori_sat_res)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_thread_workers, drop_last=False)
    return test_loader


def load_train_data(batch_size, rotation_range=0, ori_sat_res=800):
    print("loding train dataset..............")
    satmap_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    train_set = SatGrdDataset(root=root_dir, file=train_file,
                              transform=(satmap_transform, grdimage_transform),
                              rotation_range=rotation_range,
                              ori_sat_res=ori_sat_res)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)
    return train_loader
