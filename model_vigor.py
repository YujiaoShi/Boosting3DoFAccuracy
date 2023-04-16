import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import utils
import os
import torchvision.transforms.functional as TF

# from GRU1 import ElevationEsitimate,VisibilityEsitimate,VisibilityEsitimate2,GRUFuse
# from VGG import VGGUnet, VGGUnet_G2S
from VGG import VGGUnet, VGGUnet_G2S, Encoder, Decoder, Decoder2, Decoder4, VGGUnetTwoDec
from jacobian import grid_sample

from models_ford import loss_func
from RNNs import NNrefine, Uncertainty
from swin_transformer import TransOptimizerS2GP_V1, TransOptimizerG2SP_V1
from swin_transformer_cross import TransOptimizerG2SP, TransOptimizerG2SPV2, SwinTransformerSelf
from cross_attention import CrossViewAttention

EPS = utils.EPS


class ModelVIGOR(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(ModelVIGOR, self).__init__()
        '''
        loss_method: 0: direct R T loss 1: feat loss 2: noise aware feat loss
        '''
        self.args = args

        self.level = 3
        self.N_iters = args.N_iters

        self.rotation_range = args.rotation_range

        self.grd_height = -2

        torch.autograd.set_detect_anomaly(True)


    def sat2grd_uv(self, rot, shift_u, shift_v, level, H, W, meter_per_pixel):
        '''
        rot.shape = [B]
        shift_u.shape = [B]
        shift_v.shape = [B]
        H: scalar  height of grd feature map, from which projection is conducted
        W: scalar  width of grd feature map, from which projection is conducted
        '''

        B = shift_u.shape[0]

        shift_u = shift_u / np.power(2, 3 - level)
        shift_v = shift_v / np.power(2, 3 - level)

        S = 512 / np.power(2, 3 - level)

        shift_u = shift_u / 512 * S
        shift_v = shift_v / 512 * S

        ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=shift_u.device),
                                torch.arange(0, S, dtype=torch.float32, device=shift_u.device))
        ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] v dimension
        jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] u dimension

        radius = torch.sqrt((ii-(S/2-0.5 + shift_v.reshape(-1, 1, 1)))**2 + (jj-(S/2-0.5 + shift_u.reshape(-1, 1, 1)))**2)

        theta = torch.atan2(ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1)), jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1)))
        theta = (-np.pi / 2 + (theta) % (2 * np.pi)) % (2 * np.pi)
        theta = (theta + rot[:, None, None] * self.args.rotation_range / 180 * np.pi) % (2 * np.pi)

        theta = theta / 2 / np.pi * W

        # meter_per_pixel = self.meter_per_pixel_dict[city] * 512 / S
        meter_per_pixel = meter_per_pixel * np.power(2, 3-level)
        phimin = torch.atan2(radius * meter_per_pixel[:, None, None], torch.tensor(self.grd_height))
        phimin = phimin / np.pi * H

        uv = torch.stack([theta, phimin], dim=-1)

        return uv

    def project_grd_to_map(self, grd_f, rot, shift_u, shift_v, level, meter_per_pixel):
        '''
        grd_f.shape = [B, C, H, W]
        shift_u.shape = [B]
        shift_v.shape = [B]
        '''
        B, C, H, W = grd_f.size()
        uv = self.sat2grd_uv(rot, shift_u, shift_v, level, H, W, meter_per_pixel)  # [B, S, S, 2]
        grd_f_trans, _ = grid_sample(grd_f, uv)
        return grd_f_trans, uv[..., 0]

    def forward_projImg(self, sat_map, grd_img_left, meter_per_pixel, gt_shift_u=None, gt_shift_v=None, gt_rot=None, mode='train'):

        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param left_camera_k: [B, 3, 3]
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape
        A = sat_map.shape[-1]
        sat_align_cam_trans, _ = self.project_grd_to_map(
            grd_img_left, gt_rot, gt_shift_u, gt_shift_v, level=3, meter_per_pixel=meter_per_pixel)
        # print("sat_align_cam_trans: ",sat_align_cam_trans.size)

        grd_img = transforms.ToPILImage()(sat_align_cam_trans[0])
        grd_img.save('./grd2sat.png')
        sat_align_cam = transforms.ToPILImage()(grd_img_left[0])
        sat_align_cam.save('./grd.png')
        sat = transforms.ToPILImage()(sat_map[0])
        sat.save('./sat.png')

        print('done')


    def Trans_update(self, shift_u, shift_v, heading, grd_feat_proj, sat_feat, mask):
        B = shift_u.shape[0]
        grd_feat_norm = torch.norm(grd_feat_proj.reshape(B, -1), p=2, dim=-1)
        grd_feat_norm = torch.maximum(grd_feat_norm, 1e-6 * torch.ones_like(grd_feat_norm))
        grd_feat_proj = grd_feat_proj / grd_feat_norm[:, None, None, None]

        delta = self.TransRefine(grd_feat_proj, sat_feat, mask)  # [B, 3]
        # print('=======================')
        # print('delta.shape: ', delta.shape)
        # print('shift_u.shape', shift_u.shape)
        # print('=======================')

        shift_u_new = shift_u + delta[:, 0:1]
        shift_v_new = shift_v + delta[:, 1:2]
        heading_new = heading + delta[:, 2:3]

        B = shift_u.shape[0]

        rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_u.requires_grad = True
        rand_v.requires_grad = True
        shift_u_new = torch.where((shift_u_new > -2.5) & (shift_u_new < 2.5), shift_u_new, rand_u)
        shift_v_new = torch.where((shift_v_new > -2.5) & (shift_v_new < 2.5), shift_v_new, rand_v)

        return shift_u_new, shift_v_new, heading_new



