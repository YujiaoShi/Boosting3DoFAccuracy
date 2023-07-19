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


class ModelVigor(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(ModelVigor, self).__init__()

        self.args = args

        self.level = 3

        self.rotation_range = args.rotation_range

        self.SatFeatureNet = VGGUnet(self.level)
        if self.args.proj == 'CrossAttn':
            self.GrdEnc = Encoder()
            self.GrdDec = Decoder()
            self.Dec4 = Decoder4()
            self.Dec2 = Decoder2()
            self.CVattn = CrossViewAttention(blocks=2, dim=256, heads=4, dim_head=16, qkv_bias=False)
        else:
            self.GrdFeatureNet = VGGUnet(self.level)

        self.grd_height = -2

        if self.args.use_uncertainty:
            self.uncertain_net = Uncertainty()

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

        S = 512 / np.power(2, 3 - level)
        shift_u = shift_u * S / 4
        shift_v = shift_v * S / 4

        ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=shift_u.device),
                                torch.arange(0, S, dtype=torch.float32, device=shift_u.device))
        ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] v dimension
        jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] u dimension

        radius = torch.sqrt((ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1))) ** 2 + (
                    jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1))) ** 2)

        theta = torch.atan2(ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1)),
                            jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1)))
        theta = (-np.pi / 2 + (theta) % (2 * np.pi)) % (2 * np.pi)
        theta = (theta + rot[:, None, None] * self.args.rotation_range / 180 * np.pi) % (2 * np.pi)

        theta = theta / 2 / np.pi * W

        # meter_per_pixel = self.meter_per_pixel_dict[city] * 512 / S
        meter_per_pixel = meter_per_pixel * np.power(2, 3 - level)
        phimin = torch.atan2(radius * meter_per_pixel[:, None, None], torch.tensor(self.grd_height))
        phimin = phimin / np.pi * H

        uv = torch.stack([theta, phimin], dim=-1)

        return uv

    def project_grd_to_map(self, grd_f, grd_c, rot, shift_u, shift_v, level, meter_per_pixel):
        '''
        grd_f.shape = [B, C, H, W]
        shift_u.shape = [B]
        shift_v.shape = [B]
        '''
        B, C, H, W = grd_f.size()
        uv = self.sat2grd_uv(rot, shift_u, shift_v, level, H, W, meter_per_pixel)  # [B, S, S, 2]
        grd_f_trans, _ = grid_sample(grd_f, uv)
        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)
        else:
            grd_c_trans = None
        return grd_f_trans, grd_c_trans, uv[..., 0]

    def triplet_loss(self, corr_maps, gt_shift_u, gt_shift_v): 
        losses = []
        for level in range(len(corr_maps)):

            corr = corr_maps[level]
            B, corr_H, corr_W = corr.shape

            w = torch.round(corr_W / 2 - 0.5 + gt_shift_u / np.power(2, 3 - level)).reshape(-1)
            h = torch.round(corr_H / 2 - 0.5 + gt_shift_v / np.power(2, 3 - level)).reshape(-1)

            pos = corr[range(B), h.long(), w.long()]  # [B]
            # print(pos.shape)
            pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
            # import pdb; pdb.set_trace()
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))

    def forward(self, sat_map, grd_img_left, meter_per_pixel, gt_rot=None, mode='train'):
        '''
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''
        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)
        if self.args.use_uncertainty:
            sat_uncer_list = self.uncertain_net(sat_feat_list)

        grd8, grd4, grd2 = self.GrdEnc(grd_img_left)
        # [H/8, W/8] [H/4, W/4] [H/2, W/2]
        grd_feat_list = self.GrdDec(grd8, grd4, grd2)


        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        grd2sat8, _, u = self.project_grd_to_map(
            grd_feat_list[0], None, gt_rot, shift_u, shift_v, level=0, meter_per_pixel=meter_per_pixel
        )
        grd2sat4, _, _ = self.project_grd_to_map(
            grd_feat_list[1], None, gt_rot, shift_u, shift_v, level=1, meter_per_pixel=meter_per_pixel
        )
        grd2sat2, _, _ = self.project_grd_to_map(
            grd_feat_list[2], None, gt_rot, shift_u, shift_v, level=2, meter_per_pixel=meter_per_pixel
        )

        grd2sat8_attn = self.CVattn(grd2sat8, grd8, u, geo_mask=None)
        grd2sat4_attn = grd2sat4 + self.Dec4(grd2sat8_attn, grd2sat4)
        grd2sat2_attn = grd2sat2 + self.Dec2(grd2sat4_attn, grd2sat2)

        grd_feat_list = [grd2sat8_attn, grd2sat4_attn, grd2sat2_attn]

        corr_maps = []

        if mode == 'train':
            for level in range(len(sat_feat_list)):
                # meter_per_pixel = self.meters_per_pixel[level]

                sat_feat = sat_feat_list[level]
                grd_feat = grd_feat_list[level]

                A = sat_feat.shape[-1]

                crop_size = int(A * 0.4)

                g2s_feat = TF.center_crop(grd_feat, [crop_size, crop_size])
                g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_size, crop_size)

                s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
                corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

                denominator = F.avg_pool2d(sat_feat.pow(2), (crop_size, crop_size), stride=1, divisor_override=1)  # [B, 4W]
                if self.args.use_uncertainty:
                    denominator = torch.sum(denominator, dim=1) * TF.center_crop(sat_uncer_list[level], [corr.shape[1], corr.shape[2]])[:, 0]
                else:
                    denominator = torch.sum(denominator, dim=1)  # [B, H, W]
                denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
                corr = 2 - 2 * corr / denominator
                corr_maps.append(corr)

            return corr_maps[0], corr_maps[1], corr_maps[2]

        
        else:
            level = 2

            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            A = sat_feat.shape[-1]

            crop_size = int(A * 0.4)

            g2s_feat = TF.center_crop(grd_feat, [crop_size, crop_size])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_size, crop_size)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_size, crop_size), stride=1, divisor_override=1)  # [B, 4W]
            if self.args.use_uncertainty:
                denominator = torch.sum(denominator, dim=1) * TF.center_crop(sat_uncer_list[level], [corr.shape[1], corr.shape[2]])[:, 0]
            else:
                denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - (corr_W / 2 + 0.5)) #* meter_per_pixel  # / self.args.shift_range_lon
            pred_v = (max_index // corr_W - (corr_H / 2 + 0.5)) #* meter_per_pixel  # / self.args.shift_range_lat
                
            return pred_u * 2, pred_v * 2  # [B], [B]




