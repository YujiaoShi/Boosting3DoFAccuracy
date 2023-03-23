import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import utils
import torchvision.transforms.functional as TF

from VGG import VGGUnet, VGGUnet_G2S, Encoder, Decoder, Decoder2, Decoder4, VGGUnetTwoDec
from jacobian import grid_sample

from RNNs import NNrefine, Uncertainty
from swin_transformer import TransOptimizerG2SP_V1
from swin_transformer_cross import TransOptimizerG2SP, TransOptimizerG2SPV2
from cross_attention import CrossViewAttention

EPS = utils.EPS


class ModelOxford(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(ModelOxford, self).__init__()

        self.args = args

        self.level = 3
        self.N_iters = args.N_iters

        self.rotation_range = args.rotation_range

        self.SatFeatureNet = VGGUnet(self.level)
        if self.args.proj == 'nn':
            self.GrdFeatureNet = VGGUnet_G2S(self.level)
        elif self.args.proj == 'CrossAttn':
            self.GrdEnc = Encoder()
            self.GrdDec = Decoder()
            self.Dec4 = Decoder4()
            self.Dec2 = Decoder2()
            self.CVattn = CrossViewAttention(blocks=2, dim=256, heads=4, dim_head=16, qkv_bias=False)
        else:
            self.GrdFeatureNet = VGGUnet(self.level)

        self.meter_per_pixel = 0.0924 * self.args.sat_ori_res / 512  # 0.144375

        if self.args.Optimizer == 'NN':
            self.NNrefine = NNrefine()
        elif self.args.Optimizer == 'TransV1':
            self.TransRefine = TransOptimizerG2SP_V1()
        elif self.args.Optimizer == 'TransVfeat':
            self.TransRefine = TransOptimizerG2SP(pose_from='feature')
        elif self.args.Optimizer == 'TransVattn':
            self.TransRefine = TransOptimizerG2SP(pose_from='attention')
        elif self.args.Optimizer == 'TransV2feat':
            self.TransRefine = TransOptimizerG2SPV2(pose_from='feature')
        elif self.args.Optimizer == 'TransV2attn':
            self.TransRefine = TransOptimizerG2SPV2(pose_from='attention')

        if self.args.use_uncertainty:
            self.uncertain_net = Uncertainty()

        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.

    def get_warp_sat2real(self, satmap_sidelength):
        # realword: X: East, Y:Down, Z: North   origin is set to the ground plane
        # satmap_sidelength = 512
        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor([u0, v0]).cuda()

        # meter_per_pixel = 0.1235
        meter_per_pixel = self.meter_per_pixel * 512 / satmap_sidelength
        R = torch.tensor([[1, 0], [0, -1]]).float().cuda()
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # Z = 0.4023 * torch.ones_like(XY[..., :1])
        Y = torch.zeros_like(XZ[..., :1])

        XYZ = torch.cat([XZ[..., 0:1], Y, XZ[..., 1:]], dim=-1).unsqueeze(
            dim=0)  # [1, satmap_sidelength, satmap_sidelength, 3]

        return XYZ

    def seq_warp_real2camera(self, ori_shift_u, ori_shift_v, ori_heading, XYZ_1, ori_camera_k, grd_H, grd_W, ori_grdH,
                             ori_grdW):
        B = ori_heading.shape[0]

        shift_u_meters = ori_shift_u * self.meter_per_pixel
        shift_v_meters = ori_shift_v * self.meter_per_pixel
        heading = ori_heading * self.rotation_range / 180 * np.pi
        cos = torch.cos(heading)
        sin = torch.sin(heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        # R =torch.cat([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=-1) # shape = [B,9]
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)
        R = R.view(B, 3, 3)  # shape = [B,3,3]
        R_inv = torch.inverse(R)
        camera_height = utils.get_camera_height()
        height = camera_height * torch.ones_like(shift_u_meters)
        T = torch.cat([-shift_u_meters, height, shift_v_meters, ], dim=-1)  # shape = [B, 3]
        # T = torch.unsqueeze(T, dim=-1)  # shape = [B,3,1]

        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH

        KR_FL = torch.matmul(camera_k, R_inv)  # [B, 3, 3]
        XYZc = XYZ_1[:, :, :, :] + T[:, None, None, :]  # [B, H, W, 3]
        uv1 = torch.sum(KR_FL[:, None, None, :, :] * XYZc[:, :, :, None, :], dim=-1)  # [B, H, W, 3]

        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        uv = uv1[..., :2] / uv1_last  # shape = [B, H, W, 2]
        mask = torch.greater(uv1_last, torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)

        return uv, mask

    def project_grd_to_map(self, grd_f, shift_u, shift_v, heading, camera_k, satmap_sidelength, ori_grdH, ori_grdW):

        B, C, H, W = grd_f.size()

        XYZ_1 = self.get_warp_sat2real(satmap_sidelength)  # [ sidelength,sidelength,4]

        uv, mask = self.seq_warp_real2camera(shift_u, shift_v, heading, XYZ_1, camera_k, H, W, ori_grdH, ori_grdW)  # [B, S, E, H, W,2]
        grd_f_trans, _ = grid_sample(grd_f, uv)

        if self.args.proj == 'CrossAttn':
            return grd_f_trans, mask, uv[..., 0]
        else:
            return grd_f_trans, mask

    def forward(self, sat_map, grd_img_left, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                mode='train', file_name=None, gt_depth=None):

        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param left_camera_k: [B, 3, 3]
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape
        A = sat_map.shape[-1]
        sat_align_cam_trans = self.project_grd_to_map(
            grd_img_left, None, gt_shift_u, gt_shift_v, gt_heading, left_camera_k, 512, ori_grdH, ori_grdW)
        # print("sat_align_cam_trans: ",sat_align_cam_trans.size)

        grd_img = transforms.ToPILImage()(sat_align_cam_trans[0])
        grd_img.save('./grd2sat.png')
        sat_align_cam = transforms.ToPILImage()(grd_img_left[0])
        sat_align_cam.save('./grd.png')
        sat = transforms.ToPILImage()(sat_map[0])
        sat.save('./sat.png')


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



    def triplet_loss(self, corr_maps, gt_shift_u, gt_shift_v, gt_heading):
        # cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
        # sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
        #
        # gt_delta_x = - gt_shift_u[:, 0] * self.args.shift_range_lon
        # gt_delta_y = - gt_shift_v[:, 0] * self.args.shift_range_lat
        #
        # gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
        # gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos

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

    def corr(self, sat_map, grd_img_left, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                 mode='train', epoch=None):
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

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        corr_maps = []

        for level in range(len(sat_feat_list)):
            # meter_per_pixel = self.meters_per_pixel[level]

            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            A = sat_feat.shape[-1]

            heading = gt_heading

            grd_feat_proj, mask = self.project_grd_to_map(
                grd_feat, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

            # crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
            # crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)

            radius_pixel = int(np.ceil(200 * np.sqrt(2) / self.args.ori_sat_res * 512))
            crop_H = int(A - radius_pixel * 2 / np.power(2, 3 - level))
            crop_W = int(A - radius_pixel * 2 / np.power(2, 3 - level))

            g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            if self.args.use_uncertainty:
                denominator = torch.sum(denominator, dim=1) * TF.center_crop(sat_uncer_list[level], [corr.shape[1], corr.shape[2]])[:, 0]
            else:
                denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2) #* meter_per_pixel  # / self.args.shift_range_lon
            pred_v = (max_index // corr_W - corr_H / 2) #* meter_per_pixel  # / self.args.shift_range_lat

        if mode == 'train':
            return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
        else:
            return pred_u * 2, pred_v * 2  # [B], [B]


    def forward(self, sat_map, grd_img_left, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                 mode='train', epoch=None):
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

        sat8, sat4, sat2 = sat_feat_list

        grd8, grd4, grd2 = self.GrdEnc(grd_img_left)
        # [H/8, W/8] [H/4, W/4] [H/2, W/2]
        grd_feat_list = self.GrdDec(grd8, grd4, grd2)


        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        grd2sat8, mask, u = self.project_grd_to_map(
            grd_feat_list[0], shift_u, shift_v, gt_heading, left_camera_k, sat8.shape[-1], ori_grdH, ori_grdW,
        )
        grd2sat4, _, _ = self.project_grd_to_map(
            grd_feat_list[1], shift_u, shift_v, gt_heading, left_camera_k, sat4.shape[-1], ori_grdH, ori_grdW,
        )
        grd2sat2, _, _ = self.project_grd_to_map(
            grd_feat_list[2], shift_u, shift_v, gt_heading, left_camera_k, sat2.shape[-1], ori_grdH, ori_grdW,
        )

        grd2sat8_attn = self.CVattn(grd2sat8, grd8, u, mask)
        grd2sat4_attn = grd2sat4 + self.Dec4(grd2sat8_attn, grd2sat4)
        grd2sat2_attn = grd2sat2 + self.Dec2(grd2sat4_attn, grd2sat2)

        grd_feat_list = [grd2sat8_attn, grd2sat4_attn, grd2sat2_attn]

        corr_maps = []

        for level in range(len(sat_feat_list)):
            # meter_per_pixel = self.meters_per_pixel[level]

            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            meter_per_pixel = self.meter_per_pixel * np.power(2, 3 - level)
            pad = int(10 / meter_per_pixel)
            sat_feat = F.pad(sat_feat, (pad, pad, pad, pad), 'constant', 0)

            A = sat_feat.shape[-1]

            radius_pixel = int(np.ceil(200 * np.sqrt(2) / self.args.sat_ori_res * 512))
            crop_H = int(A - radius_pixel * 2 / np.power(2, 3 - level))
            crop_W = int(A - radius_pixel * 2 / np.power(2, 3 - level))

            g2s_feat = TF.center_crop(grd_feat, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            if self.args.use_uncertainty:
                denominator = torch.sum(denominator, dim=1) * TF.center_crop(sat_uncer_list[level], [corr.shape[1], corr.shape[2]])[:, 0]
            else:
                denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - (corr_W / 2+0.5)) #* meter_per_pixel  # / self.args.shift_range_lon
            pred_v = (max_index // corr_W -(corr_H / 2+0.5)) #* meter_per_pixel  # / self.args.shift_range_lat

        if mode == 'train':
            # return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
            return corr_maps[0], corr_maps[1], corr_maps[2]
        else:
            return pred_u * 2, pred_v * 2  # [B], [B]




