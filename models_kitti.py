import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import utils
import torchvision.transforms.functional as TF

from VGG import VGGUnet, Encoder, Decoder, Decoder2, Decoder4
from jacobian import grid_sample

from models_ford import loss_func
from RNNs import Uncertainty
from swin_transformer import TransOptimizerS2GP_V1, TransOptimizerG2SP_V1
from cross_attention import CrossViewAttention

EPS = utils.EPS

class Model(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(Model, self).__init__()

        self.args = args

        self.level = args.level
        self.N_iters = args.N_iters

        self.SatFeatureNet = VGGUnet(self.level)

        if self.args.proj == 'CrossAttn':
            self.GrdEnc = Encoder()
            self.GrdDec = Decoder()
            self.Dec4 = Decoder4()
            self.Dec2 = Decoder2()
            self.CVattn = CrossViewAttention(blocks=2, dim=256, heads=4, dim_head=16, qkv_bias=False)
        else:
            self.GrdFeatureNet = VGGUnet(self.level)

        self.meters_per_pixel = []
        meter_per_pixel = utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))

        if self.args.Optimizer == 'TransV1G2SP':
            self.TransRefine = TransOptimizerG2SP_V1()
        elif self.args.Optimizer == 'TransV1S2GP':
            self.TransRefine = TransOptimizerS2GP_V1()
            ori_grdH, ori_grdW = 256, 1024
            xyz_grds = []
            for level in range(4):
                grd_H, grd_W = ori_grdH / (2 ** (3 - level)), ori_grdW / (2 ** (3 - level))

                xyz_grd, mask, xyz_w = self.grd_img2cam(grd_H, grd_W, ori_grdH,
                                                        ori_grdW)  # [1, grd_H, grd_W, 3] under the grd camera coordinates
                xyz_grds.append((xyz_grd, mask, xyz_w))

            self.xyz_grds = xyz_grds

        if self.args.rotation_range > 0:
            self.coe_R = nn.Parameter(torch.tensor(-5., dtype=torch.float32), requires_grad=True)
            self.coe_T = nn.Parameter(torch.tensor(-3., dtype=torch.float32), requires_grad=True)

        if self.args.use_uncertainty:
            self.uncertain_net = Uncertainty()

        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.

    def grd_img2cam(self, grd_H, grd_W, ori_grdH, ori_grdW):

        ori_camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
                                      [0.0000, 482.7076, 125.0034],
                                      [0.0000, 0.0000, 1.0000]]],
                                    dtype=torch.float32, requires_grad=True)  # [1, 3, 3]

        camera_height = utils.get_camera_height()

        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

        v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                              torch.arange(0, grd_W, dtype=torch.float32))
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0)  # [1, grd_H, grd_W, 3]
        xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]

        w = camera_height / torch.where(torch.abs(xyz_w[..., 1:2]) > utils.EPS, xyz_w[..., 1:2],
                                        utils.EPS * torch.ones_like(xyz_w[..., 1:2]))  # [BN, grd_H, grd_W, 1]
        xyz_grd = xyz_w * w  # [1, grd_H, grd_W, 3] under the grd camera coordinates

        mask = (xyz_grd[..., -1] > 0).float()  # # [1, grd_H, grd_W]

        return xyz_grd, mask, xyz_w

    def grd2cam2world2sat(self, ori_shift_u, ori_shift_v, ori_heading, level, satmap_sidelength,):
        '''
        realword: X: south, Y:down, Z: east
        camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        Args:
            ori_shift_u: [B, 1]
            ori_shift_v: [B, 1]
            heading: [B, 1]
            XYZ_1: [H,W,4]
            ori_camera_k: [B,3,3]
            grd_H:
            grd_W:
            ori_grdH:
            ori_grdW:

        Returns:
        '''
        B, _ = ori_heading.shape
        heading = ori_heading * self.args.rotation_range / 180 * np.pi
        shift_u = ori_shift_u * self.args.shift_range_lon
        shift_v = ori_shift_v * self.args.shift_range_lat

        cos = torch.cos(heading)
        sin = torch.sin(heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B, 9]
        R = R.view(B, 3, 3)  # shape = [B, N, 3, 3]
        # this R is the inverse of the R in G2SP

        camera_height = utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u[:, :1])
        T0 = torch.cat([shift_v, height, -shift_u], dim=-1)  # shape = [B, 3]
        T = torch.sum(-R * T0[:, None, :], dim=-1)  # [B, 3]
        # The above R, T define transformation from camera to world

        xyz_grd = self.xyz_grds[level][0].detach().to(ori_shift_u.device).repeat(B, 1, 1, 1)
        mask = self.xyz_grds[level][1].detach().to(ori_shift_u.device).repeat(B, 1, 1)  # [B, grd_H, grd_W]
        grd_H, grd_W = xyz_grd.shape[1:3]

        xyz = torch.sum(R[:, None, None, :, :] * xyz_grd[:, :, :, None, :], dim=-1) + T[:, None, None, :]
        # [B, grd_H, grd_W, 3]

        R_sat = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.float32, device=ori_shift_u.device, requires_grad=True) \
            .reshape(2, 3)
        zx = torch.sum(R_sat[None, None, None, :, :] * xyz[:, :, :, None, :], dim=-1)
        # [B, grd_H, grd_W, 2]

        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        sat_uv = zx / meter_per_pixel + satmap_sidelength / 2  # [B, grd_H, grd_W, 2] sat map uv

        return sat_uv, mask

    def project_map_to_grd(self, sat_f, sat_c, shift_u, shift_v, heading, level):
        '''
        Args:
            sat_f: [B, C, H, W]
            sat_c: [B, 1, H, W]
            shift_u: [B, 2]
            shift_v: [B, 2]
            heading: [B, 1]
            camera_k: [B, 3, 3]

            ori_grdH:
            ori_grdW:

        Returns:

        '''
        B, C, satmap_sidelength, _ = sat_f.size()

        uv, mask = self.grd2cam2world2sat(shift_u, shift_v, heading, level, satmap_sidelength)
        # [B, H, W, 2], [B, H, W], [B, H, W, 2], [B, H, W, 2], [B,H, W, 2]

        B, grd_H, grd_W, _ = uv.shape

        sat_f_trans, _ = grid_sample(sat_f, uv, jac=None)
        sat_f_trans = sat_f_trans * mask[:, None, :, :]

        if sat_c is not None:
            sat_c_trans, _ = grid_sample(sat_c, uv)
            sat_c_trans = sat_c_trans * mask[:, None, :, :]
        else:
            sat_c_trans = None

        return sat_f_trans, sat_c_trans, uv * mask[:, :, :, None], mask

    def sat2world(self, satmap_sidelength):
        # satellite: u:east , v:south from bottomleft and u_center: east; v_center: north from center
        # realword: X: south, Y:down, Z: east   origin is set to the ground plane

        # meshgrid the sat pannel
        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        # uv is coordinate from top/left, v: south, u:east
        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # sat map from top/left to center coordinate
        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor(
            [u0, v0]).cuda()  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]

        # affine matrix: scale*R
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        Y = torch.zeros_like(XZ[..., 0:1])
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]

        return sat2realwap

    def World2GrdImgPixCoordinates(self, ori_shift_u, ori_shift_v, ori_heading, XYZ_1, ori_camera_k, grd_H, grd_W,
                                   ori_grdH, ori_grdW):
        # realword: X: south, Y:down, Z: east
        # camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        # XYZ_1:[H,W,4], heading:[B,1], camera_k:[B,3,3], shift:[B,2]
        B = ori_heading.shape[0]
        shift_u_meters = self.args.shift_range_lon * ori_shift_u
        shift_v_meters = self.args.shift_range_lat * ori_shift_v
        heading = ori_heading * self.args.rotation_range / 180 * np.pi

        cos = torch.cos(-heading)
        sin = torch.sin(-heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,9]
        R = R.view(B, 3, 3)  # shape = [B,3,3]

        camera_height = utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u_meters)
        T = torch.cat([shift_v_meters, height, -shift_u_meters], dim=-1)  # shape = [B, 3]
        T = torch.unsqueeze(T, dim=-1)  # shape = [B,3,1]

        # P = K[R|T]
        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        P = camera_k @ torch.cat([R, T], dim=-1)

        uv1 = torch.sum(P[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
        # only need view in front of camera ,Epsilon = 1e-6
        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]

        mask = torch.greater(uv1_last, torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)

        return uv, mask

    def project_grd_to_map(self, grd_f, grd_c, shift_u, shift_v, heading, camera_k, satmap_sidelength, ori_grdH,
                           ori_grdW):
        '''
        grd_f: [B, C, H, W]
        grd_c: [B, 1, H, W]
        shift_u: [B, 1]
        shift_v: [B, 1]
        heading: [B, 1]
        camera_k: [B, 3, 3]
        satmap_sidelength: scalar
        ori_grdH: scalar
        ori_grdW: scalar
        '''

        B, C, H, W = grd_f.size()

        XYZ_1 = self.sat2world(satmap_sidelength)  # [ sidelength,sidelength,4]
        uv, mask = self.World2GrdImgPixCoordinates(shift_u, shift_v, heading, XYZ_1, camera_k,
                                                   H, W, ori_grdH, ori_grdW)  # [B, S, E, H, W,2]
        # [B, H, W, 2], [2, B, H, W, 2], [1, B, H, W, 2]

        grd_f_trans, _ = grid_sample(grd_f, uv, jac=None)
        # [B,C,sidelength,sidelength], [3, B, C, sidelength, sidelength]
        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)
        else:
            grd_c_trans = None

        return grd_f_trans, grd_c_trans, uv[..., 0], mask

    def Trans_update(self, shift_u, shift_v, heading, grd_feat_proj, sat_feat):
        B = shift_u.shape[0]
        grd_feat_norm = torch.norm(grd_feat_proj.reshape(B, -1), p=2, dim=-1)
        grd_feat_norm = torch.maximum(grd_feat_norm, 1e-6 * torch.ones_like(grd_feat_norm))
        grd_feat_proj = grd_feat_proj / grd_feat_norm[:, None, None, None]

        delta = self.TransRefine(grd_feat_proj, sat_feat)  # [B, 3]

        shift_u_new = shift_u + delta[:, 0:1]
        shift_v_new = shift_v + delta[:, 1:2]
        heading_new = heading + delta[:, 2:3]

        B = shift_u.shape[0]

        rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_u.requires_grad = True
        rand_v.requires_grad = True
        shift_u_new = torch.where((shift_u_new > -2) & (shift_u_new < 2), shift_u_new, rand_u)
        shift_v_new = torch.where((shift_v_new > -2) & (shift_v_new < 2), shift_v_new, rand_v)

        return shift_u_new, shift_v_new, heading_new

    def corr(self, sat_map, grd_img_left, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
             mode='train'):
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

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        corr_maps = []

        for level in range(len(sat_feat_list)):
            meter_per_pixel = self.meters_per_pixel[level]

            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            A = sat_feat.shape[-1]
            heading = gt_heading + np.random.uniform(- self.args.coe_heading_aug, self.args.coe_heading_aug)
            grd_feat_proj, _, u, mask = self.project_grd_to_map(
                grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

            crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
            crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
            g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
            pred_v = -(max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

            cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
            sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)

            pred_u1 = pred_u * cos + pred_v * sin
            pred_v1 = - pred_u * sin + pred_v * cos


        if mode == 'train':
            return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
        else:
            return pred_u1, pred_v1  # [B], [B]

    def triplet_loss(self, corr_maps, gt_shift_u, gt_shift_v, gt_heading):
        cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
        sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)

        gt_delta_x = - gt_shift_u[:, 0] * self.args.shift_range_lon
        gt_delta_y = - gt_shift_v[:, 0] * self.args.shift_range_lat

        gt_delta_x_rot = - gt_delta_x * cos + gt_delta_y * sin
        gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos

        losses = []
        for level in range(len(corr_maps)):
            meter_per_pixel = self.meters_per_pixel[level]

            corr = corr_maps[level]
            B, corr_H, corr_W = corr.shape

            w = torch.round(corr_W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel)
            h = torch.round(corr_H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel)

            pos = corr[range(B), h.long(), w.long()]  # [B]
            pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))

    def CVattn_corr(self, sat_map, grd_img_left, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                    mode='train'):
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

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = gt_heading

        grd2sat8, _, u, mask = self.project_grd_to_map(
            grd_feat_list[0], None, shift_u, shift_v, heading, left_camera_k, sat8.shape[-1], ori_grdH, ori_grdW)
        grd2sat4, _, _, _ = self.project_grd_to_map(
            grd_feat_list[1], None, shift_u, shift_v, heading, left_camera_k, sat4.shape[-1], ori_grdH, ori_grdW)
        grd2sat2, _, _, _ = self.project_grd_to_map(
            grd_feat_list[2], None, shift_u, shift_v, heading, left_camera_k, sat2.shape[-1], ori_grdH, ori_grdW)

        grd2sat8_attn = self.CVattn(grd2sat8, grd8, u, mask)
        grd2sat4_attn = grd2sat4 + self.Dec4(grd2sat8_attn, grd2sat4)
        grd2sat2_attn = grd2sat2 + self.Dec2(grd2sat4_attn, grd2sat2)

        grd_feat_list = [grd2sat8_attn, grd2sat4_attn, grd2sat2_attn]

        corr_maps = []

        for level in range(len(sat_feat_list)):
            meter_per_pixel = self.meters_per_pixel[level]

            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            A = sat_feat.shape[-1]

            crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
            crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
            g2s_feat = TF.center_crop(grd_feat, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            # denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            if self.args.use_uncertainty:
                denominator = torch.sum(denominator, dim=1) * TF.center_crop(sat_uncer_list[level],
                                                                             [corr.shape[1], corr.shape[2]])[:, 0]
            else:
                denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
            pred_v = -(max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

            cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
            sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)

            pred_u1 = pred_u * cos + pred_v * sin
            pred_v1 = - pred_u * sin + pred_v * cos

        if mode == 'train':
            return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
        else:
            return pred_u1, pred_v1  # [B], [B]

    def rot_corr(self, sat_map, grd_img_left, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                 mode='train'):
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
        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        pred_feat_dict = {}
        shift_us_all = []
        shift_vs_all = []
        headings_all = []
        for iter in range(self.N_iters):
            shift_us = []
            shift_vs = []
            headings = []
            for level in range(len(sat_feat_list)):
                sat_feat = sat_feat_list[level]
                sat_conf = sat_conf_list[level]
                grd_feat = grd_feat_list[level]
                grd_conf = grd_conf_list[level]

                A = sat_feat.shape[-1]
                grd_feat_proj, _, u, mask = self.project_grd_to_map(
                    grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

                shift_u_new, shift_v_new, heading_new = self.Trans_update(
                    shift_u, shift_v, heading, grd_feat_proj, sat_feat)

                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                headings.append(heading_new[:, 0])

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                heading = heading_new.clone()

                if level not in pred_feat_dict.keys():
                    pred_feat_dict[level] = [grd_feat_proj]
                else:
                    pred_feat_dict[level].append(grd_feat_proj)

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
            headings_all.append(torch.stack(headings, dim=1))  # [B, Level]

        shift_lats = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
        thetas = torch.stack(headings_all, dim=1)  # [B, N_iters, Level]

        def corr(sat_feat_list, grd_feat_list, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                 pred_heading=None, mode='train'):
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

            shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
            shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
            # heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

            corr_maps = []

            for level in range(len(sat_feat_list)):
                meter_per_pixel = self.meters_per_pixel[level]

                sat_feat = sat_feat_list[level]
                grd_feat = grd_feat_list[level]

                A = sat_feat.shape[-1]
                if mode == 'train':
                    # if epoch == 0:
                    #     heading = gt_heading + np.random.uniform(-0.1, 0.1)
                    # else:
                    #     heading = gt_heading + np.random.uniform(-0.05, 0.05)

                    heading = gt_heading
                else:
                    heading = pred_heading
                grd_feat_proj, _, _, mask = self.project_grd_to_map(
                    grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

                crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
                crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
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
                pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
                pred_v = -(max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

                cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
                sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)

                pred_u1 = pred_u * cos + pred_v * sin
                pred_v1 = - pred_u * sin + pred_v * cos

            if mode == 'train':
                return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
            else:
                return pred_u1, pred_v1  # [B], [B]

        if mode == 'train':

            loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
            shift_lat_last, shift_lon_last, theta_last, \
                = loss_func(shift_lats, shift_lons, thetas, gt_shift_v[:, 0], gt_shift_u[:, 0], gt_heading[:, 0],
                            torch.exp(-self.coe_R), torch.exp(-self.coe_R), torch.exp(-self.coe_R))

            trans_loss = corr(sat_feat_list, grd_feat_list, left_camera_k, gt_shift_u, gt_shift_v, gt_heading,
                              thetas[:, -1, -1:], mode)

            return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                   shift_lat_last, shift_lon_last, theta_last, \
                   grd_conf_list, trans_loss
        else:
            pred_u, pred_v = corr(sat_feat_list, grd_feat_list, left_camera_k, gt_shift_u, gt_shift_v, gt_heading,
                                  thetas[:, -1, -1:], mode)
            pred_orien = thetas[:, -1, -1]

            return pred_u, pred_v, pred_orien * self.args.rotation_range

    def CVattn_rot_corr(self, sat_map, grd_img_left, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                        mode='train'):
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

        # Generate multiscale grd2sat features
        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        shift_us_all = []
        shift_vs_all = []
        headings_all = []
        for iter in range(self.N_iters):
            shift_us = []
            shift_vs = []
            headings = []
            for level in range(len(sat_feat_list)):
                sat_feat = sat_feat_list[level]
                grd_feat = grd_feat_list[level]

                if self.args.Optimizer == 'TransV1G2SP':
                    A = sat_feat.shape[-1]
                    grd_feat_proj, _, _, _ = self.project_grd_to_map(
                        grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

                    shift_u_new, shift_v_new, heading_new = self.Trans_update(
                        shift_u, shift_v, heading, grd_feat_proj, sat_feat)
                elif self.args.Optimizer == 'TransV1S2GP':
                    grd_H, grd_W = grd_feat.shape[-2:]
                    sat_feat_proj, _, sat_uv, mask = self.project_map_to_grd(
                        sat_feat, None, shift_u, shift_v, heading, level)
                    # [B, C, H, W], [B, 1, H, W], [3, B, C, H, W], [B, H, W, 2]

                    grd_feat = grd_feat * mask[:, None, :, :]

                    shift_u_new, shift_v_new, heading_new = self.Trans_update(shift_u, shift_v, heading,
                                                                               sat_feat_proj[:, :, grd_H // 2:, :],
                                                                               grd_feat[:, :, grd_H // 2:, :],
                                                                               )  # only need to compare bottom half

                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                headings.append(heading_new[:, 0])

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                heading = heading_new.clone()

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
            headings_all.append(torch.stack(headings, dim=1))  # [B, Level]

        shift_lats = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
        thetas = torch.stack(headings_all, dim=1)  # [B, N_iters, Level]


        def corr(sat_feat_list, grd_feat_list, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                 pred_heading=None, mode='train'):
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

            corr_maps = []

            for level in range(len(sat_feat_list)):
                meter_per_pixel = self.meters_per_pixel[level]

                sat_feat = sat_feat_list[level]
                grd_feat = grd_feat_list[level]

                A = sat_feat.shape[-1]

                # crop_grd = int(np.ceil(20/meter_per_pixel))
                # crop_sat = crop_grd + int(np.ceil(self.args.shift_range_lat * 3 / meter_per_pixel))

                # crop_H = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
                # crop_W = int(A - self.args.shift_range_lon * 3 / meter_per_pixel)
                crop_grd = int(A - self.args.shift_range_lat * 3 / meter_per_pixel)
                crop_sat = sat_feat.shape[-1]

                g2s_feat = TF.center_crop(grd_feat, [crop_grd, crop_grd])
                g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_grd, crop_grd)

                # s_feat = sat_feat.reshape(1, -1, A, A)  # [B, C, H, W]->[1, B*C, H, W]
                s_feat = TF.center_crop(sat_feat, [crop_sat, crop_sat]).reshape(1, -1, crop_sat, crop_sat)
                corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  # [B, H, W]

                denominator = F.avg_pool2d(sat_feat.pow(2), (crop_grd, crop_grd), stride=1, divisor_override=1)  # [B, 4W]
                if self.args.use_uncertainty:
                    denominator = torch.sum(denominator, dim=1) * TF.center_crop(sat_uncer_list[level], [corr.shape[1], corr.shape[2]])[:, 0]
                else:
                    denominator = torch.sum(denominator, dim=1)  # [B, H, W]
                # denominator = torch.sum(denominator, dim=1)  # [B, H, W]
                denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
                corr = 2 - 2 * corr / denominator

                B, corr_H, corr_W = corr.shape

                corr_maps.append(corr)

                max_index = torch.argmin(corr.reshape(B, -1), dim=1)
                pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel  # / self.args.shift_range_lon
                pred_v = -(max_index // corr_W - corr_H / 2) * meter_per_pixel  # / self.args.shift_range_lat

                cos = torch.cos(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)
                sin = torch.sin(gt_heading[:, 0] * self.args.rotation_range / 180 * np.pi)

                pred_u1 = pred_u * cos + pred_v * sin
                pred_v1 = - pred_u * sin + pred_v * cos

            if mode == 'train':
                return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v, gt_heading)
            else:
                return pred_u1, pred_v1  # [B], [B]

        if mode == 'train':
            # Rotation Loss
            loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
            shift_lat_last, shift_lon_last, theta_last, \
                = loss_func(shift_lats, shift_lons, thetas, gt_shift_v[:, 0], gt_shift_u[:, 0], gt_heading[:, 0],
                            torch.exp(-self.coe_R), torch.exp(-self.coe_R), torch.exp(-self.coe_R))

            # Translation Loss
            shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
            shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
            heading = gt_heading

            grd2sat8, _, u, mask = self.project_grd_to_map(
                grd_feat_list[0], None, shift_u, shift_v, heading, left_camera_k, sat8.shape[-1], ori_grdH, ori_grdW)
            grd2sat4, _, _, _ = self.project_grd_to_map(
                grd_feat_list[1], None, shift_u, shift_v, heading, left_camera_k, sat4.shape[-1], ori_grdH, ori_grdW)
            grd2sat2, _, _, _ = self.project_grd_to_map(
                grd_feat_list[2], None, shift_u, shift_v, heading, left_camera_k, sat2.shape[-1], ori_grdH, ori_grdW)

            grd2sat8_attn = self.CVattn(grd2sat8, grd8, u, mask)
            grd2sat4_attn = grd2sat4 + self.Dec4(grd2sat8_attn, grd2sat4)
            grd2sat2_attn = grd2sat2 + self.Dec2(grd2sat4_attn, grd2sat2)

            grd_feat_list = [grd2sat8_attn, grd2sat4_attn, grd2sat2_attn]

            trans_loss = corr(sat_feat_list, grd_feat_list, left_camera_k, gt_shift_u, gt_shift_v, gt_heading,
                              thetas[:, -1, -1:], mode)

            return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                   shift_lat_last, shift_lon_last, theta_last, \
                   trans_loss

        else:
            pred_orien = thetas[:, -1, -1:]

            shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
            shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
            # Translation
            grd2sat8, _, u, mask = self.project_grd_to_map(
                grd_feat_list[0], None, shift_u, shift_v, pred_orien, left_camera_k, sat8.shape[-1], ori_grdH, ori_grdW)
            grd2sat4, _, _, _ = self.project_grd_to_map(
                grd_feat_list[1], None, shift_u, shift_v, pred_orien, left_camera_k, sat4.shape[-1], ori_grdH, ori_grdW)
            grd2sat2, _, _, _ = self.project_grd_to_map(
                grd_feat_list[2], None, shift_u, shift_v, pred_orien, left_camera_k, sat2.shape[-1], ori_grdH, ori_grdW)

            grd2sat8_attn = self.CVattn(grd2sat8, grd8, u, mask)
            grd2sat4_attn = grd2sat4 + self.Dec4(grd2sat8_attn, grd2sat4)
            grd2sat2_attn = grd2sat2 + self.Dec2(grd2sat4_attn, grd2sat2)

            grd_feat_list = [grd2sat8_attn, grd2sat4_attn, grd2sat2_attn]
            pred_u, pred_v = corr(sat_feat_list, grd_feat_list, left_camera_k, gt_shift_u, gt_shift_v, gt_heading,
                                  pred_orien, mode)

            return pred_u, pred_v, pred_orien[:, 0] * self.args.rotation_range

