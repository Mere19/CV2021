from pandas.core.base import IndexOpsMixin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO
        self.layers = nn.Sequential(
            # The choice of dilation = 1 and bias = False is based on the paper https://arxiv.org/pdf/2012.01411.pdf
            # In my case, without this dilation and bias setting the loss does not converge very well
            # However, I know some students obtained converging loss even without dilation = 1 and bias = False
            # This might have something to do with the version of some libraries used
            nn.Conv2d(3, 8, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=(5, 5), stride=2, padding=2, dilation=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=2, padding=2, dilation=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1, dilation=1, bias=False)
        )

    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        # print("X SHAPE" + str(x.size()))
        x = self.layers(x.float())
        # print("X SHAPE AFTER: " + str(x.size()))

        return x


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO
        self.layers = [None for _ in range(6)]
        self.layers[0] = nn.Sequential(
            nn.Conv2d(G, 8, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layers[1] = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layers[2] = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layers[3] = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.layers[4] = nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.layers[5] = nn.Conv2d(8, 1, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        B, G, D, H, W = x.size()
        # print(x.size())
        x_reshape = x.permute(0,2,1,3,4).reshape(B*D,G,H,W)
        # x_reshape = torch.reshape(x, (B, G, D * H, W))
        if torch.cuda.is_available():
            for model in self.layers:
                model.cuba()

        C0 = self.layers[0](x_reshape.float())
        # print("C0 shape" + str(C0.size()))
        C1 = self.layers[1](C0)
        # print("C1 shape" + str(C1.size()))
        C2 = self.layers[2](C1)
        # print("C2 shape" + str(C2.size()))
        C3 = self.layers[3](C2)
        # print("C3 shape" + str(C3.size()))
        C4 = self.layers[4](C3 + C1)
        # print("C4 shape" + str(C4.size()))
        S_bar = self.layers[5](C4 + C0)
        # print(S_bar.size())

        S_bar = torch.reshape(S_bar, (B, D, H, W))

        return S_bar

def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        # print(rot)
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)

        # TODO
        # the following code is adapted from the code in https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/models/module.py

        # homogeneous coordinates of (x, y) = (x, y, 1)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz.double())  # [B, 3, H*W]

        # reshape depth values for multiplication with xyz
        depths = depth_values[:, :, None]
        depths = depths.repeat(1, 1, H*W) # [B, D, H*W]

        # 3D coordinate of x, y in the src camera frame
        # by multiplaying the normalized 2D homogeneous coordinates with the depth values
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, D, 1) * depths.view(B, 1, D, H * W)  # [B, 3, D, H*W]
        proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)  # [B, 3, D, H*W]

        # normalize the transformed 3D coordinates with its z coordinate
        # to obtain the corresponding 2D coordinate in the image
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, D, H*W]

        # normalize x, y for grid sample
        proj_x_normalized = proj_xy[:, 0, :, :] / ((W - 1) / 2) - 1  # [B, D, H*W]
        proj_y_normalized = proj_xy[:, 1, :, :] / ((H - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, D, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(
        src_fea,
        grid.view(B, D * H, W, 2).float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return warped_src_fea.view(B, C, D, H, W)

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    # print(warped_src_fea)
    B, C, D, H, W = warped_src_fea.size()
    # # print("warped src fea size: " + str(warped_src_fea.size()))
    # ref_fea_reshape = torch.reshape(ref_fea, (B, G, int(C/G), H, W))
    # ref_fea_reshape = ref_fea_reshape[:, :, :, None, :, :]
    # ref_fea_reshape = ref_fea_reshape.repeat((1, 1, 1, D, 1, 1))
    # # print("ref fea reshape shape: " + str(ref_fea_reshape.size()))
    # warped_src_fea_reshape = torch.reshape(warped_src_fea, (B, G, int(C/G), D, H, W))
    # # print("warped src fea reshape shape: " + str(warped_src_fea_reshape.shape))

    # # compute group wise correlation
    # element_wise_multiplication = torch.empty((B, G, int(C/G), D, H, W))
    # group_wise_similarity = torch.empty((B, G, D, H, W))
    # for b in range(B):
    #     # print(warped_src_fea_reshape[0, 0])
    #     # print(ref_fea_reshape[0, 0])
    #     element_wise_multiplication[b] = warped_src_fea_reshape[b] * ref_fea_reshape[b]
    #     # print(element_wise_multiplication[0, 0])
    # group_wise_similarity = torch.sum(element_wise_multiplication, dim=2) / (C / G)
    group_wise_similarity = (ref_fea.view(B, G, C // G, 1, H, W) * warped_src_fea.view(B, G, C // G, D, H, W)).mean(2)
    # print("group wise similarity size: " + str(group_wise_similarity.size()))

    return group_wise_similarity

def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    # B, D, H, W = p.size()
    # print(depth_values.size())
    # depth_values_reshape = depth_values[:, :, None, None]
    # depth_values_repeat = depth_values_reshape.repeat((1, 1, H, W))
    # print(depth_values_repeat)

    # regressed_depths = torch.empty((B, D, H, W))
    # for b in range(B):
    #     regressed_depths[b] = p[b] * depth_values_repeat[b]
    # regressed_depths = torch.sum(regressed_depths, dim=1) / torch.sum(p, dim=1)
    # print(regressed_depths)

    # print("regressed depth shape: " + str(regressed_depths.size()))

    # regressed_depths = torch.sum(p * depth_values_reshape, dim=1)
    # print("regressed depths size: " + str(regressed_depths.size()))

    # return regressed_depths.reshape(B, D, H, W)
    return torch.sum(p * depth_values.view(depth_values.shape[0], depth_values.shape[1], 1, 1), dim=1)

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    # l1_loss = torch.sum(torch.abs(depth_est - depth_gt) * mask)
    depth_est_masked = depth_est * mask
    depth_gt_masked = depth_gt * mask
    l1_loss = F.l1_loss(depth_est_masked, depth_gt_masked)

    return l1_loss