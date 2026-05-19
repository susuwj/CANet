import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import GCA


class ChannelAttention(nn.Module):
    def __init__(self, channel=8, reduction=2):
        super(ChannelAttention, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
                  nn.Conv2d(channel, channel//reduction, 1, bias=False),
                  nn.ReLU(),
                  nn.Conv2d(channel//reduction, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_result = self.se(max_result)
        avg_result = self.se(avg_result)
        output = self.sigmoid(max_result + avg_result)
        return output
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        output = torch.cat([max_result, avg_result], 1)
        output = self.conv(output)
        output = self.sigmoid(output)
        return output
    
class UNet_A(nn.Module):
    def __init__(self, base_channels=8):
        super(UNet_A, self).__init__()
        #base_channels = base_channels
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.conv0_0 = Conv2d(base_channels, base_channels * 2, 3, stride=1, padding=1)
        self.conv0_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)
        self.conv0_2 = Conv2d(base_channels, base_channels * 2, 1, stride=1, relu=False)

        self.conv1_0 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv1_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.conv2_0 = Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)
        self.conv4_1 = Conv2d(base_channels * 8, base_channels * 8, 3, stride=1, relu=False, padding=1)
        self.conv4_2 = Conv2d(base_channels * 4, base_channels * 8, 1, stride=2, relu=False)

        self.conv5_0 = Conv2d(base_channels * 8, base_channels * 8, 3, stride=1, padding=1)
        self.conv5_1 = Conv2d(base_channels * 8, base_channels * 8, 3, stride=1, relu=False, padding=1)

        self.conv6_0 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 3, 2, 1, 1, bias=False)
        self.conv6_1 = nn.Conv2d(base_channels * 8, base_channels * 4, 3, 1, 1, bias=False)

        self.conv7_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)
        self.conv7_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)

        self.final_conv_1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.final_conv_3 = nn.Conv2d(16, 16, 3, 1, 1, bias=False)
        
        self.conv = Conv2d(64+32+16, 64, 1, stride=1, relu=False)
        self.channel_att =  ChannelAttention(64)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        x = self.init_conv(x)

        residual = x
        x = self.conv0_1(self.conv0_0(x))
        x += self.conv0_2(residual)
        x = nn.ReLU(inplace=True)(x)

        residual = x
        x = self.conv1_1(self.conv1_0(x))
        x += residual
        out3 = nn.ReLU(inplace=True)(x)


        residual = out3
        x = self.conv2_1(self.conv2_0(out3))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        residual = out2
        x = self.conv4_1(self.conv4_0(out2))
        x += self.conv4_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv5_1(self.conv5_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        out21 = F.interpolate(out2, scale_factor=0.5, mode='bilinear', align_corners=True).squeeze(1)
        out31 = F.interpolate(out3, scale_factor=0.25, mode='bilinear', align_corners=True).squeeze(1)

        out1 = self.conv(torch.cat([out1, out21, out31], dim=1))
        att = out1 * self.channel_att(out1)
        out1 = att * self.spatial_att(att) + out1
        
        x = self.conv6_0(out1)
        x = torch.cat([x, out2], 1)
        out2 = self.conv6_1(x)

        x = self.conv7_0(out2)
        x = torch.cat([x, out3], 1)
        out3 = self.conv7_1(x)
        
        outputs = {}
        outputs["stage1"] = self.final_conv_1(out1)
        outputs["stage2"] = self.final_conv_2(out2)
        outputs["stage3"] = self.final_conv_3(out3)

        return outputs

class FeatExt3_ref(nn.Module):
    def __init__(self):
        super(FeatExt3_ref, self).__init__()
        base_channels = 8
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.conv0_0 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, padding=1)
        self.conv0_1 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, relu=False, padding=1)
        self.conv0_2 = Conv2d(base_channels * 1, base_channels * 1, 1, stride=1, relu=False)

        self.conv1_0 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, padding=1)
        self.conv1_1 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, relu=False, padding=1)

        self.conv2_0 = Conv2d(base_channels * 1, base_channels * 2, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(base_channels * 1, base_channels * 2, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv4_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv5_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv5_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv6_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)
        self.conv6_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        #self.conv6_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        #self.conv6_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        #self.conv7_0 = nn.ConvTranspose2d(base_channels * 2, base_channels * 1, 3, 2, 1, 1, bias=False)
        #self.conv7_1 = nn.Conv2d(base_channels * 2, base_channels * 1, 3, 1, 1, bias=False)
        #self.conv7_2 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, padding=1)
        #self.conv7_3 = Conv2d(base_channels * 1, base_channels * 1, 3, stride=1, relu=False, padding=1)

        self.final_conv_1 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(16, 16, 3, 1, 1, bias=False)
        #self.final_conv_3 = nn.Conv2d(8, 8, 3, 1, 1, bias=False)

        self.conv = Conv2d(32+16, 32, 1, stride=1, relu=False)
        self.channel_att =  ChannelAttention(32)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.init_conv(x)

        residual = x
        x = self.conv0_1(self.conv0_0(x))
        x += self.conv0_2(residual)
        x = nn.ReLU(inplace=True)(x)

        residual = x
        x = self.conv1_1(self.conv1_0(x))
        x += residual
        out3 = nn.ReLU(inplace=True)(x)

        residual = out3
        x = self.conv2_1(self.conv2_0(out3))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        residual = out2
        x = self.conv4_1(self.conv4_0(out2))
        x += self.conv4_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv5_1(self.conv5_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)


        out21 = F.interpolate(out2, scale_factor=0.5, mode='bilinear', align_corners=True).squeeze(1)

        out1 = self.conv(torch.cat([out1, out21], dim=1))
        att = out1 * self.channel_att(out1)
        out1 = att * self.spatial_att(att) + out1

        x = self.conv6_0(out1)
        x = torch.cat([x, out2], 1)
        out2 = self.conv6_1(x)

        #x = self.conv7_0(out2)
        #x = torch.cat([x, out3], 1)
        #x = self.conv7_1(x)
        #residual = x
        #x = self.conv7_3(self.conv7_2(x))
        #x += residual
        #out3 = nn.ReLU(inplace=True)(x)

        outputs = {}
        outputs["stage1"] = self.final_conv_1(out1)
        outputs["stage2"] = self.final_conv_2(out2)
        #outputs["stage3"] = self.final_conv_3(out3)

        return outputs

class Refinement(nn.Module):
    def __init__(self, feat_channels):
        super(Refinement, self).__init__()
        base_channels = 8

        self.conv1_0 = nn.Sequential(
            Conv2d(1, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1))

        self.conv1_2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 3, 2, 1, 1, bias=False)
        #self.conv1_3 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        self.conv2_0 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv2_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)
        self.conv2_2 = Conv2d(feat_channels+base_channels * 2, base_channels * 4, 1, stride=2, relu=False)

        self.conv3_0 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1)
        self.conv3_1 = Conv2d(base_channels * 4, base_channels * 4, 3, stride=1, relu=False, padding=1)

        self.conv4_0 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1, bias=False)
        self.conv4_1 = nn.Conv2d(feat_channels+base_channels * 4, base_channels * 2, 3, 1, 1, bias=False)
        self.conv4_2 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1)
        self.conv4_3 = Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, relu=False, padding=1)

        self.final_conv = nn.Conv2d(base_channels * 2, 1, 3, padding=1, bias=False)
        
    def forward(self, depth, img_feat):
        depth_mean = torch.mean(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_std = torch.std(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth = (depth.unsqueeze(1) - depth_mean.unsqueeze(-1).unsqueeze(-1)) / depth_std.unsqueeze(-1).unsqueeze(-1)
        depth_min, _ = torch.min(depth.reshape(depth.shape[0],-1), -1, keepdim=True)
        depth_max,_ = torch.max(depth.reshape(depth.shape[0],-1), -1, keepdim=True)

        depth_feat = self.conv1_2(self.conv1_0(depth))  # 16,h/4,w/4
        #print(img_feat.shape, depth_feat.shape)
        cat = torch.cat((img_feat, depth_feat), dim=1) # 48,h/4,w/4
        #cat = self.conv1_3(cat)

        residual = cat
        x = self.conv2_1(self.conv2_0(cat))
        x += self.conv2_2(residual)
        x = nn.ReLU(inplace=True)(x)
        residual = x
        x = self.conv3_1(self.conv3_0(x))
        x += residual
        out1 = nn.ReLU(inplace=True)(x)

        x = self.conv4_0(out1)
        x = torch.cat([x, cat], 1)
        x = self.conv4_1(x)
        residual = x
        x = self.conv4_3(self.conv4_2(x))
        x += residual
        out2 = nn.ReLU(inplace=True)(x)

        res = self.final_conv(out2)

        res_ = torch.zeros_like(res)
        for i in range(res.shape[0]):
            res_[i] = torch.clamp(res[i], min=depth_min[i].cpu().item(), max=depth_max[i].cpu().item())
        #res_ = torch.clamp(res, min=depth_min.cpu().item(), max=depth_max.cpu().item())
        #print(res.dtype)
        depth = (res_ + F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=False)) * depth_std.unsqueeze(-1).unsqueeze(-1) + depth_mean.unsqueeze(-1).unsqueeze(-1)
        
        return res_.squeeze(1), depth.squeeze(1)

class Reg2d_GCA(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, feat_dim=64, qkv_dim=4, heads=4):
        super(Reg2d_GCA, self).__init__()
        
        self.conv0 = ConvBnReLU3D(input_channel*2, base_channel, kernel_size=(1,3,3), pad=(0,1,1))
        self.conv1 = ConvBnReLU3D(base_channel, base_channel*2, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv2 = ConvBnReLU3D(base_channel*2, base_channel*2)

        self.conv3 = ConvBnReLU3D(base_channel*2, base_channel*4, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv4 = ConvBnReLU3D(base_channel*4, base_channel*4)

        self.conv5 = ConvBnReLU3D(base_channel*4, base_channel*8, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv6 = ConvBnReLU3D(base_channel*8, base_channel*8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*8, base_channel*4, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel*4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*4, base_channel*2, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)
        self.gca = GCA(feat_dim=feat_dim, cost_dim=input_channel, qkv_dim=qkv_dim, heads=heads)

    def forward(self, x, global_feat):

        x_att = self.gca(x, global_feat)

        conv0 = self.conv0(torch.cat([x, x_att], dim=1))
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.prob(x)

        return x.squeeze(1), x_att.mean(1)

class RegVis(nn.Module):

    def __init__(self):
        super(RegVis, self).__init__()
        base_channels = 8

        # self.init_conv = lambda x: x

        self.conv = ConvBnReLU3D(base_channels, 1, kernel_size=3, stride=1, pad=1)
        self.final_conv = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x = self.conv(x)
        x = self.final_conv(x)

        return x

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    C = src_fea.shape[1]
    Hs,Ws = src_fea.shape[-2:]
    B,num_depth,Hr,Wr = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, Hr, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, Wr, dtype=torch.float32, device=src_fea.device)])
        y = y.reshape(Hr*Wr)
        x = x.reshape(Hr*Wr)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.reshape(B, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # FIXME divide 0
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp==0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]
        # proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((Ws - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hs - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    if len(src_fea.shape)==4:
        warped_src_fea = F.grid_sample(src_fea, grid.reshape(B, num_depth * Hr, Wr, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_src_fea = warped_src_fea.reshape(B, C, num_depth, Hr, Wr)
    elif len(src_fea.shape)==5:
        warped_src_fea = []
        for d in range(src_fea.shape[2]):
            warped_src_fea.append(F.grid_sample(src_fea[:,:,d], grid.reshape(B, num_depth, Hr, Wr, 2)[:,d], mode='bilinear', padding_mode='zeros', align_corners=True))
        warped_src_fea = torch.stack(warped_src_fea, dim=2)

    return warped_src_fea


def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]
    itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H, W)  / (ndepths - 1)  # 1 D H W
    inverse_depth_hypo = inverse_depth_max[:,None, None, None] + (inverse_depth_min - inverse_depth_max)[:,None, None, None] * itv

    return 1./inverse_depth_hypo


def schedule_inverse_range(inverse_min_depth, inverse_max_depth, ndepths, H, W):
    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H, W)  / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:,None, :, :] + (inverse_min_depth - inverse_max_depth)[:,None, :, :] * itv  # B D H W
    return 1./inverse_depth_hypo


# --------------------------------------------------------------


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn_momentum=0.1, init_method="xavier", gn=False, group_channel=8, **kwargs):
        super(Conv2d, self).__init__()
        bn = not gn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        else:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)
