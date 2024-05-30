import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class DualLLama(nn.Module):
    def __init__(self, in_channel=16, proj_chan=2, mlp_mul=2, patch=(10, 10, 6),
                 llama_model=None, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,
                 bias_opt=True):
        super(DualLLama, self).__init__()

        self.llama_out_channel = 64 * in_channel
        self.patch = patch

        self.conv_m_1 = ConvBlock(in_channel, in_channel, True, bias_opt)  # full
        self.conv_f_1 = ConvBlock(in_channel, in_channel, True, bias_opt)

        self.d_conv_m_1 = DownConv(in_channel, bias_opt)  # 1/2
        self.d_conv_f_1 = DownConv(in_channel, bias_opt)
        self.conv_m_2 = ConvBlock(2 * in_channel, 2 * in_channel, False, bias_opt)
        self.conv_f_2 = ConvBlock(2 * in_channel, 2 * in_channel, False, bias_opt)

        self.d_conv_m_2 = DownConv(2 * in_channel, bias_opt)  # 1/4
        self.d_conv_f_2 = DownConv(2 * in_channel, bias_opt)
        self.conv_m_3 = ConvBlock(4 * in_channel, 4 * in_channel, False, bias_opt)
        self.conv_f_3 = ConvBlock(4 * in_channel, 4 * in_channel, False, bias_opt)

        self.d_conv_m_3 = DownConv(4 * in_channel, bias_opt)  # 1/8
        self.d_conv_f_3 = DownConv(4 * in_channel, bias_opt)
        self.conv_m_4 = ConvBlock(8 * in_channel, 8 * in_channel, False, bias_opt)
        self.conv_f_4 = ConvBlock(8 * in_channel, 8 * in_channel, False, bias_opt)

        self.d_conv_m_4 = DownConv(8 * in_channel, bias_opt)  # 1/16
        self.d_conv_f_4 = DownConv(8 * in_channel, bias_opt)

        self.llama_block = nn.Sequential(
            nn.Linear(2 * 16 * in_channel, mlp_mul * 2 * 16 * in_channel, bias_opt),
            nn.Linear(mlp_mul * 2 * 16 * in_channel, 4096, bias_opt),
            llama_model,

            nn.Linear(4096, mlp_mul * 2 * 16 * in_channel),
            nn.Linear(mlp_mul * 2 * 16 * in_channel, 4096),
            # nn.LayerNorm(4096),
            #
            # nn.Linear(4096, mlp_mul * 2 * 16 * in_channel, bias_opt),
            # nn.Linear(2 * 16 * mlp_mul * in_channel, 4096, bias_opt),
            llama_model,
            # nn.Linear(4096, mlp_mul * 2 * 16 * in_channel),
            # nn.Linear(2 * 16 * mlp_mul * in_channel, self.llama_out_channel),
            # nn.LayerNorm(self.llama_out_channel)
        )
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.patch[0] * self.patch[1] * self.patch[2], 2 * 16 * in_channel))
        self.pos_drop = nn.Dropout(0.)

        self.ada_1 = nn.Linear(4096, in_channel * 16, bias_opt)

        self.llama_up1 = nn.ConvTranspose3d(in_channel * 16, in_channel * 8, 2, 2,
                                            bias=bias_opt)
        self.llama_conv1 = ConvBlock(in_channel * 8 * 3, in_channel * 8, False, bias_opt)

        self.up2 = nn.ConvTranspose3d(in_channel * 8, in_channel * 4, 2, 2, bias=bias_opt)
        self.ada_2 = nn.Linear(4096, in_channel * 64, bias_opt)
        self.llama_up2 = nn.ConvTranspose3d(in_channel * 8, in_channel * 4, 2, 2,
                                            bias=bias_opt)
        self.llama_conv2 = ConvBlock(in_channel * 4 * 4, in_channel * 4, False, bias_opt)
        self.flow1 = nn.Sequential(nn.Conv3d(in_channel * 4, in_channel, 3, 1, 1, bias=bias_opt),
                                   nn.Conv3d(in_channel, 3, 3, 1, 1, bias=bias_opt))

        self.up3 = nn.ConvTranspose3d(in_channel * 4, in_channel * 2, 2, 2, bias=bias_opt)
        self.ada_3 = nn.Linear(4096, in_channel * 256, bias_opt)
        self.llama_up3 = nn.ConvTranspose3d(in_channel * 4, in_channel * 2, 2, 2,
                                            bias=bias_opt)
        self.llama_conv3 = ConvBlock(in_channel * 2 * 4 + 4, in_channel * 2, False, bias_opt)
        self.flow2 = nn.Sequential(nn.Conv3d(in_channel * 2, in_channel, 3, 1, 1, bias=bias_opt),
                                   nn.Conv3d(in_channel, 3, 3, 1, 1, bias=bias_opt))

        self.up4 = nn.ConvTranspose3d(in_channel * 2, in_channel, 2, 2, bias=bias_opt)
        self.ada_4 = nn.Linear(4096, in_channel * 1024, bias_opt)
        self.llama_up4 = nn.ConvTranspose3d(in_channel * 2, in_channel, 2, 2,
                                            bias=bias_opt)
        self.llama_conv4 = ConvBlock(in_channel * 4 + 4, in_channel, False, bias_opt)
        self.flow3 = nn.Sequential(nn.Conv3d(in_channel, in_channel, 3, 1, 1, bias=bias_opt),
                                   nn.Conv3d(in_channel, 3, 3, 1, 1, bias=bias_opt))

        # self.llama_up3 = nn.ConvTranspose3d(int(self.llama_out_channel / 4), int(self.llama_out_channel / 8), 2, 2,
        #                                     bias=bias_opt)
        # self.llama_up4 = nn.ConvTranspose3d(int(self.llama_out_channel / 8), int(self.llama_out_channel / 16), 2, 2,
        #                                     bias=bias_opt)

        self.up_tri = nn.Upsample(scale_factor=2, mode='trilinear')

        self.stn_4 = SpatialTransform([int(imgshape[0] / 4), int(imgshape[1] / 4), int(imgshape[2] / 4)])
        self.stn_2 = SpatialTransform([int(imgshape[0] / 2), int(imgshape[1] / 2), int(imgshape[2] / 2)])
        self.stn_1 = SpatialTransform(imgshape)

        self.down4 = nn.Upsample(scale_factor=0.25, mode='trilinear')
        self.down2 = nn.Upsample(scale_factor=0.5, mode='trilinear')

        #pos

    def forward(self, x, y):
        # x: m
        x_m = x
        x1 = self.conv_m_1(x)

        x2 = self.d_conv_m_1(x1)
        x2 = self.conv_m_2(x2)

        x3 = self.d_conv_m_2(x2)
        x3 = self.conv_m_3(x3)

        x4 = self.d_conv_m_3(x3)
        x4 = self.conv_m_4(x4)

        x = self.d_conv_m_4(x4)

        # y: f
        y_f = y
        y1 = self.conv_f_1(y)  # 1 #16

        y2 = self.d_conv_f_1(y1)  # 1/2
        y2 = self.conv_f_2(y2)  # 32

        y3 = self.d_conv_f_2(y2)  # 1/4
        y3 = self.conv_f_3(y3)  # 64

        y4 = self.d_conv_f_3(y3)
        y4 = self.conv_f_4(y4)  # 128

        y = self.d_conv_f_4(y4)

        xy = torch.cat([x, y], dim=1)
        mid_fea = rearrange(xy, "B C D H W -> B (D H W) C")  # 1024
        xy=self.pos_drop(self.pos_emb+mid_fea)

        llama_xy = self.llama_block(xy)

        xy = self.ada_1(llama_xy)
        xy = rearrange(xy, "B (D H W) C -> B C D H W", D=self.patch[0], H=self.patch[1], W=self.patch[2])
        xy = self.llama_up1(xy)
        xy = torch.cat([x4, y4, xy], dim=1)
        xy_pre_layer = self.llama_conv1(xy)  # 1/8
        fea_1=xy_pre_layer

        xy_pre_layer = self.up2(xy_pre_layer)
        xy = self.ada_2(llama_xy)
        xy = rearrange(xy, "B (D H W) (C d h w) -> B C (D d) (H h) (W w)", D=self.patch[0], H=self.patch[1],
                       W=self.patch[2], d=2, h=2, w=2)
        xy = self.llama_up2(xy)
        xy = torch.cat([x3, y3, xy_pre_layer, xy], dim=1)
        xy_pre_layer = self.llama_conv2(xy)  # 1/4
        fea_2=xy_pre_layer
        f1 = self.flow1(xy_pre_layer)
        warped_x_4 = self.stn_4(x_m, f1)
        f1_ = self.up_tri(f1)

        warped_x_2 = self.stn_2(x_m, f1_)
        xy_pre_layer = self.up3(xy_pre_layer)
        xy = self.ada_3(llama_xy)
        xy = rearrange(xy, "B (D H W) (C d h w) -> B C (D d) (H h) (W w)", D=self.patch[0], H=self.patch[1],
                       W=self.patch[2], d=4, h=4, w=4)
        xy = self.llama_up3(xy)
        xy = torch.cat([x2, y2, xy_pre_layer, xy, f1_, warped_x_2], dim=1)
        xy_pre_layer = self.llama_conv3(xy)  # 1/4
        fea_3=xy_pre_layer
        f2 = self.flow2(xy_pre_layer)
        f2 = f1_ + f2
        warped_x_2 = self.stn_2(x_m, f2)
        f2_ = self.up_tri(f2)

        warped_x_1 = self.stn_1(x_m, f2_)
        xy_pre_layer = self.up4(xy_pre_layer)
        xy = self.ada_4(llama_xy)
        xy = rearrange(xy, "B (D H W) (C d h w) -> B C (D d) (H h) (W w)", D=self.patch[0], H=self.patch[1],
                       W=self.patch[2], d=8, h=8, w=8)
        xy = self.llama_up4(xy)
        xy = torch.cat([x1, y1, xy_pre_layer, xy, f2_, warped_x_1], dim=1)
        xy_pre_layer = self.llama_conv4(xy)  # 1/4
        fea_4=xy_pre_layer
        f3 = self.flow3(xy_pre_layer)
        f3 = f2_ + f3
        warped_x_1 = self.stn_1(x_m, f3)
        # f2_ = self.up_tri(f2)

        # xy

        # llm_xy_4,llm_xy_3,llm_xy_2,llm_xy_1=
        # x_y_xy=torch.cat([x,y,llm_xy_4],dim=1)
        y4 = self.down4(y_f)
        y2 = self.down2(y_f)

        return warped_x_4, f1, y4, warped_x_2, f2, y2, warped_x_1, f3, y_f,mid_fea,fea_1,fea_2,fea_3,fea_4

class SecDecoder(nn.Module):
    def __init__(self,model1=None,in_channel=16,mlp_mul=2,llama_model=None,imgshape=(96,112,96),patch=(10, 10, 6),bias_opt=True):
        super(SecDecoder, self).__init__()
        # self.model1=DualLLama(in_channel=16, llama_model=llama, imgshape=(96, 112, 96)).cuda()
        self.model1=model1
        self.patch=patch
        self.llama_block = nn.Sequential(
            nn.Linear(2 * 16 * in_channel, mlp_mul * 2 * 16 * in_channel, bias_opt),
            nn.Linear(mlp_mul * 2 * 16 * in_channel, 4096, bias_opt),
            llama_model,

            nn.Linear(4096, mlp_mul * 2 * 16 * in_channel),
            nn.Linear(mlp_mul * 2 * 16 * in_channel, 4096),
            # nn.LayerNorm(4096),
            #
            # nn.Linear(4096, mlp_mul * 2 * 16 * in_channel, bias_opt),
            # nn.Linear(2 * 16 * mlp_mul * in_channel, 4096, bias_opt),
            llama_model,
            # nn.Linear(4096, mlp_mul * 2 * 16 * in_channel),
            # nn.Linear(2 * 16 * mlp_mul * in_channel, self.llama_out_channel),
            # nn.LayerNorm(self.llama_out_channel)
        )
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.patch[0] * self.patch[1] * self.patch[2], 2 * 16 * in_channel))
        self.pos_drop = nn.Dropout(0.)

        self.ada_1 = nn.Linear(4096, in_channel * 16, bias_opt)

        self.llama_up1 = nn.ConvTranspose3d(in_channel * 16, in_channel * 8, 2, 2,
                                            bias=bias_opt)
        self.llama_conv1 = ConvBlock(in_channel * 8 * 2, in_channel * 8, False, bias_opt)

        self.up2 = nn.ConvTranspose3d(in_channel * 8, in_channel * 4, 2, 2, bias=bias_opt)
        self.ada_2 = nn.Linear(4096, in_channel * 64, bias_opt)
        self.llama_up2 = nn.ConvTranspose3d(in_channel * 8, in_channel * 4, 2, 2,
                                            bias=bias_opt)
        self.llama_conv2 = ConvBlock(in_channel * 4 +in_channel*4+in_channel*4, in_channel * 4, False, bias_opt)
        self.flow1 = nn.Sequential(nn.Conv3d(in_channel * 4, in_channel, 3, 1, 1, bias=bias_opt),
                                   nn.Conv3d(in_channel, 3, 3, 1, 1, bias=bias_opt))

        self.up3 = nn.ConvTranspose3d(in_channel * 4, in_channel * 2, 2, 2, bias=bias_opt)
        self.ada_3 = nn.Linear(4096, in_channel * 256, bias_opt)
        self.llama_up3 = nn.ConvTranspose3d(in_channel * 4, in_channel * 2, 2, 2,
                                            bias=bias_opt)
        self.llama_conv3 = ConvBlock(in_channel * 2 * 3 + 4, in_channel * 2, False, bias_opt)
        self.flow2 = nn.Sequential(nn.Conv3d(in_channel * 2, in_channel, 3, 1, 1, bias=bias_opt),
                                   nn.Conv3d(in_channel, 3, 3, 1, 1, bias=bias_opt))

        self.up4 = nn.ConvTranspose3d(in_channel * 2, in_channel, 2, 2, bias=bias_opt)
        self.ada_4 = nn.Linear(4096, in_channel * 1024, bias_opt)
        self.llama_up4 = nn.ConvTranspose3d(in_channel * 2, in_channel, 2, 2,
                                            bias=bias_opt)
        self.llama_conv4 = ConvBlock(in_channel * 3 + 4, in_channel, False, bias_opt)
        self.flow3 = nn.Sequential(nn.Conv3d(in_channel, in_channel, 3, 1, 1, bias=bias_opt),
                                   nn.Conv3d(in_channel, 3, 3, 1, 1, bias=bias_opt))

        # self.llama_up3 = nn.ConvTranspose3d(int(self.llama_out_channel / 4), int(self.llama_out_channel / 8), 2, 2,
        #                                     bias=bias_opt)
        # self.llama_up4 = nn.ConvTranspose3d(int(self.llama_out_channel / 8), int(self.llama_out_channel / 16), 2, 2,
        #                                     bias=bias_opt)

        self.up_tri = nn.Upsample(scale_factor=2, mode='trilinear')

        self.stn_4 = SpatialTransform([int(imgshape[0] / 4), int(imgshape[1] / 4), int(imgshape[2] / 4)])
        self.stn_2 = SpatialTransform([int(imgshape[0] / 2), int(imgshape[1] / 2), int(imgshape[2] / 2)])
        self.stn_1 = SpatialTransform(imgshape)

        self.down4 = nn.Upsample(scale_factor=0.25, mode='trilinear')
        self.down2 = nn.Upsample(scale_factor=0.5, mode='trilinear')

    def forward(self,x,y):

        warped_x_4, f1, y4, warped_x_2, f2, y2, warped_x_1, f3, y_f,mid_fea,fea_1,fea_2,fea_3,fea_4=self.model1(x,y)

        xy=self.pos_drop(self.pos_emb+mid_fea)

        llama_xy = self.llama_block(xy)

        xy = self.ada_1(llama_xy)
        xy = rearrange(xy, "B (D H W) C -> B C D H W", D=self.patch[0], H=self.patch[1], W=self.patch[2])
        xy = self.llama_up1(xy)
        xy = torch.cat([fea_1,xy], dim=1)
        xy_pre_layer = self.llama_conv1(xy)  # 1/8
        fea_1=xy_pre_layer

        xy_pre_layer = self.up2(xy_pre_layer)
        xy = self.ada_2(llama_xy)
        xy = rearrange(xy, "B (D H W) (C d h w) -> B C (D d) (H h) (W w)", D=self.patch[0], H=self.patch[1],
                       W=self.patch[2], d=2, h=2, w=2)
        xy = self.llama_up2(xy)
        xy = torch.cat([fea_2,xy_pre_layer, xy], dim=1)
        xy_pre_layer = self.llama_conv2(xy)  # 1/4
        fea_2=xy_pre_layer
        f1 = self.flow1(xy_pre_layer)+f1
        warped_x_4 = self.stn_4(x, f1)
        f1_ = self.up_tri(f1)

        warped_x_2 = self.stn_2(x, f1_)
        xy_pre_layer = self.up3(xy_pre_layer)
        xy = self.ada_3(llama_xy)
        xy = rearrange(xy, "B (D H W) (C d h w) -> B C (D d) (H h) (W w)", D=self.patch[0], H=self.patch[1],
                       W=self.patch[2], d=4, h=4, w=4)
        xy = self.llama_up3(xy)
        xy = torch.cat([fea_3,xy_pre_layer, xy, f1_, warped_x_2], dim=1)
        xy_pre_layer = self.llama_conv3(xy)  # 1/4
        fea_3=xy_pre_layer
        f2 = self.flow2(xy_pre_layer)+f2
        warped_x_2 = self.stn_2(x, f2)
        f2_ = self.up_tri(f2)

        warped_x_1 = self.stn_1(x, f2_)
        xy_pre_layer = self.up4(xy_pre_layer)
        xy = self.ada_4(llama_xy)
        xy = rearrange(xy, "B (D H W) (C d h w) -> B C (D d) (H h) (W w)", D=self.patch[0], H=self.patch[1],
                       W=self.patch[2], d=8, h=8, w=8)
        xy = self.llama_up4(xy)
        xy = torch.cat([fea_4,xy_pre_layer, xy, f2_, warped_x_1], dim=1)
        xy_pre_layer = self.llama_conv4(xy)  # 1/4
        fea_4=xy_pre_layer
        f3 = self.flow3(xy_pre_layer)+f3
        warped_x_1 = self.stn_1(x, f3)

        return warped_x_4, f1, y4, warped_x_2, f2, y2, warped_x_1, f3, y_f,mid_fea,fea_1,fea_2,fea_3,fea_4

class OtherDecoder(nn.Module):
    def __init__(self,model=None,in_channel=16,mlp_mul=2,llama_model=None,imgshape=(96,112,96),patch=(10, 10, 6),bias_opt=True):
        super(OtherDecoder, self).__init__()
        # self.model1=DualLLama(in_channel=16, llama_model=llama, imgshape=(96, 112, 96)).cuda()
        self.model=model
        self.patch=patch
        self.llama_block = nn.Sequential(
            nn.Linear(2 * 16 * in_channel, mlp_mul * 2 * 16 * in_channel, bias_opt),
            nn.Linear(mlp_mul * 2 * 16 * in_channel, 4096, bias_opt),
            llama_model,

            nn.Linear(4096, mlp_mul * 2 * 16 * in_channel),
            nn.Linear(mlp_mul * 2 * 16 * in_channel, 4096),
            # nn.LayerNorm(4096),
            #
            # nn.Linear(4096, mlp_mul * 2 * 16 * in_channel, bias_opt),
            # nn.Linear(2 * 16 * mlp_mul * in_channel, 4096, bias_opt),
            llama_model,
            # nn.Linear(4096, mlp_mul * 2 * 16 * in_channel),
            # nn.Linear(2 * 16 * mlp_mul * in_channel, self.llama_out_channel),
            # nn.LayerNorm(self.llama_out_channel)
        )
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.patch[0] * self.patch[1] * self.patch[2], 2 * 16 * in_channel))
        self.pos_drop = nn.Dropout(0.)

        self.ada_1 = nn.Linear(4096, in_channel * 16, bias_opt)

        self.llama_up1 = nn.ConvTranspose3d(in_channel * 16, in_channel * 8, 2, 2,
                                            bias=bias_opt)
        self.llama_conv1 = ConvBlock(in_channel * 8 * 2, in_channel * 8, False, bias_opt)

        self.up2 = nn.ConvTranspose3d(in_channel * 8, in_channel * 4, 2, 2, bias=bias_opt)
        self.ada_2 = nn.Linear(4096, in_channel * 64, bias_opt)
        self.llama_up2 = nn.ConvTranspose3d(in_channel * 8, in_channel * 4, 2, 2,
                                            bias=bias_opt)
        self.llama_conv2 = ConvBlock(in_channel * 4 +in_channel*4+in_channel*4, in_channel * 4, False, bias_opt)
        self.flow1 = nn.Sequential(nn.Conv3d(in_channel * 4, in_channel, 3, 1, 1, bias=bias_opt),
                                   nn.Conv3d(in_channel, 3, 3, 1, 1, bias=bias_opt))

        self.up3 = nn.ConvTranspose3d(in_channel * 4, in_channel * 2, 2, 2, bias=bias_opt)
        self.ada_3 = nn.Linear(4096, in_channel * 256, bias_opt)
        self.llama_up3 = nn.ConvTranspose3d(in_channel * 4, in_channel * 2, 2, 2,
                                            bias=bias_opt)
        self.llama_conv3 = ConvBlock(in_channel * 2 * 3 + 4, in_channel * 2, False, bias_opt)
        self.flow2 = nn.Sequential(nn.Conv3d(in_channel * 2, in_channel, 3, 1, 1, bias=bias_opt),
                                   nn.Conv3d(in_channel, 3, 3, 1, 1, bias=bias_opt))

        self.up4 = nn.ConvTranspose3d(in_channel * 2, in_channel, 2, 2, bias=bias_opt)
        self.ada_4 = nn.Linear(4096, in_channel * 1024, bias_opt)
        self.llama_up4 = nn.ConvTranspose3d(in_channel * 2, in_channel, 2, 2,
                                            bias=bias_opt)
        self.llama_conv4 = ConvBlock(in_channel * 3 + 4, in_channel, False, bias_opt)
        self.flow3 = nn.Sequential(nn.Conv3d(in_channel, in_channel, 3, 1, 1, bias=bias_opt),
                                   nn.Conv3d(in_channel, 3, 3, 1, 1, bias=bias_opt))

        # self.llama_up3 = nn.ConvTranspose3d(int(self.llama_out_channel / 4), int(self.llama_out_channel / 8), 2, 2,
        #                                     bias=bias_opt)
        # self.llama_up4 = nn.ConvTranspose3d(int(self.llama_out_channel / 8), int(self.llama_out_channel / 16), 2, 2,
        #                                     bias=bias_opt)

        self.up_tri = nn.Upsample(scale_factor=2, mode='trilinear')

        self.stn_4 = SpatialTransform([int(imgshape[0] / 4), int(imgshape[1] / 4), int(imgshape[2] / 4)])
        self.stn_2 = SpatialTransform([int(imgshape[0] / 2), int(imgshape[1] / 2), int(imgshape[2] / 2)])
        self.stn_1 = SpatialTransform(imgshape)

        self.down4 = nn.Upsample(scale_factor=0.25, mode='trilinear')
        self.down2 = nn.Upsample(scale_factor=0.5, mode='trilinear')

    def forward(self,x,y):

        warped_x_4, f1, y4, warped_x_2, f2, y2, warped_x_1, f3, y_f,mid_fea,fea_1,fea_2,fea_3,fea_4=self.model(x,y)

        xy=self.pos_drop(self.pos_emb+mid_fea)

        llama_xy = self.llama_block(xy)

        xy = self.ada_1(llama_xy)
        xy = rearrange(xy, "B (D H W) C -> B C D H W", D=self.patch[0], H=self.patch[1], W=self.patch[2])
        xy = self.llama_up1(xy)
        xy = torch.cat([fea_1,xy], dim=1)
        xy_pre_layer = self.llama_conv1(xy)  # 1/8
        fea_1=xy_pre_layer

        xy_pre_layer = self.up2(xy_pre_layer)
        xy = self.ada_2(llama_xy)
        xy = rearrange(xy, "B (D H W) (C d h w) -> B C (D d) (H h) (W w)", D=self.patch[0], H=self.patch[1],
                       W=self.patch[2], d=2, h=2, w=2)
        xy = self.llama_up2(xy)
        xy = torch.cat([fea_2,xy_pre_layer, xy], dim=1)
        xy_pre_layer = self.llama_conv2(xy)  # 1/4
        fea_2=xy_pre_layer
        f1 = self.flow1(xy_pre_layer)+f1
        warped_x_4 = self.stn_4(x, f1)
        f1_ = self.up_tri(f1)

        warped_x_2 = self.stn_2(x, f1_)
        xy_pre_layer = self.up3(xy_pre_layer)
        xy = self.ada_3(llama_xy)
        xy = rearrange(xy, "B (D H W) (C d h w) -> B C (D d) (H h) (W w)", D=self.patch[0], H=self.patch[1],
                       W=self.patch[2], d=4, h=4, w=4)
        xy = self.llama_up3(xy)
        xy = torch.cat([fea_3,xy_pre_layer, xy, f1_, warped_x_2], dim=1)
        xy_pre_layer = self.llama_conv3(xy)  # 1/4
        fea_3=xy_pre_layer
        f2 = self.flow2(xy_pre_layer)+f2
        warped_x_2 = self.stn_2(x, f2)
        f2_ = self.up_tri(f2)

        warped_x_1 = self.stn_1(x, f2_)
        xy_pre_layer = self.up4(xy_pre_layer)
        xy = self.ada_4(llama_xy)
        xy = rearrange(xy, "B (D H W) (C d h w) -> B C (D d) (H h) (W w)", D=self.patch[0], H=self.patch[1],
                       W=self.patch[2], d=8, h=8, w=8)
        xy = self.llama_up4(xy)
        xy = torch.cat([fea_4,xy_pre_layer, xy, f2_, warped_x_1], dim=1)
        xy_pre_layer = self.llama_conv4(xy)  # 1/4
        fea_4=xy_pre_layer
        f3 = self.flow3(xy_pre_layer)+f3
        warped_x_1 = self.stn_1(x, f3)

        return warped_x_4, f1, y4, warped_x_2, f2, y2, warped_x_1, f3, y_f,mid_fea,fea_1,fea_2,fea_3,fea_4


class DownConv(nn.Module):
    def __init__(self, in_channel, bias_opt):
        super(DownConv, self).__init__()
        self.conv = nn.Conv3d(in_channel, 2 * in_channel, 3, 2, 1, bias=bias_opt)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, start=False, bias_opt=True):
        super(ConvBlock, self).__init__()
        if not start:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1, bias=bias_opt),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channel, out_channel, 3, 1, 1, bias=bias_opt),
                nn.LeakyReLU(0.2))

        else:
            self.conv = nn.Sequential(
                nn.Conv3d(1, in_channel, 3, 1, 1, bias=bias_opt),
                nn.LeakyReLU(0.2),
                nn.Conv3d(in_channel, in_channel, 3, 1, 1, bias=bias_opt),
                nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class SpatialTransform(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size):
        super(SpatialTransform, self).__init__()

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow, mode='bilinear'):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=mode)

