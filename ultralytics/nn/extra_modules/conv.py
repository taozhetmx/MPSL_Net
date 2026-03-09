import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from torch.utils.checkpoint import checkpoint

# from mmcv.cnn import CONV_LAYERS
from torch import Tensor
import math
# from timm.models.layers import trunc_normal_
import time
import math
import copy
from functools import partial
from typing import Optional, Callable

import timm
from typing import List, Optional, Tuple


from collections import OrderedDict


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class KernelSpatialModulation_Global(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16,
                 temp=1.0, kernel_temp=None, kernel_att_init='dyconv_as_extra', att_multi=2.0,
                 ksm_only_kernel_att=False, att_grid=1, stride=1, spatial_freq_decompose=False,
                 act_type='sigmoid'):
        super(KernelSpatialModulation_Global, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num

        self.temperature = temp
        self.kernel_temp = kernel_temp

        self.ksm_only_kernel_att = ksm_only_kernel_att

        # self.temperature = nn.Parameter(torch.FloatTensor([temp]), requires_grad=True)
        self.kernel_att_init = kernel_att_init
        self.att_multi = att_multi
        # self.kn = nn.Parameter(torch.FloatTensor([kernel_num]), requires_grad=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.att_grid = att_grid
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        # self.bn = nn.Identity()
        self.bn = nn.BatchNorm2d(attention_channel)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = StarReLU()
        # self.dropout = nn.Dropout2d(p=0.1)
        # self.sp_att = SpatialGate(stride=stride, out_channels=1)

        # self.attup = AttUpsampler(inplane=in_planes, flow_make_k=1)

        self.spatial_freq_decompose = spatial_freq_decompose
        # self.channel_compress = ChannelPool()
        # self.channel_spatial = BasicConv(
        #     # 2, 1, 7, stride=1, padding=(7 - 1) // 2, relu=False
        #     2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        # )
        # self.filter_spatial = BasicConv(
        #     # 2, 1, 7, stride=stride, padding=(7 - 1) // 2, relu=False
        #     2, 1, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, relu=False
        # )
        if ksm_only_kernel_att:
            self.func_channel = self.skip
        else:
            if spatial_freq_decompose:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes * 2 if self.kernel_size > 1 else in_planes, 1,
                                            bias=True)
            else:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
            # self.channel_fc_bias = nn.Parameter(torch.zeros(1, in_planes, 1, 1), requires_grad=True)
            self.func_channel = self.get_channel_attention

        if (in_planes == groups and in_planes == out_planes) or self.ksm_only_kernel_att:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            if spatial_freq_decompose:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes * 2, 1, stride=stride, bias=True)
            else:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, stride=stride, bias=True)
            # self.filter_fc_bias = nn.Parameter(torch.zeros(1, in_planes, 1, 1), requires_grad=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1 or self.ksm_only_kernel_att:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            # self.kernel_fc = nn.Conv2d(attention_channel, kernel_num * kernel_size * kernel_size, 1, bias=True)
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if hasattr(self, 'channel_spatial'):
            nn.init.normal_(self.channel_spatial.conv.weight, std=1e-6)
        if hasattr(self, 'filter_spatial'):
            nn.init.normal_(self.filter_spatial.conv.weight, std=1e-6)

        if hasattr(self, 'spatial_fc') and isinstance(self.spatial_fc, nn.Conv2d):
            # nn.init.constant_(self.spatial_fc.weight, 0)
            nn.init.normal_(self.spatial_fc.weight, std=1e-6)
            # self.spatial_fc.weight *= 1e-6
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            else:
                # nn.init.constant_(self.spatial_fc.weight, 0)
                # nn.init.constant_(self.spatial_fc.bias, 0)
                pass

        if hasattr(self, 'func_filter') and isinstance(self.func_filter, nn.Conv2d):
            # nn.init.constant_(self.func_filter.weight, 0)
            nn.init.normal_(self.func_filter.weight, std=1e-6)
            # self.func_filter.weight *= 1e-6
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            else:
                # nn.init.constant_(self.func_filter.weight, 0)
                # nn.init.constant_(self.func_filter.bias, 0)
                pass

        if hasattr(self, 'kernel_fc') and isinstance(self.kernel_fc, nn.Conv2d):
            # nn.init.constant_(self.kernel_fc.weight, 0)
            nn.init.normal_(self.kernel_fc.weight, std=1e-6)
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
                # nn.init.constant_(self.kernel_fc.weight, 0)
                # nn.init.constant_(self.kernel_fc.bias, -10)
                # nn.init.constant_(self.kernel_fc.weight[0], 6)
                # nn.init.constant_(self.kernel_fc.weight[1:], -6)
            else:
                # nn.init.constant_(self.kernel_fc.weight, 0)
                # nn.init.constant_(self.kernel_fc.bias, 0)
                # nn.init.constant_(self.kernel_fc.bias, -10)
                # nn.init.constant_(self.kernel_fc.bias[0], 10)
                pass

        if hasattr(self, 'channel_fc') and isinstance(self.channel_fc, nn.Conv2d):
            # nn.init.constant_(self.channel_fc.weight, 0)
            nn.init.normal_(self.channel_fc.weight, std=1e-6)
            # nn.init.constant_(self.channel_fc.bias[1], 6)
            # nn.init.constant_(self.channel_fc.bias, 0)
            if self.kernel_att_init == 'dyconv_as_extra':
                pass
            else:
                # nn.init.constant_(self.channel_fc.weight, 0)
                # nn.init.constant_(self.channel_fc.bias, 0)
                pass

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        if self.act_type == 'sigmoid':
            channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(
                -1)) / self.temperature) * self.att_multi  # b, kn, cout, cin, k, k
        elif self.act_type == 'tanh':
            channel_attention = 1 + torch.tanh_(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(
                -1)) / self.temperature)  # b, kn, cout, cin, k, k
        else:
            raise NotImplementedError
        # channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi # b, kn, cout, cin, k, k
        # channel_attention = torch.sigmoid(self.channel_fc(x) / self.temperature) * self.att_multi # b, kn, cout, cin, k, k
        # channel_attention = self.channel_fc(x) # b, kn, cout, cin, k, k
        # channel_attention = torch.tanh_(self.channel_fc(x) / self.temperature) + 1 # b, kn, cout, cin, k, k
        return channel_attention

    def get_filter_attention(self, x):
        if self.act_type == 'sigmoid':
            filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(
                -1)) / self.temperature) * self.att_multi  # b, kn, cout, cin, k, k
        elif self.act_type == 'tanh':
            filter_attention = 1 + torch.tanh_(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(
                -1)) / self.temperature)  # b, kn, cout, cin, k, k
        else:
            raise NotImplementedError
        # filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi # b, kn, cout, cin, k, k
        # filter_attention = self.filter_fc(x) # b, kn, cout, cin, k, k
        # filter_attention = torch.tanh_(self.filter_fc(x) / self.temperature) + 1 # b, kn, cout, cin, k, k
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        if self.act_type == 'sigmoid':
            spatial_attention = torch.sigmoid(spatial_attention / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            spatial_attention = 1 + torch.tanh_(spatial_attention / self.temperature)
        else:
            raise NotImplementedError
        return spatial_attention

    def get_kernel_attention(self, x):
        # kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, self.kernel_size, self.kernel_size)
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type == 'softmax':
            kernel_attention = F.softmax(kernel_attention / self.kernel_temp, dim=1)
        elif self.act_type == 'sigmoid':
            kernel_attention = torch.sigmoid(kernel_attention / self.kernel_temp) * 2 / kernel_attention.size(1)
        elif self.act_type == 'tanh':
            kernel_attention = (1 + torch.tanh(kernel_attention / self.kernel_temp)) / kernel_attention.size(1)
        else:
            raise NotImplementedError

        # kernel_attention = kernel_attention / self.temperature
        # kernel_attention = kernel_attention / kernel_attention.abs().sum(dim=1, keepdims=True)
        return kernel_attention

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        # comp_x = self.channel_compress(x)
        # csg = self.channel_spatial(comp_x).sigmoid_() * self.att_multi
        # csg = 1
        # fsg = self.filter_spatial(comp_x).sigmoid_() * self.att_multi
        # fsg = 1
        # x_h = x.mean(dim=-1, keepdims=True)
        # x_w = x.mean(dim=-2, keepdims=True)
        # x_h = self.relu(self.bn(self.fc(x_h)))
        # x_w = self.relu(self.bn(self.fc(x_w)))
        # avg_x = (self.avgpool(x_h) + self.avgpool(x_w)) * 0.5
        # avg_x = self.avgpool(self.relu(self.bn(self.fc(x))))
        avg_x = self.relu(self.bn(self.fc(x)))
        return self.func_channel(avg_x), self.func_filter(avg_x), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return self.attup.flow_warp(self.func_channel(x), grid), self.attup.flow_warp(self.func_filter(x), grid), self.func_spatial(avg_x), self.func_kernel(avg_x), sp_gate
        # return (self.func_channel(x_h) * self.func_channel(x_w)).sqrt(), (self.func_filter(x_h) * self.func_filter(x_w)).sqrt(), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return (self.func_channel(x_h) * self.func_channel(x_w)), (self.func_filter(x_h) * self.func_filter(x_w)), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return ((self.func_channel(x_h) + self.func_channel(x_w)) * csg).sigmoid_() * self.att_multi, ((self.func_filter(x_h) + self.func_filter(x_w)) * fsg).sigmoid_() * self.att_multi, self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return (self.func_channel(x_h) * self.func_channel(x_w) * csg), (self.func_filter(x_h) * self.func_filter(x_w) * fsg), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # return (self.dropout(self.func_channel(x_h) * self.func_channel(x_w))), (self.dropout(self.func_filter(x_h) * self.func_filter(x_w))), self.func_spatial(avg_x), self.func_kernel(avg_x)
        # k_att = F.relu(self.func_kernel(x) - 0.8 * self.func_kernel(x_inverse))
        # k_att = k_att / (k_att.sum(dim=1, keepdim=True) + 1e-8)
        # return self.func_channel(x), self.func_filter(x), self.func_spatial(x), k_att


class KernelSpatialModulation_Local(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        super(KernelSpatialModulation_Local, self).__init__()
        self.kn = kernel_num
        self.out_n = out_n
        self.channel = channel
        if channel is not None: k_size = round((math.log2(channel) / 2) + 0.5) // 2 * 2 + 1
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6)
        self.use_global = use_global
        if self.use_global:
            self.complex_weight = nn.Parameter(torch.randn(1, self.channel // 2 + 1, 2, dtype=torch.float32) * 1e-6)
            # self.norm = nn.GroupNorm(num_groups=32, num_channels=channel)
        self.norm = nn.LayerNorm(self.channel)
        # self.norm_std = nn.LayerNorm(self.channel)
        # trunc_normal_(self.complex_weight, std=.02)
        # self.sigmoid = nn.Sigmoid()
        # nn.init.constant(self.conv.weight.data) # nn.init.normal_(self.conv.weight, std=1e-6)
        # nn.init.zeros_(self.conv.weight)

    def forward(self, x, x_std=None):
        # feature descriptor on the global spatial information
        # y = self.avg_pool(x)
        # b,c,1, -> b,1,c, -> b, kn * out_n, c
        # x = torch.cat([x, x_std], dim=-2)
        x = x.squeeze(-1).transpose(-1, -2)  # b,1,c,
        b, _, c = x.shape
        if self.use_global:
            x_rfft = torch.fft.rfft(x.float(), dim=-1)  # b, 1 or 2, c // 2 +1
            # print(x_rfft.shape)
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)),
                                    dim=-1)  # b, 1, c // 2 +1
        x = self.norm(x)
        # x = torch.stack([self.norm(x[:, 0]), self.norm_std(x[:, 1])], dim=1)
        # b,1,c, -> b, kn * out_n, c
        att_logit = self.conv(x)
        # print(att_logit.shape)
        # print(att.shape)
        # Multi-scale information fusion
        # att = self.sigmoid(att) * 2
        att_logit = att_logit.reshape(x.size(0), self.kn, self.out_n, c)  # b, kn, k1*k2, cin
        att_logit = att_logit.permute(0, 1, 3, 2)  # b, kn, cin, k1*k2
        # print(att_logit.shape)
        return att_logit


import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyBandModulation(nn.Module):
    def __init__(self,
                 in_channels,
                 k_list=[2],
                 lowfreq_att=False,
                 fs_feat='feat',
                 act='sigmoid',
                 spatial='conv',
                 spatial_group=1,
                 spatial_kernel=3,
                 init='zero',
                 max_size=(64, 64),  # 预计算mask的最大尺寸
                 **kwargs,
                 ):
        super().__init__()
        self.k_list = k_list
        self.lowfreq_att = lowfreq_att
        self.in_channels = in_channels
        self.fs_feat = fs_feat
        self.act = act

        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group

        # 构建注意力卷积层 (这部分逻辑不变)
        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att:
                _n += 1
            for i in range(_n):
                freq_weight_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.spatial_group,
                    stride=1,
                    kernel_size=spatial_kernel,
                    groups=self.spatial_group,
                    padding=spatial_kernel // 2,
                    bias=True
                )
                if init == 'zero':
                    nn.init.normal_(freq_weight_conv.weight, std=1e-6)
                    if freq_weight_conv.bias is not None:
                        freq_weight_conv.bias.data.zero_()
                self.freq_weight_conv_list.append(freq_weight_conv)
            # freq_weight_conv = nn.Conv2d(
            #         in_channels=in_channels,
            #         out_channels=self.spatial_group * _n,
            #         stride=1,
            #         kernel_size=spatial_kernel,
            #         groups=self.spatial_group,
            #         padding=spatial_kernel // 2,
            #         bias=True
            #     )
            # if init == 'zero':
            #     nn.init.normal_(freq_weight_conv.weight, std=1e-6)
            #     if freq_weight_conv.bias is not None:
            #         freq_weight_conv.bias.data.zero_()
        else:
            raise NotImplementedError

        # 【优化核心】预计算并缓存不同频率的mask
        self.register_buffer('cached_masks', self._precompute_masks(max_size, k_list))

    def _precompute_masks(self, max_size, k_list):
        """
        在初始化时预先计算一组最大尺寸的掩码。
        """
        max_h, max_w = max_size
        _, freq_indices = get_fft2freq(d1=max_h, d2=max_w, use_rfft=True)
        # print(freq_indices.shape)
        # print(freq_indices)
        freq_indices = freq_indices.abs().max(dim=-1, keepdims=False)[0]  # (max_h, max_w//2 + 1)
        # print(freq_indices)

        # freq_list = [0, *[0.5 / freq for freq in k_list], 0.5]
        masks = []
        for freq in k_list:
            # 创建一个布尔掩码
            mask = freq_indices < 0.5 / freq + 1e-8
            # print(freq)
            # print(mask)
            masks.append(mask)

        # 将列表堆叠成一个张量 (num_masks, max_h, max_w//2 + 1)
        # 增加一个维度以方便广播
        return torch.stack(masks, dim=0).unsqueeze(1)  # (num_masks, 1, max_h, max_w//2 + 1)

    def sp_act(self, freq_weight):
        # (这部分逻辑不变)
        if self.act == 'sigmoid':
            return freq_weight.sigmoid() * 2
        elif self.act == 'tanh':
            return 1 + freq_weight.tanh()
        elif self.act == 'softmax':
            return freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError

    def forward(self, x, att_feat=None):
        if att_feat is None:
            att_feat = x

        x_list = []
        x = x.to(torch.float32)
        pre_x = x.clone()
        b, _, h, w = x.shape

        # x_fft = torch.fft.rfft2(x, norm='ortho').contiguous()
        # 移除了 .contiguous()，因为rfft2的输出通常是连续的。如果遇到性能问题可以再加回来。
        x_fft = torch.fft.rfft2(x, norm='ortho')

        # 【优化核心】获取并调整缓存的mask大小
        # 将缓存的mask插值到当前特征图的频域尺寸
        # 注意频域尺寸是 (h, w//2 + 1)
        freq_h, freq_w = h, w // 2 + 1

        # 将mask从 (num_masks, 1, max_h, max_w//2+1) 转为 (num_masks, 1, h, w//2+1)
        # 使用 nearest 插值，因为它对于0/1掩码来说既快速又准确
        current_masks = F.interpolate(self.cached_masks.float(), size=(freq_h, freq_w), mode='nearest')

        for idx, freq in enumerate(self.k_list):
            # 直接从缓存中获取mask
            mask = current_masks[idx]

            # 应用掩码并进行逆傅里叶变换
            # `s=(h,w)` 确保 irfft2 的输出尺寸与原始 `x` 匹配
            low_part = torch.fft.irfft2(x_fft * mask, s=(h, w), norm='ortho')

            high_part = pre_x - low_part
            pre_x = low_part

            # 注意力计算部分不变
            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self.sp_act(freq_weight)

            # 将注意力权重和高频部分相乘
            # 重塑形状以进行广播
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * \
                  high_part.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))

        # 处理低频部分
        if self.lowfreq_att:
            freq_weight = self.freq_weight_conv_list[len(self.k_list)](att_feat)
            freq_weight = self.sp_act(freq_weight)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * \
                  pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)

        return sum(x_list)


def get_fft2freq(d1, d2, use_rfft=False):
    # Frequency components for rows and columns
    freq_h = torch.fft.fftfreq(d1)  # Frequency for the rows (d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)  # Frequency for the columns (d2)
    else:
        freq_w = torch.fft.fftfreq(d2)

    # Meshgrid to create a 2D grid of frequency coordinates
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w), dim=-1)
    # print(freq_hw)
    # print(freq_hw.shape)
    # Calculate the distance from the origin (0, 0) in the frequency space
    dist = torch.norm(freq_hw, dim=-1)
    # print(dist.shape)
    # Sort the distances and get the indices
    sorted_dist, indices = torch.sort(dist.view(-1))  # Flatten the distance tensor for sorting
    # print(sorted_dist.shape)

    # Get the corresponding coordinates for the sorted distances
    if use_rfft:
        d2 = d2 // 2 + 1
        # print(d2)
    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)  # Convert flat indices to 2D coords
    # print(sorted_coords.shape)
    # # Print sorted distances and corresponding coordinates
    # for i in range(sorted_dist.shape[0]):
    #     print(f"Distance: {sorted_dist[i]:.4f}, Coordinates: ({sorted_coords[i, 0]}, {sorted_coords[i, 1]})")

    if False:
        # Plot the distance matrix as a grayscale image
        plt.imshow(dist.cpu().numpy(), cmap='gray', origin='lower')
        plt.colorbar()
        plt.title('Frequency Domain Distance')
        plt.show()
    return sorted_coords.permute(1, 0), freq_hw


# @CONV_LAYERS.register_module()  # for mmdet, mmseg
class FDConv(nn.Conv2d):
    def __init__(self,
                 *args,
                 reduction=0.0625,
                 kernel_num=4,
                 use_fdconv_if_c_gt=16,  # if channel greater or equal to 16, e.g., 64, 128, 256, 512
                 use_fdconv_if_k_in=[1, 3],  # if kernel_size in the list
                 use_fbm_if_k_in=[3],  # if kernel_size in the list
                 kernel_temp=1.0,
                 temp=None,
                 att_multi=2.0,
                 param_ratio=1,
                 param_reduction=1.0,
                 ksm_only_kernel_att=False,
                 att_grid=1,
                 use_ksm_local=True,
                 ksm_local_act='sigmoid',
                 ksm_global_act='sigmoid',
                 spatial_freq_decompose=False,
                 convert_param=True,
                 linear_mode=False,
                 fbm_cfg={
                     'k_list': [2, 4, 8],
                     'lowfreq_att': False,
                     'fs_feat': 'feat',
                     'act': 'sigmoid',
                     'spatial': 'conv',
                     'spatial_group': 1,
                     'spatial_kernel': 3,
                     'init': 'zero',
                     'global_selection': False,
                 },
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = use_fdconv_if_k_in
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.spatial_freq_decompose = spatial_freq_decompose
        self.use_fbm_if_k_in = use_fbm_if_k_in

        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act
        assert self.ksm_local_act in ['sigmoid', 'tanh']
        assert self.ksm_global_act in ['softmax', 'sigmoid', 'tanh']

        ### Kernel num & Kernel temp setting
        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)
        if temp is None:
            temp = kernel_temp

        print('*** kernel_num:', self.kernel_num)
        self.alpha = min(self.out_channels,
                         self.in_channels) // 2 * self.kernel_num * self.param_ratio / param_reduction
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[
            0] not in self.use_fdconv_if_k_in:
            return
        self.KSM_Global = KernelSpatialModulation_Global(self.in_channels, self.out_channels, self.kernel_size[0],
                                                         groups=self.groups,
                                                         temp=temp,
                                                         kernel_temp=kernel_temp,
                                                         reduction=reduction,
                                                         kernel_num=self.kernel_num * self.param_ratio,
                                                         kernel_att_init=None, att_multi=att_multi,
                                                         ksm_only_kernel_att=ksm_only_kernel_att,
                                                         act_type=self.ksm_global_act,
                                                         att_grid=att_grid, stride=self.stride,
                                                         spatial_freq_decompose=spatial_freq_decompose)

        if self.kernel_size[0] in use_fbm_if_k_in:
            self.FBM = FrequencyBandModulation(self.in_channels, **fbm_cfg)
            # self.FBM = OctaveFrequencyAttention(2 * self.in_channels // 16, **fbm_cfg)
            # self.channel_comp = ChannelPool(reduction=16)

        if self.use_ksm_local:
            self.KSM_Local = KernelSpatialModulation_Local(channel=self.in_channels, kernel_num=1, out_n=int(
                self.out_channels * self.kernel_size[0] * self.kernel_size[1]))

        self.linear_mode = linear_mode
        self.convert2dftweight(convert_param)

    def convert2dftweight(self, convert_param):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True)  # 2, d1 * k1 * (d2 * k2 // 2 + 1)
        # freq_indices = freq_indices.reshape(2, self.kernel_num, -1)
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))  # d1 * k1, d2 * k2 // 2 + 1
        if self.param_reduction < 1:
            freq_indices = freq_indices[:, torch.randperm(freq_indices.size(1), generator=torch.Generator().manual_seed(
                freq_indices.size(1)))]  # 2, indices
            freq_indices = freq_indices[:, :int(freq_indices.size(1) * self.param_reduction)]  # 2, indices
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]
            weight_rfft = weight_rfft.reshape(-1, 2)[None,].repeat(self.param_ratio, 1, 1) / (
                        min(self.out_channels, self.in_channels) // 2)
        else:
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None,].repeat(self.param_ratio, 1,
                                                                                                  1, 1) / (
                                      min(self.out_channels, self.in_channels) // 2)  # param_ratio, d1, d2, k*k, 2

        if convert_param:
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)
            del self.weight
        else:
            if self.linear_mode:
                assert self.kernel_size[0] == 1 and self.kernel_size[1] == 1
                self.weight = torch.nn.Parameter(self.weight.squeeze(), requires_grad=True)
        indices = []
        for i in range(self.param_ratio):
            indices.append(freq_indices.reshape(2, self.kernel_num,
                                                -1))  # paramratio, 2, kernel_num, d1 * k1 * (d2 * k2 // 2 + 1) // kernel_num
        self.register_buffer('indices', torch.stack(indices, dim=0), persistent=False)

    def get_FDW(self, ):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1)).contiguous()  # d1 * k1, d2 * k2 // 2 + 1
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None,].repeat(self.param_ratio, 1, 1,
                                                                                              1) / (
                                  min(self.out_channels, self.in_channels) // 2)  # param_ratio, d1, d2, k*k, 2
        return weight_rfft

    def forward(self, x):
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[
            0] not in self.use_fdconv_if_k_in:
            return super().forward(x)
        global_x = F.adaptive_avg_pool2d(x, 1)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)
        if self.use_ksm_local:
            # global_x_std = torch.std(x, dim=(-1, -2), keepdim=True)
            hr_att_logit = self.KSM_Local(global_x)  # b, kn, cin, cout * ratio, k1*k2,
            hr_att_logit = hr_att_logit.reshape(x.size(0), 1, self.in_channels, self.out_channels, self.kernel_size[0],
                                                self.kernel_size[1])
            # hr_att_logit = hr_att_logit + self.hr_cin_bias[None, None, :, None, None, None] + self.hr_cout_bias[None, None, None, :, None, None] + self.hr_spatial_bias[None, None, None, None, :, :]
            hr_att_logit = hr_att_logit.permute(0, 1, 3, 2, 4, 5)
            if self.ksm_local_act == 'sigmoid':
                hr_att = hr_att_logit.sigmoid() * self.att_multi
            elif self.ksm_local_act == 'tanh':
                hr_att = 1 + hr_att_logit.tanh()
            else:
                raise NotImplementedError
        else:
            hr_att = 1
        b = x.size(0)
        batch_size, in_planes, height, width = x.size()
        DFT_map = torch.zeros(
            (b, self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1] // 2 + 1, 2),
            device=x.device)
        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)
        if hasattr(self, 'dft_weight'):
            dft_weight = self.dft_weight
        else:
            dft_weight = self.get_FDW()
            # print('get_FDW')

        # _t0 = time.perf_counter()
        for i in range(self.param_ratio):
            # print(i)
            # print(DFT_map.device)
            indices = self.indices[i]
            if self.param_reduction < 1:
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None]
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)
            else:
                w = dft_weight[i][indices[0, :, :], indices[1, :, :]][None] * self.alpha  # 1, kernel_num, -1, 2
                # print(w.shape)
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)
                pass
        # print(time.perf_counter() - _t0)
        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(DFT_map), dim=(1, 2)).reshape(batch_size, 1,
                                                                                                self.out_channels,
                                                                                                self.kernel_size[0],
                                                                                                self.in_channels,
                                                                                                self.kernel_size[1])
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5)
        # print(spatial_attention, channel_attention, filter_attention)
        if hasattr(self, 'FBM'):
            x = self.FBM(x)
            # x = self.FBM(x, self.channel_comp(x))

        if self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] < (
                in_planes + self.out_channels) * height * width:
            # print(channel_attention.shape, filter_attention.shape, hr_att.shape)
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
            # aggregate_weight = spatial_attention * channel_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            # print(aggregate_weight.abs().max())
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            if isinstance(filter_attention, float):
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2),
                                     output.size(-1))  # * filter_attention.reshape(b, -1, 1, 1)
        else:
            aggregate_weight = spatial_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            if not isinstance(channel_attention, float):
                x = x * channel_attention.view(b, -1, 1, 1)
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            # if isinstance(filter_attention, torch.FloatTensor):
            if isinstance(filter_attention, float):
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2),
                                     output.size(-1)) * filter_attention.view(b, -1, 1, 1)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output

    def profile_module(
            self, input: Tensor, *args, **kwargs
    ):
        # TODO: to edit it
        b_sz, c, h, w = input.shape
        seq_len = h * w

        # FFT iFFT
        p_ff, m_ff = 0, 5 * b_sz * seq_len * int(math.log(seq_len)) * c
        # others
        # params = macs = sum([p.numel() for p in self.parameters()])
        params = macs = self.hidden_size * self.hidden_size_factor * self.hidden_size * 2 * 2 // self.num_blocks
        # // 2 min n become half after fft
        macs = macs * b_sz * seq_len

        # return input, params, macs
        return input, params, macs + m_ff


if __name__ == '__main__':
    x = torch.rand(4, 128, 64, 64) * 1
    # m = ODPEConv2d(in_channels=128, out_channels=128, kernel_num=8, kernel_size=3, padding=1, mirror_weight=False, weight_residual=False, use_rfft=True)
    # m = ODPEAdaptConv2d(in_channels=128, out_channels=64, kernel_num=8, kernel_size=3, padding=1, mirror_weight=False, weight_residual=False, use_rfft=True, bias=True, param_ratio=4, omni_only_kernel_att=False, use_hr_att=False, att_grid=1, stride=2, spatial_freq_decompose=False)
    m = FDConv(in_channels=128, out_channels=64, kernel_num=8, kernel_size=3, padding=1, bias=True)
    # m2 = DFT_Att(n=128)
    print(m)
    # m.convert2dftweight()
    y = m(x)
    print(y.shape)
    pass
class FDConv_BN(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = FDConv(in_channels=c1, out_channels=c2, kernel_size=k, stride=s,
                           padding=autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))





class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DSConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, k, s=1, act=True):
        super().__init__()
        self.dconv = Conv(in_channels, in_channels, k=k,s=s, g=in_channels, act=act)
        self.pconv = Conv(in_channels, out_channels, k=1,s=1, g=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)
class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, act)
        self.cv2 = Conv(c_, c_, 5, 1, 2, c_, d, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1), (x_LL, x_HL, x_LH, x_HH)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / 4), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    device = x.device
    h = torch.zeros([out_batch, out_channel, out_height, out_width], device=device).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)
class WinvIWT(nn.Module):
    def __init__(self):
        super(WinvIWT, self).__init__()
        self.requires_grad = False

    def forward(self, LL, HH):
        return iwt_init(torch.cat([LL, HH], dim=1))
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return self.identity(x) + resi
class WInvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, d=1, clamp=0.8):
        super(WInvBlock, self).__init__()
        self.channel_num = channel_num
        self.channel_split_num = channel_split_num
        self.iwt = WinvIWT()
        self.split_len1 = channel_split_num
        self.split_len2 = 3 * channel_split_num
        self.P1 = HinResBlock(self.split_len1, self.split_len2)
        self.U1 = HinResBlock(self.split_len2, self.split_len1)
        self.P2 = HinResBlock(self.split_len1, self.split_len2)
        self.U2 = HinResBlock(self.split_len2, self.split_len1)
        self.flow_permutation = self.permute_flow

    def permute_flow(self, z, logdet, rev):
        return self.invconv(z, logdet, rev)
    def forward(self, x_L, x_HL, x_LH, x_HH):

        low = x_L
        high = torch.cat([x_HL, x_LH, x_HH], dim=1)
        p1 = self.P1(low) - high
        u1 = low + self.U1(p1)
        phres = self.P2(u1) - p1
        u_res = self.U2(phres) + u1
        LL = u_res
        H = phres
        return self.iwt(LL, H)
class Fusion(nn.Module):
    def __init__(self, channel, block_num=2):
        super(Fusion, self).__init__()
        self.channel = channel
        self.dwt = DWT()
        self.high_down = nn.Conv2d(channel * 3, channel, 1, 1)
        self.high_up = nn.Conv2d(channel, channel * 3, 1, 1)
        self.low_fusion=nn.Conv2d(channel,channel,3,1)
        self.high_fusion = nn.Conv2d(channel, channel, 3, 1)

    def forward(self, fea):
        _, (x_L, x_HL, x_LH, x_HH) = self.dwt(fea)
        x_H = self.high_down(torch.cat([x_HL, x_LH, x_HH], dim=1))
        B, C, H,W = x_L.shape
        L_first_half = x_L[:,:C // 2 ,:,:]
        H_first_half = x_H[:,:C // 2 ,:,:]
        L_swap = torch.cat([H_first_half, x_L[:,C // 2: ,:,: ]], dim=1)
        H_swap = torch.cat([L_first_half, x_H[:,C // 2: ,:,: ]], dim=1)
        L_swap=self.low_fusion(L_swap)
        H_swap=self.high_fusion(H_swap)
        H_swap = self.high_up(H_swap)
        L_swap = F.adaptive_avg_pool2d(L_swap, (1, 1))
        H_swap = F.adaptive_avg_pool2d(H_swap, (1, 1))
        x_L = x_L * L_swap
        x_HL = x_HL * H_swap[:, :C, :, :]
        x_LH = x_LH * H_swap[:, C:2 * C, :, :]
        x_HH = x_HH * H_swap[:, 2 * C:, :, :]
        return x_L, x_HL, x_LH, x_HH

class twostepfusion(nn.Module):
    def __init__(self, channel):
        super(twostepfusion, self).__init__()
        self.coarse_fusion = Fusion(channel)
        self.fine_fusion = WInvBlock(channel, channel)

    def forward(self, fea):
        x_L, x_HL, x_LH, x_HH = self.coarse_fusion(fea)
        fea = self.fine_fusion(x_L, x_HL, x_LH, x_HH)

        return fea


class C2f_xiaobo(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = twostepfusion(self.c)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(torch.unsqueeze(self.m(y[-1]), dim=0))
        return self.cv2(torch.cat(y, 1))



class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class CRU(nn.Module):

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn, m=-0.8):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            ScConv(in_chn, in_chn),
            torch.nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Conv2d(in_chn, in_chn, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x3 = self.conv3(x + x1)
        x4=x3+x1
        y1 = self.conv1(y)
        y3 = self.conv3(y + y1)
        y4=y3+y1
        chayi=x4-y4
        chayi=self.conv6(chayi)
        chayi_zhi=self.sigmoid(chayi)
        y=chayi_zhi*(x4-y4)+y4
        y=self.conv_out(y)
        return y


class DFM(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True, m=-0.8):
        super(DFM, self).__init__()
        dim_in=dim_in//2
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in // 2, kernel_size=1, padding=0),
                torch.nn.BatchNorm2d(dim_in // 2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in // 2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)

    def forward(self, x):
        x1, x2 = x[0], x[1]
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        return x_add



# class DFM(nn.Module):
#     def __init__(self, inchannels,outchannels,dummy=1):
#         super(DFM, self).__init__()
#         self.dfm = DF_Module(channels, channels)
#
#     def forward(self, x):
#         a, b = x[0], x[1]
#         print(f"x[0] shape: {x[0].shape}, x[1] shape: {x[1].shape}")
#         xo = self.dfm(a, b)
#         return xo

class SNI(nn.Module):
    '''
    https://github.com/AlanLi1997/rethinking-fpn
    soft nearest neighbor interpolation for up-sampling
    secondary features aligned
    '''
    def __init__(self, c1=0, c2=0, up_f=2):
        super(SNI, self).__init__()
        self.us = nn.Upsample(None, up_f, 'nearest')
        self.alpha = 1/(up_f**2)

    def forward(self, x):
        return self.alpha*self.us(x)



def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def dsconv_3x3(in_channels, out_channels):
    assert in_channels % 1 == 0, f"Invalid in_channels={in_channels} for dsconv"
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),  # depthwise
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),  # pointwise
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )



def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class TFF(nn.Module):
    def __init__(self, in_channel, out_channel,inplace=False):
        super(TFF, self).__init__()
        in_channel = in_channel // 2
        out_channel= out_channel // 2
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, in_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xA, xB = x[0], x[1]
        x_diff = xA - xB  # 通过相减获得粗略的变化表示: (B,C,H,W)

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1)) #将变化特征与xA拼接,通过DWConv提取特征: (B,C,H,W)-cat-(B,C,H,W)-->(B,2C,H,W);  (B,2C,H,W)-catconvA-->(B,C,H,W)
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1)) #将变化特征与xB拼接,通过DWConv提取特征: (B,C,H,W)-cat-(B,C,H,W)-->(B,2C,H,W);  (B,2C,H,W)-catconvB-->(B,C,H,W)

        A_weight = self.sigmoid(self.convA(x_diffA)) # 通过卷积映射到1个通道,生成空间描述符,然后通过sigmoid生成权重: (B,C,H,W)-convA->(B,1,H,W)
        B_weight = self.sigmoid(self.convB(x_diffB)) # 通过卷积映射到1个通道,生成空间描述符,然后通过sigmoid生成权重: (B,C,H,W)-convB->(B,1,H,W)

        xA = A_weight * xA # 使用生成的权重A_weight调整对应输入xA: (B,1,H,W) * (B,C,H,W) == (B,C,H,W)
        xB = B_weight * xB # 使用生成的权重B_weight调整对应输入xB: (B,1,H,W) * (B,C,H,W) == (B,C,H,W)

        x = self.catconv(torch.cat([xA, xB], dim=1)) # 两个特征拼接,然后恢复与输入相同的shape: (B,C,H,W)-cat-(B,C,H,W)-->(B,2C,H,W); (B,2C,H,W)--catconv->(B,C,H,W)

        return x

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #(B,C,H,W)
        b, c, h, w = x.size()

        ### 坐标注意力模块  ###
        group_x = x.reshape(b * self.groups, -1, h, w)  # 在通道方向上将输入分为G组: (B,C,H,W)-->(B*G,C/G,H,W)
        x_h = self.pool_h(group_x) # 使用全局平均池化压缩水平空间方向: (B*G,C/G,H,W)-->(B*G,C/G,H,1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) # 使用全局平均池化压缩垂直空间方向: (B*G,C/G,H,W)-->(B*G,C/G,1,W)-->(B*G,C/G,W,1)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))# 将水平方向和垂直方向的全局特征进行拼接: (B*G,C/G,H+W,1), 然后通过1×1Conv进行变换,来编码空间水平和垂直方向上的特征
        x_h, x_w = torch.split(hw, [h, w], dim=2) # 沿着空间方向将其分割为两个矩阵表示: x_h:(B*G,C/G,H,1); x_w:(B*G,C/G,W,1)

        ### 1×1分支和3×3分支的输出表示  ###
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()) # 通过水平方向权重和垂直方向权重调整输入,得到1×1分支的输出: (B*G,C/G,H,W) * (B*G,C/G,H,1) * (B*G,C/G,1,W)=(B*G,C/G,H,W)
        x2 = self.conv3x3(group_x) # 通过3×3卷积提取局部上下文信息: (B*G,C/G,H,W)-->(B*G,C/G,H,W)

        ### 跨空间学习 ###
        ## 1×1分支生成通道描述符来调整3×3分支的输出
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # 对1×1分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符: (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将3×3分支的输出进行变换,以便与1×1分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y1 = torch.matmul(x11, x12) # (B*G,1,C/G) @ (B*G,C/G,H*W) = (B*G,1,H*W)

        ## 3×3分支生成通道描述符来调整1×1分支的输出
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # 对3×3分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符: (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw  # 将1×1分支的输出进行变换,以便与3×3分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y2 = torch.matmul(x21, x22)  # (B*G,1,C/G) @ (B*G,C/G,H*W) = (B*G,1,H*W)

        # 聚合两种尺度的空间位置信息, 通过sigmoid生成空间权重, 从而再次调整输入表示
        weights = (y1+y2).reshape(b * self.groups, 1, h, w)  # 将两种尺度下的空间位置信息进行聚合: (B*G,1,H*W)-->reshape-->(B*G,1,H,W)
        weights_ =  weights.sigmoid() # 通过sigmoid生成权重表示: (B*G,1,H,W)
        # 假设 B, C, G 已知
        BG, _, H, W = weights_.shape
        G=BG//b
        B = BG // G
        channels_per_group =c // G  # 每个组的通道数

        # 方法1：expand + reshape（不复制内存，效率高）
        weights_expanded = weights_.expand(BG, channels_per_group, H, W)  # (B*G, C/G, H, W)
        weights_expanded = weights_expanded.reshape(B, c, H, W)  # (B, C, H, W)


        return weights_expanded


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   # 计算通道方向平均值:(B,d,H,W)-avg->(B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # 计算通道方向最大值:(B,d,H,W)-max->(B,1,H,W)
        x = torch.cat([avg_out, max_out], dim=1) # 通道方向拼接: (B,1,H,W)-cat-(B,1,H,W)-->(B,2,H,W);
        x = self.conv1(x) # 降维: (B,2,H,W)-->(B,1,H,W)
        return self.sigmoid(x) # 通过sigmoid生成权重表示:(B,1,H,W)

class SKAttention(nn.Module):

    def __init__(self, channel,kernels=[1,3,5,7],reduction=8,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        # 有几个卷积核,就有几个尺度, 每个尺度对应的卷积层由Conv-bn-relu实现
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=channel)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        # 将全局向量降维
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):
        # (B, C, H, W)
        B, C, H, W = x.size()
        # 存放多尺度的输出
        conv_outs=[]
        # Split: 执行K个尺度对应的卷积操作
        for conv in self.convs:
            scale = conv(x)  #每一个尺度的输出shape都是: (B, C, H, W),是因为使用了padding操作
            conv_outs.append(scale)
        feats=torch.stack(conv_outs,0) # 将K个尺度的输出在第0个维度上拼接: (K,B,C,H,W)

        # Fuse: 首先将多尺度的信息进行相加,sum()默认在第一个维度进行求和
        U=sum(conv_outs) #(K,B,C,H,W)-->(B,C,H,W)
        # 全局平均池化操作: (B,C,H,W)-->mean-->(B,C,H)-->mean-->(B,C)  【mean操作等价于全局平均池化的操作】
        S=U.mean(-1).mean(-1)
        # 降低通道数,提高计算效率: (B,C)-->(B,d)
        Z=self.fc(S)

        # 将紧凑特征Z通过K个全连接层得到K个尺度对应的通道描述符表示, 然后基于K个通道描述符计算注意力权重
        weights=[]
        for fc in self.fcs:
            weight=fc(Z) #恢复预输入相同的通道数: (B,d)-->(B,C)
            weights.append(weight.view(B,C,1,1)) # (B,C)-->(B,C,1,1)
        scale_weight=torch.stack(weights,0) #将K个通道描述符在0个维度上拼接: (K,B,C,1,1)
        scale_weight=self.softmax(scale_weight) #在第0个维度上执行softmax,获得每个尺度的权重: (K,B,C,1,1)

        # Select
        V=(scale_weight*feats).sum(0) # 将每个尺度的权重与对应的特征进行加权求和,第一步是加权，第二步是求和：(K,B,C,1,1) * (K,B,C,H,W) = (K,B,C,H,W)-->sum-->(B,C,H,W)
        return V


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        self.SKfusion=SKAttention(channel=in_channels, reduction=8)
        #self.spatial_attention = SpatialAttentionModule()
        self.spatial_attention = EMA(channels=in_channels)
        # self.up = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        # self.down_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):


        x_fused_c = self.SKfusion(x)

        # x_fused_c = x_fused * self.channel_attention(x_fused)
        # x_3x3 = self.conv_3x3(x_fused)
        # x_5x5 = self.conv_5x5(x_fused)
        # x_7x7 = self.conv_7x7(x_fused)
        #x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_c * self.spatial_attention(x)

        x_out = x_fused_s + x

        return x_out


class gatedFusion(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super(gatedFusion, self).__init__()
        dim = dim//2
        # 使用1x1卷积代替全连接层，保持空间维度
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        # 输入形状: [B, C, H, W]
        x1,x2=x[0],x[1]
        x11 = self.conv1(x1)  # 形状保持 [B, C, H, W]
        x22 = self.conv2(x2)  # 形状保持 [B, C, H, W]

        # 通过门控单元生成权重
        z = torch.sigmoid(x11 + x22)  # 形状 [B, C, H, W]

        # 加权融合
        out = z * x1 + (1 - z) * x2  # 形状 [B, C, H, W]

        return out

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GSConvE3(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    '''
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 4
        # 使用关键字参数，避免参数顺序混淆
        self.cv1 = Conv(c1=c1, c2=c_, k=k, s=s, p=None, g=g, d=d, act=act)

        # 这里每个子段用两个 Conv，第二个是 1x1 变换，明确传 act
        self.cv2 = nn.Sequential(
            Conv(c1=c_, c2=c_, k=3, s=1, p=1, g=c_, d=d, act=act),
            Conv(c1=c_, c2=c_, k=1, s=1, p=None, g=1, d=1, act=act)
        )
        self.cv3 = nn.Sequential(
            Conv(c1=c_, c2=c_, k=3, s=1, p=1, g=c_, d=d, act=act),
            Conv(c1=c_, c2=c_, k=1, s=1, p=None, g=1, d=1, act=act)
        )
        self.cv4 = nn.Sequential(
            Conv(c1=c_, c2=c_, k=3, s=1, p=1, g=c_, d=d, act=act),
            Conv(c1=c_, c2=c_, k=1, s=1, p=None, g=1, d=1, act=act)
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # 拼接： [B, 4*C_, H, W]
        y = torch.cat((x1, x2, x3, x4), dim=1)

        B, C, H, W = y.shape
        group = 4
        if C % group != 0:
            raise ValueError(f"channels({C}) must be divisible by group({group})")
        c_per = C // group  # 每块通道数

        # reshape -> permute -> reshape 实现 interleave shuffle
        y = y.view(B, group, c_per, H, W)         # [B, 4, c_, H, W]
        y = y.permute(0, 2, 1, 3, 4).reshape(B, C, H, W)  # [B, C, H, W] with interleaved channels

        return y
class GSConvE2(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    '''
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.Conv2d(c_, c_, 1, 1,bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
class GSConvE(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    '''
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.Conv2d(c_, c_, 3, 1,1,bias=False),
            nn.BatchNorm2d(c_),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

# class gatedFusion2(nn.Module):
#     def __init__(self, dim,dim2, *args, **kwargs):
#         super(gatedFusion2, self).__init__()
#         dim=dim2
#         dim2=dim2*2
#
#
#         # 使用1x1卷积代替全连接层，保持空间维度
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
#         self.conv2 = nn.Conv2d(dim2, dim, kernel_size=1, bias=True)
#         print(f"dim: {dim}, dim2: {dim2}")
#
#     def forward(self, x):
#         # 输入形状: [B, C, H, W]
#         x1,x2=x[0],x[1]
#         print(f"x[0] shape: {x[0].shape}, x[1] shape: {x[1].shape}")
#         x11 = self.conv1(x1)  # 形状保持 [B, C, H, W]
#         x22 = self.conv2(x2)  # 形状保持 [B, C, H, W]
#         x23=self.conv1(x22)
#         print(f"x[0] shape: {x11.shape}, x[1] shape: {x22.shape}")
#
#         # 通过门控单元生成权重
#         z = torch.sigmoid(x11 + x23)  # 形状 [B, C, H, W]
#
#         # 加权融合
#         out = z * x1 + (1 - z) * x22  # 形状 [B, C, H, W]
#
#         return out

class gatedFusion2(nn.Module):
    def __init__(self, dim,dim2, *args, **kwargs):
        super(gatedFusion2, self).__init__()
        dim=dim2//2
        dim2=dim2


        # 使用1x1卷积代替全连接层，保持空间维度
        self.conv1 = nn.Conv2d(dim, dim2, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dim2, dim2, kernel_size=1, bias=True)
        #print(f"dim: {dim}, dim2: {dim2}")

    def forward(self, x):
        # 输入形状: [B, C, H, W]
        x1,x2=x[0],x[1]
        #print(f"x[0] shape: {x[0].shape}, x[1] shape: {x[1].shape}")
        x11 = self.conv1(x1)  # 形状保持 [B, C, H, W]
        x22 = self.conv2(x2)  # 形状保持 [B, C, H, W]
        x13=self.conv2(x11)
        #print(f"x[0] shape: {x11.shape}, x[1] shape: {x22.shape}")

        # 通过门控单元生成权重
        z = torch.sigmoid(x13 + x22)  # 形状 [B, C, H, W]

        # 加权融合
        out = z * x11 + (1 - z) * x2  # 形状 [B, C, H, W]

        return out

class gatedFusion3(nn.Module):
    def __init__(self, dim,dim2, *args, **kwargs):
        super(gatedFusion3, self).__init__()
        dim=dim2//2
        dim2=dim2


        # 使用1x1卷积代替全连接层，保持空间维度
        self.conv1 = nn.Conv2d(dim, dim2, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dim2, dim2, kernel_size=1, bias=True)
        #print(f"dim: {dim}, dim2: {dim2}")

    def forward(self, x):
        # 输入形状: [B, C, H, W]
        x1,x2=x[0],x[1]
        #print(f"x[0] shape: {x1.shape}, x[1] shape: {x2.shape}")
        x22 = self.conv1(x2)  # 形状保持 [B, C, H, W]
        x11 = self.conv2(x1)  # 形状保持 [B, C, H, W]
        x23=self.conv2(x22)
        #print(f"x[0] shape: {x11.shape}, x[1] shape: {x22.shape}")

        # 通过门控单元生成权重
        z = torch.sigmoid(x11 + x23)  # 形状 [B, C, H, W]

        # 加权融合
        out = z * x1 + (1 - z) * x22  # 形状 [B, C, H, W]

        return out


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
            self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = FDConv_BN(c1, c_, k[0], 1)
        self.cv2 = FDConv_BN(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class F2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        #print(f"c1: {c1}, c2: {c2}")
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



# import torch
# from torch.autograd import Function
# import triton
# import triton.language as tl
# from torch.amp import custom_fwd, custom_bwd
# import math
#
# def _grid(numel: int, bs: int) -> tuple:
#     return (triton.cdiv(numel, bs),)
#
# @triton.jit
# def _idx(i, n: int, c: int, h: int, w: int):
#     ni = i // (c * h * w)
#     ci = (i // (h * w)) % c
#     hi = (i // w) % h
#     wi = i % w
#     m = i < (n * c * h * w)
#     return ni, ci, hi, wi, m
#
# @triton.jit
# def ska_fwd(
#     x_ptr, w_ptr, o_ptr,
#     n, ic, h, w, ks, pad, wc,
#     BS: tl.constexpr,
#     CT: tl.constexpr, AT: tl.constexpr
# ):
#     pid = tl.program_id(0)
#     start = pid * BS
#     offs = start + tl.arange(0, BS)
#
#     ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
#     val = tl.zeros((BS,), dtype=AT)
#
#     for kh in range(ks):
#         hin = hi - pad + kh
#         hb = (hin >= 0) & (hin < h)
#         for kw in range(ks):
#             win = wi - pad + kw
#             b = hb & (win >= 0) & (win < w)
#
#             x_off = ((ni * ic + ci) * h + hin) * w + win
#             w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi
#
#             x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
#             w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
#             val += tl.where(b & m, x_val * w_val, 0.0).to(AT)
#
#     tl.store(o_ptr + offs, val.to(CT), mask=m)
#
# @triton.jit
# def ska_bwd_x(
#     go_ptr, w_ptr, gi_ptr,
#     n, ic, h, w, ks, pad, wc,
#     BS: tl.constexpr,
#     CT: tl.constexpr, AT: tl.constexpr
# ):
#     pid = tl.program_id(0)
#     start = pid * BS
#     offs = start + tl.arange(0, BS)
#
#     ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
#     val = tl.zeros((BS,), dtype=AT)
#
#     for kh in range(ks):
#         ho = hi + pad - kh
#         hb = (ho >= 0) & (ho < h)
#         for kw in range(ks):
#             wo = wi + pad - kw
#             b = hb & (wo >= 0) & (wo < w)
#
#             go_off = ((ni * ic + ci) * h + ho) * w + wo
#             w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + ho * w + wo
#
#             go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
#             w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
#             val += tl.where(b & m, go_val * w_val, 0.0).to(AT)
#
#     tl.store(gi_ptr + offs, val.to(CT), mask=m)
#
# @triton.jit
# def ska_bwd_w(
#     go_ptr, x_ptr, gw_ptr,
#     n, wc, h, w, ic, ks, pad,
#     BS: tl.constexpr,
#     CT: tl.constexpr, AT: tl.constexpr
# ):
#     pid = tl.program_id(0)
#     start = pid * BS
#     offs = start + tl.arange(0, BS)
#
#     ni, ci, hi, wi, m = _idx(offs, n, wc, h, w)
#
#     for kh in range(ks):
#         hin = hi - pad + kh
#         hb = (hin >= 0) & (hin < h)
#         for kw in range(ks):
#             win = wi - pad + kw
#             b = hb & (win >= 0) & (win < w)
#             w_off = ((ni * wc + ci) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi
#
#             val = tl.zeros((BS,), dtype=AT)
#             steps = (ic - ci + wc - 1) // wc
#             for s in range(tl.max(steps, axis=0)):
#                 cc = ci + s * wc
#                 cm = (cc < ic) & m & b
#
#                 x_off = ((ni * ic + cc) * h + hin) * w + win
#                 go_off = ((ni * ic + cc) * h + hi) * w + wi
#
#                 x_val = tl.load(x_ptr + x_off, mask=cm, other=0.0).to(CT)
#                 go_val = tl.load(go_ptr + go_off, mask=cm, other=0.0).to(CT)
#                 val += tl.where(cm, x_val * go_val, 0.0).to(AT)
#
#             tl.store(gw_ptr + w_off, val.to(CT), mask=m)
#
# class SkaFn(Function):
#     @staticmethod
#     @custom_fwd(device_type='cuda')
#     def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
#         ks = int(math.sqrt(w.shape[2]))
#         pad = (ks - 1) // 2
#         ctx.ks, ctx.pad = ks, pad
#         n, ic, h, width = x.shape
#         wc = w.shape[1]
#         o = torch.empty(n, ic, h, width, device=x.device, dtype=x.dtype)
#         numel = o.numel()
#
#         x = x.contiguous()
#         w = w.contiguous()
#
#         grid = lambda meta: _grid(numel, meta["BS"])
#
#         ct = tl.float16 if x.dtype == torch.float16 else (tl.float32 if x.dtype == torch.float32 else tl.float64)
#         at = tl.float32 if x.dtype == torch.float16 else ct
#
#         ska_fwd[grid](x, w, o, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)
#
#         ctx.save_for_backward(x, w)
#         ctx.ct, ctx.at = ct, at
#         return o
#
#     @staticmethod
#     @custom_bwd(device_type='cuda')
#     def backward(ctx, go: torch.Tensor) -> tuple:
#         ks, pad = ctx.ks, ctx.pad
#         x, w = ctx.saved_tensors
#         n, ic, h, width = x.shape
#         wc = w.shape[1]
#
#         go = go.contiguous()
#         gx = gw = None
#         ct, at = ctx.ct, ctx.at
#
#         if ctx.needs_input_grad[0]:
#             gx = torch.empty_like(x)
#             numel = gx.numel()
#             ska_bwd_x[lambda meta: _grid(numel, meta["BS"])](go, w, gx, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)
#
#         if ctx.needs_input_grad[1]:
#             gw = torch.empty_like(w)
#             numel = gw.numel() // w.shape[2]
#             ska_bwd_w[lambda meta: _grid(numel, meta["BS"])](go, x, gw, n, wc, h, width, ic, ks, pad, BS=1024, CT=ct, AT=at)
#
#         return gx, gw, None, None
#
# class SKA(torch.nn.Module):
#     def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
#         return SkaFn.apply(x, w) # type: ignore
#
#
# import torch
# import itertools
#
# from timm.models.vision_transformer import trunc_normal_
# from timm.models.layers import SqueezeExcite
# from timm.models.registry import register_model
# #from .ska import SKA
#
# from timm.models.helpers import build_model_with_cfg
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
#
#
# class Conv2d_BN(torch.nn.Sequential):
#     def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
#                  groups=1, bn_weight_init=1):
#         super().__init__()
#         self.add_module('c', torch.nn.Conv2d(
#             a, b, ks, stride, pad, dilation, groups, bias=False))
#         self.add_module('bn', torch.nn.BatchNorm2d(b))
#         torch.nn.init.constant_(self.bn.weight, bn_weight_init)
#         torch.nn.init.constant_(self.bn.bias, 0)
#
#     @torch.no_grad()
#     def fuse(self):
#         c, bn = self._modules.values()
#         w = bn.weight / (bn.running_var + bn.eps) ** 0.5
#         w = c.weight * w[:, None, None, None]
#         b = bn.bias - bn.running_mean * bn.weight / \
#             (bn.running_var + bn.eps) ** 0.5
#         m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
#             0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
#                             groups=self.c.groups,
#                             device=c.weight.device)
#         m.weight.data.copy_(w)
#         m.bias.data.copy_(b)
#         return m
#
#
# class BN_Linear(torch.nn.Sequential):
#     def __init__(self, a, b, bias=True, std=0.02):
#         super().__init__()
#         self.add_module('bn', torch.nn.BatchNorm1d(a))
#         self.add_module('l', torch.nn.Linear(a, b, bias=bias))
#         trunc_normal_(self.l.weight, std=std)
#         if bias:
#             torch.nn.init.constant_(self.l.bias, 0)
#
#     @torch.no_grad()
#     def fuse(self):
#         bn, l = self._modules.values()
#         w = bn.weight / (bn.running_var + bn.eps) ** 0.5
#         b = bn.bias - self.bn.running_mean * \
#             self.bn.weight / (bn.running_var + bn.eps) ** 0.5
#         w = l.weight * w[None, :]
#         if l.bias is None:
#             b = b @ self.l.weight.T
#         else:
#             b = (l.weight @ b[:, None]).view(-1) + self.l.bias
#         m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
#         m.weight.data.copy_(w)
#         m.bias.data.copy_(b)
#         return m
#
#
# class Residual(torch.nn.Module):
#     def __init__(self, m, drop=0.):
#         super().__init__()
#         self.m = m
#         self.drop = drop
#
#     def forward(self, x):
#         if self.training and self.drop > 0:
#             return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
#                                               device=x.device).ge_(self.drop).div(1 - self.drop).detach()
#         else:
#             return x + self.m(x)
#
#
# class FFN(torch.nn.Module):
#     def __init__(self, ed, h):
#         super().__init__()
#         self.pw1 = Conv2d_BN(ed, h)
#         self.act = torch.nn.ReLU()
#         self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)
#
#     def forward(self, x):
#         x = self.pw2(self.act(self.pw1(x)))
#         return x
#
#
# class Attention(torch.nn.Module):
#     def __init__(self, dim, key_dim, num_heads=8,
#                  attn_ratio=4,
#                  resolution=14):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = key_dim ** -0.5
#         self.key_dim = key_dim
#         self.nh_kd = nh_kd = key_dim * num_heads
#         self.d = int(attn_ratio * key_dim)
#         self.dh = int(attn_ratio * key_dim) * num_heads
#         self.attn_ratio = attn_ratio
#         h = self.dh + nh_kd * 2
#         self.qkv = Conv2d_BN(dim, h, ks=1)
#         self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
#             self.dh, dim, bn_weight_init=0))
#         self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
#         points = list(itertools.product(range(resolution), range(resolution)))
#         N = len(points)
#         attention_offsets = {}
#         idxs = []
#         for p1 in points:
#             for p2 in points:
#                 offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
#                 if offset not in attention_offsets:
#                     attention_offsets[offset] = len(attention_offsets)
#                 idxs.append(attention_offsets[offset])
#         self.attention_biases = torch.nn.Parameter(
#             torch.zeros(num_heads, len(attention_offsets)))
#         self.register_buffer('attention_bias_idxs',
#                              torch.LongTensor(idxs).view(N, N))
#
#     @torch.no_grad()
#     def train(self, mode=True):
#         super().train(mode)
#         if mode and hasattr(self, 'ab'):
#             del self.ab
#         else:
#             self.ab = self.attention_biases[:, self.attention_bias_idxs]
#
#     def forward(self, x):
#         B, _, H, W = x.shape
#         N = H * W
#         qkv = self.qkv(x)
#         q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
#         q = self.dw(q)
#         q, k, v = q.view(B, self.num_heads, -1, N), k.view(B, self.num_heads, -1, N), v.view(B, self.num_heads, -1, N)
#         attn = (
#                 (q.transpose(-2, -1) @ k) * self.scale
#                 +
#                 (self.attention_biases[:, self.attention_bias_idxs]
#                  if self.training else self.ab)
#         )
#         attn = attn.softmax(dim=-1)
#         x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
#         x = self.proj(x)
#         return x
#
#
# class RepVGGDW(torch.nn.Module):
#     def __init__(self, ed) -> None:
#         super().__init__()
#         self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
#         self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
#         self.dim = ed
#
#     def forward(self, x):
#         return self.conv(x) + self.conv1(x) + x
#
#     @torch.no_grad()
#     def fuse(self):
#         conv = self.conv.fuse()
#         conv1 = self.conv1.fuse()
#
#         conv_w = conv.weight
#         conv_b = conv.bias
#         conv1_w = conv1.weight
#         conv1_b = conv1.bias
#
#         conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])
#
#         identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
#                                            [1, 1, 1, 1])
#
#         final_conv_w = conv_w + conv1_w + identity
#         final_conv_b = conv_b + conv1_b
#
#         conv.weight.data.copy_(final_conv_w)
#         conv.bias.data.copy_(final_conv_b)
#         return conv
#
#
# import torch.nn as nn
#
#
# class LKP(nn.Module):
#     def __init__(self, dim, lks, sks, groups):
#         super().__init__()
#         self.cv1 = Conv2d_BN(dim, dim // 2)
#         self.act = nn.ReLU()
#         self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
#         self.cv3 = Conv2d_BN(dim // 2, dim // 2)
#         self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
#         self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
#
#         self.sks = sks
#         self.groups = groups
#         self.dim = dim
#
#     def forward(self, x):
#         x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
#         w = self.norm(self.cv4(x))
#         b, _, h, width = w.size()
#         w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
#         return w
#
#
# class LSConv(nn.Module):
#     def __init__(self, dim):
#         super(LSConv, self).__init__()
#         self.lkp = LKP(dim, lks=7, sks=3, groups=8)
#         self.ska = SKA()
#         self.bn = nn.BatchNorm2d(dim)
#
#     def forward(self, x):
#         return self.bn(self.ska(x, self.lkp(x))) + x
#
#
# class Block(torch.nn.Module):
#     def __init__(self,
#                  ed, kd, nh=8,
#                  ar=4,
#                  resolution=14,
#                  stage=-1, depth=-1):
#         super().__init__()
#
#         if depth % 2 == 0:
#             self.mixer = RepVGGDW(ed)
#             self.se = SqueezeExcite(ed, 0.25)
#         else:
#             self.se = torch.nn.Identity()
#             if stage == 3:
#                 self.mixer = Residual(Attention(ed, kd, nh, ar, resolution=resolution))
#             else:
#                 self.mixer = LSConv(ed)
#
#         self.ffn = Residual(FFN(ed, int(ed * 2)))
#
#     def forward(self, x):
#         return self.ffn(self.se(self.mixer(x)))
#
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class CAPABlock(nn.Module):
    def __init__(self,dim):
        super(CAPABlock, self).__init__()
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.calayer(x)
        res=self.palayer(res)
        res += x
        return res



class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
class PSAplus(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        # >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        # >>> input_tensor = torch.randn(1, 256, 64, 64)
        # >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
        self.w = CAPABlock(self.c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        a = self.w(a)
        return self.cv2(torch.cat((a, b), 1))


class EMA2(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA2, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #(B,C,H,W)
        b, c, h, w = x.size()

        ### 坐标注意力模块  ###
        group_x = x.reshape(b * self.groups, -1, h, w)  # 在通道方向上将输入分为G组: (B,C,H,W)-->(B*G,C/G,H,W)
        x_h = self.pool_h(group_x) # 使用全局平均池化压缩水平空间方向: (B*G,C/G,H,W)-->(B*G,C/G,H,1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) # 使用全局平均池化压缩垂直空间方向: (B*G,C/G,H,W)-->(B*G,C/G,1,W)-->(B*G,C/G,W,1)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))# 将水平方向和垂直方向的全局特征进行拼接: (B*G,C/G,H+W,1), 然后通过1×1Conv进行变换,来编码空间水平和垂直方向上的特征
        x_h, x_w = torch.split(hw, [h, w], dim=2) # 沿着空间方向将其分割为两个矩阵表示: x_h:(B*G,C/G,H,1); x_w:(B*G,C/G,W,1)

        ### 1×1分支和3×3分支的输出表示  ###
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()) # 通过水平方向权重和垂直方向权重调整输入,得到1×1分支的输出: (B*G,C/G,H,W) * (B*G,C/G,H,1) * (B*G,C/G,1,W)=(B*G,C/G,H,W)
        x2 = self.conv3x3(group_x) # 通过3×3卷积提取局部上下文信息: (B*G,C/G,H,W)-->(B*G,C/G,H,W)

        ### 跨空间学习 ###
        ## 1×1分支生成通道描述符来调整3×3分支的输出
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # 对1×1分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符: (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将3×3分支的输出进行变换,以便与1×1分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y1 = torch.matmul(x11, x12) # (B*G,1,C/G) @ (B*G,C/G,H*W) = (B*G,1,H*W)

        ## 3×3分支生成通道描述符来调整1×1分支的输出
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # 对3×3分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符: (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw  # 将1×1分支的输出进行变换,以便与3×3分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y2 = torch.matmul(x21, x22)  # (B*G,1,C/G) @ (B*G,C/G,H*W) = (B*G,1,H*W)

        # 聚合两种尺度的空间位置信息, 通过sigmoid生成空间权重, 从而再次调整输入表示
        weights = (y1+y2).reshape(b * self.groups, 1, h, w)  # 将两种尺度下的空间位置信息进行聚合: (B*G,1,H*W)-->reshape-->(B*G,1,H,W)
        weights_ =  weights.sigmoid() # 通过sigmoid生成权重表示: (B*G,1,H,W)
        out = (group_x * weights_).reshape(b, c, h, w) # 通过空间权重再次校准输入: (B*G,C/G,H,W)*(B*G,1,H,W)==(B*G,C/G,H,W)-->reshape(B,C,H,W)
        return out







class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

import torch
import torch.nn as nn
class StripConv2(nn.Module):
    def __init__(self, dim, k1, k2):   # ✅ 正确
        super().__init__()             # ✅ 正确
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim, dim, kernel_size=(k1, k2),
                                       stride=1, padding=(k1 // 2, k2 // 2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim, dim, kernel_size=(k2, k1),
                                       stride=1, padding=(k2 // 2, k1 // 2), groups=dim)
        self.conv1 = nn.Conv2d(2*dim, dim, 1)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.act(self.bn(attn))
        attn1 = self.conv_spatial1(attn)
        attn2 = self.conv_spatial2(attn)
        attn = self.conv1(torch.cat([attn1, attn2], dim=1))
        attn = torch.sigmoid(attn)
        return x * attn


class StripConv(nn.Module):
    def __init__(self, dim, k1, k2):   # ✅ 正确
        super().__init__()             # ✅ 正确
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim, dim, kernel_size=(k1, k2),
                                       stride=1, padding=(k1 // 2, k2 // 2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim, dim, kernel_size=(k2, k1),
                                       stride=1, padding=(k2 // 2, k1 // 2), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
        return x * attn


class StripModule(nn.Module):
    def __init__(self, d_model, k1, k2):  # ✅ 正确
        super().__init__()                # ✅ 正确
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripConv(d_model, k1, k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

class FusionConv2(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv2, self).__init__()
        self.cv1=Conv(in_channels,out_channels,1,1)
        self.SKfusion=SKAttention(channel=out_channels, reduction=8)
        #self.spatial_attention = SpatialAttentionModule()
        self.spatial_attention = EMA(channels=out_channels)
        # self.up = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        # self.down_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        print(f'x.shape:',x.shape)
        x=self.cv1(x)
        print(f'x.shape:', x.shape)
        x_fused_c = self.SKfusion(x)

        # x_fused_c = x_fused * self.channel_attention(x_fused)
        # x_3x3 = self.conv_3x3(x_fused)
        # x_5x5 = self.conv_5x5(x_fused)
        # x_7x7 = self.conv_7x7(x_fused)
        #x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_c * self.spatial_attention(x)
        print(f'x_fused_s:', x_fused_s.shape)
        x_out = x_fused_s + x

        return x_out

class FC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1)
        self.m=FusionConv(c2,c2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = self.cv1(x)
        return self.m(y)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)





    #
    # def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
    #     super().__init__()
    #     c_ = c2 // 2
    #     self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
    #     self.cv2 = nn.Sequential(
    #         nn.Conv2d(c_, c_, 3, 1,1,bias=False),
    #         nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
    #         nn.BatchNorm2d(c_),
    #         nn.GELU()
    #     )
    #
    # def forward(self, x):
    #     x1 = self.cv1(x)
    #     x2 = self.cv2(x1)
    #     y = torch.cat((x1, x2), dim=1)
    #     # shuffle
    #     y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
    #     y = y.permute(0, 2, 1, 3, 4)
    #     return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
# class GSConvE2(nn.Module):
#     '''
#     GSConv enhancement for representation learning: generate various receptive-fields and
#     texture-features only in one Conv module
#     https://github.com/AlanLi1997/slim-neck-by-gsconv
#     '''
#     def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
#         super().__init__()
#         c_ = c2 // 2
#         self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
#         self.cv2 = nn.Sequential(
#             nn.Conv2d(c_, c_, 3, 1,1,bias=False),
#             nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
#             nn.BatchNorm2d(c_),
#             nn.GELU()
#         )
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = self.cv2(x1)
#         y = torch.cat((x1, x2), dim=1)
#         # shuffle
#         y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
#         y = y.permute(0, 2, 1, 3, 4)
#         return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
#

# class GSConv(nn.Module):
#     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         c_ = c2 // 2
#         self.cv1 = Conv(c1, c_, k, s, p, g, d, act)
#         self.cv2 = Conv(c_, c_, 5, 1, 2, c_, d, act)
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = torch.cat((x1, self.cv2(x1)), 1)
#         # shuffle
#         y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
#         y = y.permute(0, 2, 1, 3, 4)
#         return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
# ====== START: 兼容 Ultralytics 的类实现补丁 ======
import math
import torch
import torch.nn as nn

# 假定你的项目已经定义了一个 Conv 基类（与原代码一致）
# 如果 Conv 在别的模块，请保持导入或不重复定义

class DWConv(Conv):
    # Depth-wise convolution
    # 改为接受额外参数并把 groups 设为 in_channels（更接近 depthwise）
    def __init__(self, c1, c2=None, k=1, s=1, d=1, act=True, *args, **kwargs):
        if c2 is None:
            c2 = c1
        # 深度可分卷积常用 groups = c1
        groups = c1 if isinstance(c1, int) and c1 > 0 else math.gcd(c1 or 1, c2 or 1)
        super().__init__(c1, c2, k, s, g=groups, d=d, act=act)

class GhostConv(nn.Module):
    # Ghost Convolution — 接受可变参数以兼容 parse_model
    def __init__(self, c1, c2=None, k=1, s=1, g=1, act=True, *args, **kwargs):
        super().__init__()
        if c2 is None:
            c2 = c1
        c_ = c2 // 2  # hidden channels
        # 保持原有 Conv 接口
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

class GhostBottleneck(nn.Module):
    # Ghost Bottleneck — 接受可变参数以容忍 parse_model 传入额外参数
    def __init__(self, c1, c2=None, k=3, s=1, *args, **kwargs):
        super().__init__()
        if c2 is None:
            c2 = c1
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False)
        )
        self.shortcut = (nn.Sequential(
            DWConv(c1, c1, k, s, act=False),
            Conv(c1, c2, 1, 1, act=False)
        ) if s == 2 else nn.Identity())

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class Bottleneck_C2F(nn.Module):
    # Standard bottleneck — 兼容额外参数
    def __init__(self, c1, c2=None, shortcut=True, g=1, k=(3, 3), e=0.5, *args, **kwargs):
        super().__init__()
        if c2 is None:
            c2 = c1
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5, *args, **kwargs):
        super().__init__()
        if c2 is None:
            c2 = c1
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # 使用兼容的 Bottleneck_C2F 构造
        self.m = nn.ModuleList(Bottleneck_C2F(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2fGhost(C2f):
    # 继承 C2f 并使用 GhostBottleneck
    def __init__(self, c1, c2=None, n=1, shortcut=False, g=1, e=0.5, *args, **kwargs):
        super().__init__(c1, c2, n, shortcut, g, e)
        if c2 is None:
            c2 = c1
        self.c = int(c2 * e)
        # 这里调用 GhostBottleneck(self.c, self.c)（签名已兼容）
        self.m = nn.ModuleList(GhostBottleneck(self.c, self.c) for _ in range(n))

class ECA(nn.Module):
    """ECA module — 构造函数对不同调用风格更鲁棒。"""
    def __init__(self, channel, maybe_none=None, k_size=3, *args, **kwargs):
        """
        兼容：
        - ECA(ch)
        - ECA(ch, other) （parse_model 可能传入多个位置参数，我们只取第一个）
        """
        super(ECA, self).__init__()
        # 若 channel 是列表/tuple 或者传了额外参数，确保取第一个作为通道数
        if isinstance(channel, (list, tuple)):
            channels = int(channel[0])
        else:
            channels = int(channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
# ====== END: 兼容补丁 ======
import torch
from torch import nn
from collections import OrderedDict

class SKAttention2(nn.Module):

    def __init__(self, channel,kernels=[1,3,5,7],reduction=8,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        # 有几个卷积核,就有几个尺度, 每个尺度对应的卷积层由Conv-bn-relu实现
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=channel)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        # 将全局向量降维
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=0)
        self.conv1 =nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)
        self.conv2 =nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)
        self.conv3 =nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)
        self.conv4 =nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)



    def forward(self, x):
        # (B, C, H, W)
        B, C, H, W = x.size()
        # 存放多尺度的输出
        conv_outs=[]
        # Split: 执行K个尺度对应的卷积操作
        for conv in self.convs:
            scale = conv(x)  #每一个尺度的输出shape都是: (B, C, H, W),是因为使用了padding操作
            conv_outs.append(scale)
        feats=torch.stack(conv_outs,0) # 将K个尺度的输出在第0个维度上拼接: (K,B,C,H,W)

        # Fuse: 首先将多尺度的信息进行相加,sum()默认在第一个维度进行求和
        U=sum(conv_outs) #(K,B,C,H,W)-->(B,C,H,W)
        # 全局平均池化操作: (B,C,H,W)-->mean-->(B,C,H)-->mean-->(B,C)  【mean操作等价于全局平均池化的操作】
        S=self.gap(U)

        weights=[]
        weight=self.conv1(S) #恢复预输入相同的通道数: (B,d)-->(B,C)
        weights.append(weight.view(B,C,1,1)) # (B,C)-->(B,C,1,1)
        weight=self.conv2(S) #恢复预输入相同的通道数: (B,d)-->(B,C)
        weights.append(weight.view(B,C,1,1)) # (B,C)-->(B,C,1,1)
        weight=self.conv3(S) #恢复预输入相同的通道数: (B,d)-->(B,C)
        weights.append(weight.view(B,C,1,1)) # (B,C)-->(B,C,1,1)
        weight=self.conv4(S) #恢复预输入相同的通道数: (B,d)-->(B,C)
        weights.append(weight.view(B,C,1,1)) # (B,C)-->(B,C,1,1)
        scale_weight=torch.stack(weights,0) #将K个通道描述符在0个维度上拼接: (K,B,C,1,1)
        scale_weight=self.softmax(scale_weight) #在第0个维度上执行softmax,获得每个尺度的权重: (K,B,C,1,1)

        # Select
        V=(scale_weight*feats).sum(0) # 将每个尺度的权重与对应的特征进行加权求和,第一步是加权，第二步是求和：(K,B,C,1,1) * (K,B,C,H,W) = (K,B,C,H,W)-->sum-->(B,C,H,W)
        return V





    def forward(self, x):
        # (B, C, H, W)
        B, C, H, W = x.size()
        # 存放多尺度的输出
        conv_outs=[]
        # Split: 执行K个尺度对应的卷积操作
        for conv in self.convs:
            scale = conv(x)  #每一个尺度的输出shape都是: (B, C, H, W),是因为使用了padding操作
            conv_outs.append(scale)
        feats=torch.stack(conv_outs,0) # 将K个尺度的输出在第0个维度上拼接: (K,B,C,H,W)

        # Fuse: 首先将多尺度的信息进行相加,sum()默认在第一个维度进行求和
        U=sum(conv_outs) #(K,B,C,H,W)-->(B,C,H,W)
        # 全局平均池化操作: (B,C,H,W)-->mean-->(B,C,H)-->mean-->(B,C)  【mean操作等价于全局平均池化的操作】
        S=U.mean(-1).mean(-1)
        # 降低通道数,提高计算效率: (B,C)-->(B,d)
        Z=self.fc(S)

        # 将紧凑特征Z通过K个全连接层得到K个尺度对应的通道描述符表示, 然后基于K个通道描述符计算注意力权重
        weights=[]
        for fc in self.fcs:
            weight=fc(Z) #恢复预输入相同的通道数: (B,d)-->(B,C)
            weights.append(weight.view(B,C,1,1)) # (B,C)-->(B,C,1,1)
        scale_weight=torch.stack(weights,0) #将K个通道描述符在0个维度上拼接: (K,B,C,1,1)
        scale_weight=self.softmax(scale_weight) #在第0个维度上执行softmax,获得每个尺度的权重: (K,B,C,1,1)

        # Select
        V=(scale_weight*feats).sum(0) # 将每个尺度的权重与对应的特征进行加权求和,第一步是加权，第二步是求和：(K,B,C,1,1) * (K,B,C,H,W) = (K,B,C,H,W)-->sum-->(B,C,H,W)
        return V


class FusionConv3(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv3, self).__init__()
        self.SKfusion=SKAttention2(channel=in_channels)
        #self.spatial_attention = SpatialAttentionModule()
        self.spatial_attention = EMA(channels=in_channels)
        # self.up = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        # self.down_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):


        x_fused_c = self.SKfusion(x)

        # x_fused_c = x_fused * self.channel_attention(x_fused)
        # x_3x3 = self.conv_3x3(x_fused)
        # x_5x5 = self.conv_5x5(x_fused)
        # x_7x7 = self.conv_7x7(x_fused)
        #x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_c * self.spatial_attention(x)

        x_out = x_fused_s + x

        return x_out

class GSConvE4(nn.Module):
    """
    修正版：确保 groups 合理、并在拼接前做空间对齐，同时保留指定的通道 shuffle。
    假设 c2 % 4 == 0，否则请先调整 c2。
    """
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        # 要求 c2 能被 4 整除以保证分支尺寸和 shuffle 逻辑正确
        assert c2 % 4 == 0, "c2 must be divisible by 4"
        c_ = c2 // 2    # x1 channels
        c__ = c2 // 4   # x2 and x3 channels

        # 主分支（可以带 stride s）
        self.cv1 = Conv(c1=c1, c2=c_, k=k, s=s, p=None, g=g, d=d, act=act)

        # 次分支：这里我把 g 设为 1（普通 conv），保持行为可控
        self.cv2 = Conv(c1=c_, c2=c__, k=3, s=1, p=None, g=1, d=d, act=act)
        self.cv3 = Conv(c1=c__, c2=c__, k=3, s=1, p=None, g=1, d=d, act=act)

        # 第三分支：depthwise conv (groups == in_channels) + 1x1
        # # 注意输入通道为 c__
        # self.cv3 = nn.Sequential(
        #     Conv(c1=c__, c2=c__, k=3, s=1, p=1, g=c__, d=d, act=act),  # depthwise
        #     Conv(c1=c__, c2=c__, k=1, s=1, p=None, g=1, d=1, act=act)
        # )

    def forward(self, x):
        x1 = self.cv1(x)      # [B, c_, H1, W1]
        x2 = self.cv2(x1)     # [B, c__, H2, W2]  (可能和 x1 不同)
        x3 = self.cv3(x2)     # [B, c__, H2, W2]  (与 x2 相同空间尺寸)

        # --- 在 cat 前对齐空间尺寸（按 x1 的尺寸）

        # 现在可以安全 cat
        y = torch.cat((x1, x2, x3), dim=1)  # [B, C, H, W], C == c2

        # --- shuffle channels to pattern [0, c_, 1, c_+c__, 2, c_+1, ...] per your requested ordering
        B, C, H, W = y.shape
        # 这里我们假设 c2 % 4 == 0，c__ = C // 4, c_ = 2*c__
        c__ = C // 4
        c_ = 2 * c__

        # 生成索引： [0_from_x1, 0_from_x2, 1_from_x1, 0_from_x3, 2_from_x1, 1_from_x2, 3_from_x1, 1_from_x3, ...]
        # 按你之前例子期望 [0,4,1,6,2,5,3,7,...] 的规律生成索引
        idx = []
        for i in range(c__):
            idx.append(2 * i)            # x1 channels 0,2,4,...
            idx.append(c_ + i)           # x2 channels
            idx.append(2 * i + 1)        # x1 channels 1,3,5,...
            idx.append(c_ + c__ + i)     # x3 channels

        idx = torch.tensor(idx, dtype=torch.long, device=y.device)
        y = y.index_select(dim=1, index=idx)

        return y

class GSConvE5(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    '''
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c__ = c2 // 4
        c_ = c__*3
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            Conv(c1=c_, c2=c_, k=3, s=1, p=1, g=c_, d=d, act=act),
            Conv(c1=c_, c2=c__, k=1, s=1, p=None, g=1, d=1, act=act)
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        return y

class Concat2(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)
from ultralytics.nn.modules.block import C2f
class F_Concat(nn.Module):
    def __init__(self, dimension=1):
        super(F_Concat, self).__init__()
        self.d=dimension
        # self.Channel1=Channel1
        # self.w = nn.Parameter(torch.ones(self.c1, dtype=torch.float32), requires_grad=True)



    def forward(self, x: List[torch.Tensor]):
        x1, x2 = x
        N1, C1, H1, W1 = x1.size()
        N2, C2, H2, W2 = x2.size()
        C3=C1+C2
        epsilon = 1e-4
        w = nn.Parameter(torch.ones(C3, dtype=torch.float32), requires_grad=True)
        w = w[:(C1 + C2)]
        weight = w / (torch.sum(w) + epsilon)

        x1 = (weight[:C1] * x1.view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:] * x2.view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        y=[x1,x2]
        return torch.cat(y, 1)

class F_Concat2(nn.Module):
    def __init__(self, dimension=1, Channel1 = 1, Channel2 = 1):
        super(F_Concat2, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel_all = int(Channel1 + Channel2)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型 parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化

    def forward(self, x):
        # x: list/tuple of two tensors
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()

        real_total = C1 + C2
        w = self.w

        # 如果 self.w 长度不够就 pad，如果太长就截断（不改变原来注册的 self.w）
        if w.numel() < real_total:
            # padding with ones (neutral)
            pad = torch.ones(real_total - w.numel(), device=w.device, dtype=w.dtype)
            weight_full = torch.cat([w, pad], dim=0)
        elif w.numel() > real_total:
            weight_full = w[:real_total]
        else:
            weight_full = w

        weight = weight_full / (torch.sum(weight_full, dim=0) + self.epsilon)

        # shape alignment: use broadcasting safely
        # x[0]: (N1, C1, H1, W1) -> permute to (N1, H1, W1, C1)
        x0_perm = x[0].permute(0, 2, 3, 1)
        x1_perm = x[1].permute(0, 2, 3, 1)

        w0 = weight[:C1].view(1, 1, 1, C1)
        w1 = weight[C1:C1 + C2].view(1, 1, 1, C2)

        x1_scaled = (w0 * x0_perm).permute(0, 3, 1, 2)
        x2_scaled = (w1 * x1_perm).permute(0, 3, 1, 2)

        return torch.cat([x1_scaled, x2_scaled], dim=self.d)


# class F_Concat(nn.Module):
#     # NOTE: 参数顺序必须是 (c1, c2, dimension=1)
#     def __init__(self, c1, c2, dimension=1):
#         super().__init__()
#         # parser will pass c1 and c2 automatically
#         self.c1 = int(c1)                  # input channels (first input)
#         self.c2 = int(c2)                  # input channels (second input)
#         self.d = int(dimension)            # concat dim (usually 1 for channel)
#         self.Channel_all = self.c1 + self.c2
#
#         # learnable fusion weights (length = output channels)
#         self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
#         self.epsilon = 1e-4
#
#         # tell parser module output channels
#         # self.c1 already input; set self.c2 to output channels per Ultralytics convention
#         self.out_channels = self.Channel_all  # optional name
#         self.c3=self.c2
#         self.c2 = self.Channel_all
#         return self.c2f(y)



    # def forward(self, x):
    #     # x is a list of two tensors [x1, x2]
    #     x1, x2 = x
    #     N1, C1, H1, W1 = x1.shape
    #     N2, C2, H2, W2 = x2.shape
    #
    #     # use actual channel counts at runtime (in case of dynamic shapes)
    #     total_C = C1 + C2
    #     w = self.w[:total_C]                         # cut to actual length
    #     weight = w / (torch.sum(w) + self.epsilon)   # normalized scalar weights
    #
    #     # apply per-channel scaling: expand weight to match feature layout
    #     # weight[:C1] is shape (C1,), we need to broadcast multiply on channel dim
    #     # easiest: reshape to (1, C, 1, 1)
    #     w1 = weight[:C1].view(1, C1, 1, 1)
    #     w2 = weight[C1:].view(1, C2, 1, 1)
    #
    #     x1 = x1 * w1
    #     x2 = x2 * w2
    #     x=[x1, x2]
    #     y=torch.cat(x, dim=self.d)
    #     return self.c2f(y)


class gatedFusion(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super(gatedFusion, self).__init__()
        dim = dim//2
        # 使用1x1卷积代替全连接层，保持空间维度
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        # 输入形状: [B, C, H, W]
        x1,x2=x[0],x[1]
        x11 = self.conv1(x1)  # 形状保持 [B, C, H, W]
        x22 = self.conv2(x2)  # 形状保持 [B, C, H, W]

        # 通过门控单元生成权重
        z = torch.sigmoid(x11 + x22)  # 形状 [B, C, H, W]

        # 加权融合
        out = z * x1 + (1 - z) * x2  # 形状 [B, C, H, W]

        return out






class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

        # carafe
        # from mmcv.ops import CARAFEPack
        # self.upsample = nn.Sequential(
        #     BasicConv(in_channels, out_channels, 1),
        #     CARAFEPack(out_channels, scale_factor=scale_factor)
        # )

    def forward(self, x):
        x = self.upsample(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        x = self.downsample(x)

        return x


class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[256, 512]):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        self.upsample = Upsample(channel[1], channel[0])
        self.downsample = Downsample(channel[0], channel[1])
        self.level = level
        self.c2f = C2f(inter_dim, inter_dim)
    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return self.c2f(out)


# class LAE(nn.Module):
#     # Light-weight Adaptive Extraction
#     def __init__(self, ch, group=16) -> None:
#         super().__init__()
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.attention = nn.Sequential(
#             nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
#             Conv(ch, ch, k=1)
#         )
#
#         self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))
#
#     def forward(self, x):
#         # bs, ch, 2*h, 2*w => bs, ch, h, w, 4
#         att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
#         att = self.softmax(att)
#
#         # bs, 4 * ch, h, w => bs, ch, h, w, 4
#         x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
#         x = torch.sum(x * att, dim=-1)
#         return x

class GConv(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv2 = Conv(dim, dim, 3, 2, 1, g=dim // 2, act=False)
        self.conv4 = Conv(dim, dim_out, 1, 1)

    def forward(self, x):
        x2 = self.conv2(x)
        x2 = self.conv4(x2)
        return x2

import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision import transforms
import cv2
import os
from thop import profile

class single_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(single_conv, self).__init__()
        self.s_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.s_conv(x)
        return x
class WDown(nn.Module):
    """
    Downsampling block used in ASCNet
    Consists of:
    1) Conv branch: 3×3 Conv (stride=2)
    2) Wavelet branch: Haar DWT → concat → 3×3 Conv
    3) Element-wise summation
    """

    def __init__(self, in_channels, out_channels):
        super(WDown, self).__init__()

        # -------- Conv downsample branch --------
        self.conv_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # -------- Wavelet branch --------
        self.dwt = DWTForward(J=1, wave='haar')

        # 对应 dconv_encode1 / dconv_encode2 / dconv_encode3
        self.wavelet_conv = single_conv(
            in_channels=4 * in_channels,
            out_channels=out_channels
        )

    def _transformer(self, yl, yh):
        """
        与 ASCNet 中 _transformer 完全一致
        yl: [B, C, H/2, W/2]
        yh[0]: [B, C, 3, H/2, W/2]
        """
        a = yh[0]
        return torch.cat([
            yl,
            a[:, :, 0, :, :],
            a[:, :, 1, :, :],
            a[:, :, 2, :, :]
        ], dim=1)

    def forward(self, x):
        # Conv branch
        out_conv = self.conv_down(x)

        # Wavelet branch
        yl, yh = self.dwt(x)
        wavelet_feat = self._transformer(yl, yh)
        out_wavelet = self.wavelet_conv(wavelet_feat)

        # Fusion
        out = torch.add(out_conv, out_wavelet)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import pywt.data
import torch
import torch.nn.functional as F


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, _ = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c // 4, padding=pad)
    return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale, requires_grad=True)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class WD2(nn.Module):
    def __init__(self, in_channels, out_channels, wt_type='db1'):
        super().__init__()

        self.wt_filter, _ = create_wavelet_filter(wt_type, in_channels, in_channels)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)

        self.para_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, 3, 2,groups=8,padding=1),
            nn.BatchNorm2d(in_channels * 4),
            nn.SiLU(inplace=True)
        )

        self.ca = ChannelAttention(in_channels * 4)
        self.wavelet_scale = _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x_conv = self.para_conv(x)
        ca = self.ca(x_conv)
        x_conv = x_conv * ca

        x_wt = wavelet_transform(x, self.wt_filter)
        b, c, _, h, w = x_wt.shape
        x_wt = x_wt.view(b, c * 4, h, w)
        x_wt = self.wavelet_scale(x_wt) * ca

        out = x_conv + x_wt
        out = self.proj(out)
        return out
