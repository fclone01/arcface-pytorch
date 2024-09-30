from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin

        cos(theta + m)
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        # m = 0.5 (paper recommend)
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin  # False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # Ngưỡng  = -0.87758256189. m = 60 độ
        self.mm = math.sin(math.pi - m) * m  # hằng số điều chỉnh = 0.239712769302

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))  # kẹp sine (0,1)
        phi = cosine * self.cos_m - sine * self.sin_m  # phi = cos(theta - m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Next step
            #  cosine > self.th <=> theta < pi - m <=> theta + m < pi <=> theta <  120 (ok). !ok =>
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
            (1.0 - one_hot) * cosine
        )  # you can use torch.where if your torch.__version__ is 0.4

        output *= self.s

        # output : True_label: phi; False_label: cosin

        return output
