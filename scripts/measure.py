# Copyright (c) Alibaba, Inc. and its affiliates.
import torch

from models.common import *
from utils.torch_utils import time_synchronized


def get_gflops(module, cfg):
    from thop import profile

    model = eval(module)(*cfg).cuda()
    params = sum([x.numel() for x in model.parameters()]) / 1024

    s = 640 // 8
    feat = torch.zeros((1,cfg[0],s,s)).cuda()  # input
    flops = profile(model, inputs=(feat,), verbose=False)[0] / 1E9 * 2  # GFLOPs

    n = 10
    t0 = 0
    for _ in range(n):
        t = time_synchronized()
        model(feat)
        t0 += time_synchronized() - t
    speed = t0 / n * 1e3

    print('%10s%10.2f%10.2f%10.2f' % (module, params, flops, speed))



if __name__ == '__main__':
    modules = [
        ['Conv', [192, 192, 3, 1]],
        ['DCN', [192, 192, 3, 1]],  # DCN+Conv
        ['DWConv', [192, 192, 7, 1]],
        ['DWConv', [192, 192, 13, 1]],
        ['DWConv', [192, 192, 31, 1]],
        ['SPP', [192, 192, [5, 9, 13]]],
        ['ASPP', [192, 192, [1, 2, 4, 6]]],
    ]
    for m, cfg in modules:
        get_gflops(m, cfg)
