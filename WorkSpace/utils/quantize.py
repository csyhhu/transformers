"""
A set of helper functions for quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np

def quantize(x, k):
    n = float(2 ** k - 1)
    return torch.round(x * n) / n

def fw(param, bitW = 1):
    x = torch.tanh(param)
    x = x / torch.max(torch.abs(x)) * 0.5 + 0.5
    return 2 * quantize(x, bitW) - 1


class Function_BWN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight):
        # alpha = torch.sum(torch.abs(weight.data)) / weight.numel()
        ctx.save_for_backward(weight)
        return torch.sign(weight)

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, = ctx.saved_tensors
        gate = (torch.abs(weight) <= 1).float()
        grad_inputs = grad_outputs * gate
        return grad_inputs, None


class Function_STE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, bitW):
        ctx.save_for_backward(weight)
        n = float(2 ** bitW - 1)
        return torch.round(weight * n) / n

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, = ctx.saved_tensors
        gate = (torch.abs(weight) <= 1).float()
        grad_inputs = grad_outputs * gate
        return grad_inputs, None


class quantized_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, bitW = 1):
        super(quantized_Linear, self).__init__(in_features, out_features, bias=bias)

        self.quantized_weight = None
        self.pre_quantized_weight = None
        self.bitW = bitW
        self.alpha = None

        print('Initial quantized Linear with bit %d' % self.bitW)

    def forward(self, input, quantized = 'dorefa'):

        if quantized == 'dorefa':
            temp_weight = torch.tanh(self.weight)
            self.pre_quantized_weight = (temp_weight / torch.max(torch.abs(temp_weight)).detach()) * 0.5 + 0.5
            self.quantized_weight = 2 * Function_STE.apply(self.pre_quantized_weight, self.bitW) - 1
        elif quantized in ['BWN', 'BWN-F']:
            self.alpha = torch.sum(torch.abs(self.weight.data)) / self.weight.numel()
            self.pre_quantized_weight = self.weight
            self.quantized_weight = self.alpha.data * Function_BWN.apply(self.pre_quantized_weight)
        else:
            self.quantized_weight = self.weight.clone()

        return F.linear(input, self.quantized_weight, self.bias)


# def check_quantized_bits(net, bitW):
#
#     for layer_name, layer_idx in net.layer_name_list:
#         layer = get_layer(net, layer_idx)
#
#         quantized_params = layer.quantized_params
#         assert len(torch.unique(quantized_params)) <= 2**bitW


if __name__ == '__main__':
    pass