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

    def __init__(self, in_features, out_features, bias=True, bitW = 8):
        super(quantized_Linear, self).__init__(in_features, out_features, bias=bias)

        self.quantized_weight = None
        self.pre_quantized_weight = None
        self.bitW = bitW
        self.alpha = None

        print('Initial Quantized Linear with bit %d' % self.bitW)

    def forward(self, input, quantized_type = 'basic'):

        if self.bitW == 32:
            self.quantized_weight = self.weight * 1.0
            return F.linear(input, self.quantized_weight, self.bias)

        if quantized_type == 'dorefa':
            temp_weight = torch.tanh(self.weight)
            self.pre_quantized_weight = (temp_weight / torch.max(torch.abs(temp_weight)).detach()) * 0.5 + 0.5
            self.quantized_weight = 2 * Function_STE.apply(self.pre_quantized_weight, self.bitW) - 1
        elif quantized_type in ['BWN', 'BWN-F']:
            self.alpha = torch.sum(torch.abs(self.weight.data)) / self.weight.numel()
            self.pre_quantized_weight = self.weight
            self.quantized_weight = self.alpha.data * Function_BWN.apply(self.pre_quantized_weight)
        elif quantized_type == 'basic':
            max_value = torch.max(torch.abs(self.weight.data))
            self.pre_quantized_weight = (self.weight / max_value) * 0.5 + 0.5
            self.quantized_weight = max_value * (2 * Function_STE.apply(self.pre_quantized_weight, self.bitW) - 1)
        else:
            self.quantized_weight = self.weight * 1.0

        return F.linear(input, self.quantized_weight, self.bias)


class quantized_Embedding(nn.Module):

    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None, bitW=1):
        super(quantized_Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)
        self.sparse = sparse

        self.quantized_weight = None
        self.pre_quantized_weight = None
        self.bitW = bitW

        print('Initial Quantized Embedding with bit %d' %self.bitW)

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input, quantized_type = 'basic'):

        if self.bitW == 32:
            self.quantized_weight = self.weight * 1.0
        else:
            if quantized_type == 'dorefa':
                temp_weight = torch.tanh(self.weight)
                self.pre_quantized_weight = (temp_weight / torch.max(torch.abs(temp_weight)).detach()) * 0.5 + 0.5
                self.quantized_weight = 2 * Function_STE.apply(self.pre_quantized_weight, self.bitW) - 1
            elif quantized_type in ['BWN', 'BWN-F']:
                self.alpha = torch.sum(torch.abs(self.weight.data)) / self.weight.numel()
                self.pre_quantized_weight = self.weight
                self.quantized_weight = self.alpha.data * Function_BWN.apply(self.pre_quantized_weight)
            elif quantized_type == 'basic':
                max_value = torch.max(torch.abs(self.weight.data))
                self.pre_quantized_weight = (self.weight / max_value) * 0.5 + 0.5
                self.quantized_weight = max_value * (2 * Function_STE.apply(self.pre_quantized_weight, self.bitW) - 1)
            else:
                raise NotImplementedError

        return F.embedding(
            input, self.quantized_weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding



def check_quantized_bits(layer, bitW):

    if bitW == 32:
        return

    quantized_weight = layer.quantized_weight
    assert len(torch.unique(quantized_weight)) <= 2**bitW


if __name__ == '__main__':
    bitW = 2
    m = quantized_Linear(in_features=3, out_features=5, bitW=bitW)
    inputs = torch.rand([10, 3])
    outputs = m(inputs, quantized_type = 'basic')
    check_quantized_bits(m, bitW=bitW)