import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Function_sparsify(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, mask):
        ctx.save_for_backward(weight)
        return weight * mask

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, = ctx.saved_tensors
        return grad_outputs, torch.mul(grad_outputs, weight)


class sparse_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(sparse_Linear, self).__init__(in_features, out_features, bias=bias)
        """
        mask: [O, I]
        bias: [O]
        """

        self.sparse_weight = None
        self.sparse_bias = None
        print('Initialize sparse Linear.')

    def forward(self, input, mask = None, backward ='direct'):

        if mask is not None:
            if backward == 'direct':
                self.sparse_weight = self.weight * mask
                self.sparse_bias = self.bias * torch.mean(mask, dim=1)
            elif backward == 'STE':
                self.sparse_weight = Function_sparsify.apply(self.weight, mask)
                self.sparse_bias   = Function_sparsify.apply(self.bias, torch.mean(mask, dim=1))
            else:
                raise NotImplementedError

            return F.linear(input, self.sparse_weight, self.sparse_bias)

        else:
            return F.linear(input, self.weight, self.bias)


def generate_structure_mask(criterion, structure_dim = 0, hidden_size=768, num_attention_heads=12, CR=0.5):
    """
    This function generate structure mask for a projection layer (key, query, value)
    Args:
        criterion:
            The weights of projection layer:
            [hidden_size, all_head_size] / [hidden_size, attention_head_size x num_attention_heads]

    Returns:

    """
    hidden_size = hidden_size
    num_attention_heads = num_attention_heads
    head_size = int(hidden_size / num_attention_heads)
    multi_head_criterion = criterion.view(num_attention_heads, head_size, hidden_size) # [12, 64, 768]
    prune_dim = [0, 1, 2]
    prune_dim.pop(structure_dim)
    # print(prune_dim)
    mean_head_criterion = torch.mean(torch.abs(multi_head_criterion), dim=prune_dim)
    # print(mean_head_criterion.shape)
    n_remain = int(np.ceil(CR * mean_head_criterion.numel()))
    # print(n_remain)
    thres_list, _ = torch.topk(mean_head_criterion, n_remain)
    pruned_criterion = (mean_head_criterion >= thres_list[-1]).float() # [12]
    # print(pruned_criterion)
    # pruned_criterion = np.repeat(pruned_criterion, head_size)
    # pruned_criterion = np.repeat(pruned_criterion.reshape(1, -1), hidden_size, axis=0)
    pruned_criterion = torch.repeat_interleave(pruned_criterion, head_size)
    pruned_criterion = torch.repeat_interleave(pruned_criterion.view(1, -1), hidden_size, dim=0)

    return pruned_criterion.transpose(1, 0)


def check_head_sparsity(projection_output, structure_dim = 0, CR = 0.5):

    mean_dim = [0, 1, 2]
    mean_dim.pop(structure_dim)
    mean_projection_output = projection_output.sum(0).sum(mean_dim)
    # print(mean_projection_output)
    # print(torch.sum(mean_projection_output != 0.0))
    # print(mean_projection_output.numel())
    cur = torch.sum(mean_projection_output != 0.0).item() / mean_projection_output.numel()
    assert  CR -0.1 <= cur <= CR + 0.1

def check_mask_sparsity(mask, CR):

    nnz = torch.sum(mask != 0)
    total = mask.numel()

    assert CR - 0.1 <= nnz / total <= CR + 0.1

def transpose_for_scores(x, num_attention_heads, attention_head_size):
        """
        This function transpose [bs, seq, hidden] => [bs, num_heads, seq, attention_head_size]
        x: [bs, seq, hidden]
        x.permute(0,2,1,3): [bs, num_attention_heads, seq, attention_head_size]
        """
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


if __name__ == '__main__':

    from transformers import BertConfig
    import torch.nn as nn

    bertConfig = BertConfig()
    bertConfig.CR = 0.5
    m = sparse_Linear(768, 768)
    mask = generate_structure_mask(
        m.weight.data, hidden_size=bertConfig.hidden_size, num_attention_heads=bertConfig.num_attention_heads, CR=bertConfig.CR
    )
    # print(mask)

    inputs = torch.rand([10, 128, 768])

    outputs = m(inputs, mask) #[10, 128, 768]
    # print(torch.sum(outputs != 0).item() / outputs.numel())
    attention_head_size = int(bertConfig.hidden_size / bertConfig.num_attention_heads)
    # [10, 128, 768] => [10, 128, 12, 64] => [10, 12, 128, 64]
    value_layer = transpose_for_scores(outputs, bertConfig.num_attention_heads, attention_head_size)
    check_head_sparsity(value_layer, CR=bertConfig.CR)