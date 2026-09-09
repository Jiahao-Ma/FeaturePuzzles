# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch

try:
    import curope as _kernels # run `python setup.py install`
except ModuleNotFoundError:
    from . import curope as _kernels # run `python setup.py build_ext --inplace`


class cuRoPE2D_func (torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        _kernels.rope_2d( tokens, positions, base, F0 )
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        grad_res = grad_res.contiguous()
        _kernels.rope_2d( grad_res, positions, base, -F0 )
        return grad_res, None, None, None


class cuRoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq 
        self.F0 = F0

    def forward(self, tokens, positions):
        orig_dtype = tokens.dtype
        if orig_dtype != torch.float32:
            tokens = tokens.float()
        t = tokens.transpose(1, 2).contiguous()
        t = cuRoPE2D_func.apply(t, positions, self.base, self.F0)
        result = t.transpose(1, 2).contiguous()
        if result.dtype != orig_dtype:
            result = result.to(orig_dtype)
        return result