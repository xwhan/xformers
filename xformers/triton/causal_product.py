# Copyright (c) Meta, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton

from xformers.triton.k_causal_product import k_causal_product


def causal_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    assert q.ndim == 3, "We assume Batch x Seq x Dim as input dimensions"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.size() == k.size()
    assert q.size()[:-1] == v.size()[:-1], (q.size(), v.size())
    assert q.stride(-1) < 2 and k.stride(-1) < 2 and v.stride(-1) < 2  # 0 or 1 is ok

    # We need to preallocate the output
    B, SEQ, DIM = q.size()
    DIMv = v.size(-1)

    block_size_v = triton.next_power_of_2(DIMv)

    # Chunk the prefix-sum computations
    # TODO: If small enough, do it in one go
    BLOCK_SIZE_QK = DIM  # 32
    if SEQ > 512:
        BLOCK_SIZE_QK = 8

    N_QK_BLOCKS = triton.cdiv(DIM, BLOCK_SIZE_QK)

    output = torch.empty((B, SEQ, N_QK_BLOCKS, DIMv), dtype=q.dtype, device=q.device)

    # grid is one per batch dim x one per chunk
    def grid(_):
        return (B, N_QK_BLOCKS)

    # fmt: off
    k_causal_product[grid](
        output,
        q, k, v,
        SEQ, DIM, DIMv,
        output.stride(0), output.stride(1), output.stride(2),
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        # num_warps=2,
        BLOCK_SIZE_QK=BLOCK_SIZE_QK,
        BLOCK_SIZE_V=block_size_v,
    )
    # fmt: on

    # Epilogue, sum up the chunk contributions
    return torch.sum(output, -2)
