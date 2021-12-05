# Copyright (c) SEQeta, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import triton
import triton.language as tl


# fmt: off
@triton.jit
def k_causal_product(
    OUT,                # out ptr
    Q, K, V,            # in ptrs
    SEQ, DIM, DIMv,     # dims
    stride_out_b, stride_out_seq, stride_out_chunk,
    stride_q_b, stride_q_seq,
    stride_k_b, stride_k_seq,
    stride_v_b, stride_v_seq,
    **meta,             # Optional meta-parameters for the kernel
):
    # fmt: on

    BLOCK_SIZE_QK = meta["BLOCK_SIZE_QK"]
    BLOCK_SIZE_V = meta["BLOCK_SIZE_V"]

    pid_b = tl.program_id(axis=0)
    pid_chunk = tl.program_id(axis=1)

    # We pull in the full V row, but chunk over K and Q
    # so that the bigger matrices can fit in.
    # Not doing that would mean that we need to materialize the whole prefix-sum
    # in the shared memory, and this does not fit most of the time

    range_qk = pid_chunk * BLOCK_SIZE_QK + tl.arange(0, BLOCK_SIZE_QK)
    range_v = tl.arange(0, BLOCK_SIZE_V)

    mask_v = range_v < DIMv
    mask_qk = range_qk < DIM

    # Offset by batch size
    cur_q_pos = Q + pid_b * stride_q_b
    cur_k_pos = K + pid_b * stride_k_b
    cur_v_pos = V + pid_b * stride_v_b
    cur_out_pos = OUT + pid_b * stride_out_b + pid_chunk * stride_out_chunk

    # Carry over matrix containing the latest prefix_sum chunk
    prefix_sum = tl.zeros((BLOCK_SIZE_QK, BLOCK_SIZE_V), dtype=tl.float32)

    for _ in range(0, SEQ):
        # -- Compute the latest prefix-sum
        # Note that each kernel focuses on a few lines of the prefix_sum
        # We only load a chunk of K
        k_ptrs = cur_k_pos + range_qk
        k = tl.load(k_ptrs, mask=mask_qk, other=0)

        # We load the full V row
        v_ptrs = cur_v_pos + range_v
        v = tl.load(v_ptrs, mask=mask_v, other=0)

        # Compute latest prefix_sum, outer product
        prefix_sum += tl.dot(k[:, None], v[None, :])

        # -- Now use a new query, and save the result
        # Load a single chunk of Q
        q_ptrs = cur_q_pos + range_qk
        q = tl.load(q_ptrs, mask=mask_qk, other=0)

        # Compute output = QKV. [1, BLOCK] x [BLOCK, D] => [1, D]
        output = tl.dot(q[None, :], prefix_sum)

        # Store the result of this row
        out_ptr = cur_out_pos + range_v
        tl.store(out_ptr[None, :], output, mask=mask_v[None, :])

        # move to next row
        cur_q_pos += stride_q_seq
        cur_k_pos += stride_k_seq
        cur_v_pos += stride_v_seq
        cur_out_pos += stride_out_seq
