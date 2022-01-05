# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# from xformers.triton.k_outer_product_mean import k_outer_product_mean
from xformers.triton.k_fused_matmul_fw import fused_matmul


def _sanitize(x):
    if x.ndim > 2:
        x_ = x.reshape(-1, x.shape[-1])
    else:
        x_ = x

    if not x_.is_contiguous():
        x_ = x_.contiguous()

    return x_


def outer_product_mean(a, b, average: bool = True):
    """
    Implements Algorithm 10 in the supplementary data of
    "Highly accurate protein structure prediction with AlphaFold",
    Jumper et al. (https://doi.org/10.1038/s41586-021-03819-2)

    The notations are preserved, in that we'll compute the outer product in between
    A(s, i) and B(s, j), and then mean over s

    average: compute the average
        (if not the results of the outer product are passed as is)
    """

    # Make sure that we're in the known [i, s] and [j, s] configuration
    assert a.shape[-1] == b.shape[-1]
    assert a.ndim == b.ndim

    a_ = _sanitize(a)
    b_ = _sanitize(b)

    outputs, _ = fused_matmul(
        a_,
        b_,
        bias=None,
        activation=None,
        save_act_inputs=False,
        scale=1.0 / a_.shape[1] if average else None,
    )

    return outputs.reshape(a.shape[0], a.shape[-2], b.shape[-2])
