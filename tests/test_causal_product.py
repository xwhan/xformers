# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers import _is_triton_available
from xformers.components.attention.utils import causal_product as pytorch_causal_product

if _is_triton_available:
    try:
        from xformers.triton import causal_product

    except (ImportError, ModuleNotFoundError):
        _is_triton_available = False

SHAPES = [
    (2, 128, 128),
    # (2, 1024, 1024),  # Test something which cannot hold in shared memory on purpose
]


@pytest.mark.skipif(
    not _is_triton_available, reason="Triton requires a recent CUDA gpu"
)
@pytest.mark.parametrize("shape", SHAPES)
def test_causal_product(shape):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    q = torch.rand(shape, device=torch.device("cuda"))
    k = torch.rand(shape, device=torch.device("cuda"))
    v = torch.rand(shape, device=torch.device("cuda"))

    ref = pytorch_causal_product(q, k, v)
    test = causal_product(q, k, v)
    assert torch.allclose(ref, test, rtol=1e-3)
