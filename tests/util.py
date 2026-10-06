import os

import torch

C64, C128 = torch.complex64, torch.complex128

# Physics gates run on CUDA when present (fast), otherwise on CPU.
# Set QS_TEST_DEVICE=cpu to force the CPU path.
DEVICE = os.environ.get("QS_TEST_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def rel_l2(a, b):
    return float((a - b).abs().pow(2).sum().sqrt() / b.abs().pow(2).sum().sqrt())


def max_abs(a, b):
    return float((a - b).abs().max())
