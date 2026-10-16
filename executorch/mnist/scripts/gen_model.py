#!/usr/bin/env python3
"""Export the MNIST MLP (784-32-16-10) to an ExecuTorch .pte lowered to XNNPACK.

Usage:  python gen_model.py --precision fp32|bf16 --pte models/mnist_mlp_fp32.pte

Needs the `executorch` and `torch` Python packages (opt-in dependency, see the ExecuTorch section of the repo README.md).

Credits: model definition (TinyMLPMNIST, create_balanced_model) and the
export flow (export -> to_edge_transform_and_lower(XnnpackPartitioner) -> to_executorch)
are taken from samples/exeggutorch/model/gen_pte.py of the author's repo
pandabon/zephyr-chipyard-sw, branch exeggutorch @ e12bfa38034 (author: Jonathan Lin,
Berkeley; hand-crafted weights recognising digits 0, 1, 4, 7).
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export

# If the pip-installed executorch has no (or a dangling) backends/xnnpack, borrow the
# python sources of the checkout in $EXECUTORCH_DIR (same version as the C++ runtime).
try:
    import executorch.backends.xnnpack  # noqa: F401
except ImportError:
    _et = os.environ.get("EXECUTORCH_DIR")
    if not _et:
        raise SystemExit("executorch.backends.xnnpack is not importable: install executorch "
                         "(pip) or set EXECUTORCH_DIR to an ExecuTorch checkout")
    import executorch.backends as _b
    _b.__path__.append(os.path.join(_et, "backends"))

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

_S = 28  # MNIST image side length
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


class TinyMLPMNIST(nn.Module):
    """Small MLP for MNIST digit classification (784 -> 32 -> 16 -> 10)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_S * _S, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def create_balanced_model():
    """Hand-crafted weights that recognise MNIST digits 0, 1, 4, and 7."""
    model = TinyMLPMNIST()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        w1 = model.fc1.weight
        w1[0, 14::_S] = 1.87                                  # vertical line at column 14
        w1[1, : 3 * _S] = 1.87                                # top band
        w1[2, 25 * _S:] = 1.87                                # bottom band
        w1[3, _S + 8: _S + 20] = 1.87                         # oval detector for 0
        w1[3, 26 * _S + 8: 26 * _S + 20] = 1.87
        rows = torch.arange(4, 24)
        w1[3, rows * _S + 7] = 1.87
        w1[3, rows * _S + 20] = 1.87
        rows = torch.arange(10, 18)
        w1[3, rows * _S + 14] = -1.5                          # hollow centre
        w1[4, 13 * _S: 15 * _S] = 3.0                         # middle band

        w2 = model.fc2.weight
        w2[0, [3, 0, 4]] = torch.tensor([5.0, -2.0, -3.0])
        w2[1, [0, 2, 1, 3]] = torch.tensor([3.0, 2.0, -1.0, -2.0])
        w2[2, [0, 1, 4, 3]] = torch.tensor([2.0, 1.0, 4.0, -2.0])
        w2[3, [1, 0, 2]] = torch.tensor([3.0, 1.0, -2.0])

        w3 = model.fc3.weight
        w3[0, 0] = 5.0
        w3[1, 1] = 5.0
        w3[4, 2] = 5.0
        w3[7, 3] = 5.0
        model.fc3.bias[[2, 3, 5, 6, 8, 9]] = -3.0
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=list(DTYPES), default="fp32")
    ap.add_argument("--pte", required=True, help="output .pte")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = create_balanced_model().eval()
    sample = (torch.randn(1, _S, _S),)
    if args.precision != "fp32":
        model = model.to(DTYPES[args.precision])
        sample = tuple(t.to(DTYPES[args.precision]) for t in sample)

    edge = to_edge_transform_and_lower(export(model, sample), partitioner=[XnnpackPartitioner()])
    with open(args.pte, "wb") as f:
        edge.to_executorch().write_to_file(f)
    print("wrote", args.pte, os.path.getsize(args.pte), "bytes")


if __name__ == "__main__":
    main()
