#!/usr/bin/env python3
"""Export one SmolVLA action-expert layer to an ExecuTorch .pte (XNNPACK delegate) + golden data.

  python export_expert_layer.py --layer 1 --prefix-len 241 --out exports/expert_l1_fp32

Writes <out>.pte, <out>.inputs.bin (all inputs in the model dtype, concatenated in input order),
<out>.expected.bin (PyTorch output, model dtype) and <out>.json (shapes, delegation summary).
Export flow follows ExecuTorch's XNNPACK tutorial (torch.export -> to_edge_transform_and_lower
with XnnpackPartitioner -> to_executorch), as executorch/mnist/scripts/gen_model.py in this repo.
"""

import argparse
import glob
import json
import os
import subprocess

import torch
from torch.export import export

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import get_delegation_info
from executorch.backends.xnnpack.utils.configs import get_transform_passes
from executorch.exir import to_edge_transform_and_lower

from expert_layer import (KV_HEADS, HEAD_DIM, load_expert_layer, load_expert_layer_wl5,
                          load_expert_layer_lerobot_bf16)

# Model input/output dtype per mode.
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "wl5": torch.float32, "lerobot_bf16": torch.bfloat16}
# bf16: weights, activations and inputs ALL in bf16 (blanket .to(bf16)); this is NOT the mixed-precision policy and the
#       XNNPACK runtime rejects it at load (it only accepts bf16 as matmul input with f32 output).
# wl5:  our mixed-precision policy (see ExpertLayerWL5 in expert_layer.py): inputs/outputs f32, bf16 weights
#       and bf16 casts only at matmul inputs, f32 everywhere else.
# lerobot_bf16: lerobot's own dtypes (ExpertLayerLerobotBF16 documents each cast): x is f32 for layer 0 (the f32
#       suffix embedding) and bf16 for layers >= 1; the prefix K/V cache is bf16; the output is bf16.
# ExecuTorch 1.5.1's XNNPACK partitioner delegates bf16 only with enable_bf16=True
# (backends/xnnpack/partition/config/xnnpack_config.py:53-57, opt-in).


def default_ckpt():
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/*/model.safetensors"))
    return hits[0] if hits else None


def executorch_provenance():
    """Which ExecuTorch produced this .pte (recorded in the .json).
    `executorch` is a namespace package (no __file__), so locate it via the partitioner module."""
    import importlib.metadata
    from executorch.backends.xnnpack.partition import xnnpack_partitioner as xp
    part = os.path.realpath(xp.__file__)
    try:
        top = subprocess.run(["git", "-C", os.path.dirname(part), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True).stdout.strip()
        g = lambda *a: subprocess.run(["git", "-C", top, *a], capture_output=True, text=True).stdout.strip()
        git = "%s %s%s" % (g("branch", "--show-current"), g("rev-parse", "--short", "HEAD"),
                           " dirty" if g("status", "--porcelain") else " clean")
    except (subprocess.CalledProcessError, FileNotFoundError):  # not a git checkout: a pip wheel
        git = "pip wheel " + importlib.metadata.version("executorch")
    return git, part


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layer", type=int, default=1, help="expert layer 0..15 (odd = cross-attn)")
    ap.add_argument("--seq-len", type=int, default=50, help="action tokens (chunk_size)")
    ap.add_argument("--prefix-len", type=int, default=241,
                    help="prefix tokens: 3 cams x 64 + 48 lang + 1 state at 512x512")
    ap.add_argument("--precision", default="fp32", choices=list(DTYPES))
    ap.add_argument("--ckpt", default=default_ckpt())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--etrecord", action="store_true",
                    help="also write <out>.etrecord (ExecuTorch ETRecord, for devtools Inspector/ETDump profiling); "
                         "the .pte itself is built the same way")
    a = ap.parse_args()
    dtype = DTYPES[a.precision]

    if a.precision == "lerobot_bf16":
        model = load_expert_layer_lerobot_bf16(a.ckpt, a.layer, a.seq_len, pos_offset=a.prefix_len)
    elif a.precision == "wl5":
        model = load_expert_layer_wl5(a.ckpt, a.layer, a.seq_len, pos_offset=a.prefix_len)
    else:
        model = load_expert_layer(a.ckpt, a.layer, a.seq_len, pos_offset=a.prefix_len, dtype=dtype)
    g = torch.Generator().manual_seed(a.seed)
    kv = KV_HEADS * HEAD_DIM
    x_dtype = model.x_dtype if a.precision == "lerobot_bf16" else dtype   # layer 0 of lerobot gets an f32 x
    inputs = (
        torch.randn(1, a.seq_len, 720, generator=g, dtype=torch.float32).to(x_dtype),
        torch.randn(1, a.prefix_len, kv, generator=g, dtype=torch.float32).to(dtype),
        torch.randn(1, a.prefix_len, kv, generator=g, dtype=torch.float32).to(dtype),
    )
    with torch.no_grad():
        expected = model(*inputs)
        ep = export(model, inputs)
    use_bf16 = a.precision in ("bf16", "wl5", "lerobot_bf16")
    # transform_passes=get_transform_passes() is what ExecuTorch's own XNNPACK flow passes
    # (executorch examples/xnnpack/aot_compiler.py:111, upstream pytorch/executorch #19978): it runs
    # RemoveNoopExpandCopyPass, which drops the same-shape expand_copy that matmul/bmm decomposition leaves.
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner(enable_bf16=use_bf16)],
                                       transform_passes=get_transform_passes(),
                                       generate_etrecord=a.etrecord)   # ETRecord: docs/source/etrecord.rst, Method 1
    gm = edge.exported_program().graph_module
    info = get_delegation_info(gm)
    print(info.get_summary())
    df = info.get_operator_delegation_dataframe()
    print(df.to_string())
    n_subgraphs = sum(1 for n in gm.graph.nodes if n.op == "call_function" and "executorch_call_delegate" in str(n.target))
    print("delegate subgraphs (executorch_call_delegate nodes):", n_subgraphs)
    # Casts (dtype conversions) are the ops the mixed-precision policy adds. Show which ran outside the delegate (portable ops).
    cast_ops = [str(n.target) for n in gm.graph.nodes if n.op == "call_function" and "_to_dim_order_copy" in str(n.target)]
    print("portable (non-delegated) cast ops:", len(cast_ops))
    prog = edge.to_executorch()
    if a.etrecord:
        prog.get_etrecord().save(a.out + ".etrecord")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".pte", "wb") as f:
        prog.write_to_file(f)
    def raw(t):  # bytes in the tensor's own dtype (bf16 has no numpy dtype: write its 16-bit patterns)
        t = t.contiguous()
        return (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()
    with open(a.out + ".inputs.bin", "wb") as f:
        for t in inputs:
            f.write(raw(t))
    with open(a.out + ".expected.bin", "wb") as f:
        f.write(raw(expected))
    # Error of the model vs the fp32 reference (same layer, same seed, fp32 weights/compute).
    err = {}
    if a.precision != "fp32":
        ref_model = load_expert_layer(a.ckpt, a.layer, a.seq_len, pos_offset=a.prefix_len, dtype=torch.float32)
        ref_inputs = tuple(t.float() for t in inputs)
        with torch.no_grad():
            ref = ref_model(*ref_inputs)
        d = expected.float() - ref
        err = {"rel_l2_vs_fp32": float(d.norm() / ref.norm()),
               "max_abs_vs_fp32": float(d.abs().max()),
               "cosine_vs_fp32": float(torch.nn.functional.cosine_similarity(expected.float().flatten(), ref.flatten(), dim=0))}
        print("WL error vs fp32 reference:", err)
        if a.precision == "lerobot_bf16":   # for comparison: the mixed-precision variant on the same (upcast) inputs
            wl5 = load_expert_layer_wl5(a.ckpt, a.layer, a.seq_len, pos_offset=a.prefix_len)
            with torch.no_grad():
                w = wl5(*ref_inputs)
            dw = w - ref
            err["wl5_rel_l2_vs_fp32"] = float(dw.norm() / ref.norm())
            err["wl5_max_abs_vs_fp32"] = float(dw.abs().max())
            print("mixed-precision variant error vs fp32 reference (same inputs):", err)
    et_git, et_part = executorch_provenance()
    meta = {
        **err,
        "executorch_file": os.path.realpath(__import__("executorch").__path__[0]),  # package dir (namespace pkg: no __file__)
        "executorch_git": et_git,
        "xnnpack_partitioner_file": et_part,
        "layer": a.layer, "cross_attn": a.layer % 2 == 1, "precision": a.precision,
        "inputs": [list(t.shape) for t in inputs], "output": list(expected.shape),
        "seed": a.seed, "ckpt": a.ckpt, "pte_bytes": os.path.getsize(a.out + ".pte"),
        "delegated_ops": int(info.num_delegated_nodes), "non_delegated_ops": int(info.num_non_delegated_nodes),
        "torch": torch.__version__,
        "expected_head": expected.flatten()[:8].tolist(),
        "expected_sum": float(expected.double().sum()), "expected_abs_sum": float(expected.double().abs().sum()),
    }
    with open(a.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
