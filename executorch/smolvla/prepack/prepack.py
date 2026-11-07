#!/usr/bin/env python3
"""prepack.py: pre-pack an ExecuTorch model's XNNPACK linear weights on the host.

What it does
------------
Reads an ExecuTorch program (.pte) whose XNNPACK delegates keep their constants
in a .ptd (FlatTensor, keys are content hashes) or in the .pte as named data,
finds every fully-connected (linear) node with a named static weight, packs the
weights the way XNNPACK packs them on the Penguin board (RISC-V RVV, VLEN 256),
and writes one "XPWC v2" file. ExecuTorch's XNNPACK weights cache
reads that format (XNNWeightsCache::load_packed_cache from a file, or
use_packed_memory from a blob linked into the image, which is how the
smolvla app uses it: CMake option EXEC_PREPACK) and serves the packed bytes
in place, so the board skips packing at load time. A JSON sidecar lists
every entry and everything that is NOT covered.

The packing itself is done by the C++ helper xnn_prepack (built from the
XNNPACK fork's own reference packers, see below); this script finds the layers,
works out the cache names, runs the helper and writes the file.

Build the helper (one command; run it from this folder)
--------------------------------------------------------
  X=<XNNPACK fork>; E=<ExecuTorch checkout>; \\
  O=/path/to/outdir; mkdir -p $O; \\
  INC="-I$X -I$X/include -I$E/backends/xnnpack/third-party/pthreadpool/include -I$E/backends/xnnpack/third-party/FP16/include -I$E/backends/xnnpack/third-party/FXdiv/include"; \\
  c++ -std=c++17 -O2 -DXNN_LOG_LEVEL=0 -U__x86_64__ -U__amd64__ -U__x86_64 -U__amd64 -D__riscv=1 -c $INC xnn_prepack_support.cc -o $O/xnn_prepack_support.o \\
  && c++ -std=c++17 -O2 -DXNN_LOG_LEVEL=0 $INC xnn_prepack.cc $X/src/reference/packing.cc $O/xnn_prepack_support.o -o $O/xnn_prepack

(the pthreadpool / FP16 / FXdiv headers are only needed because xnnpack.h
includes them; nothing from them is linked.)

Run
---
  python prepack.py --pte X.pte --ptd X.ptd \\
      --xnnpack-dir <XNNPACK fork> \\
      --helper $O/xnn_prepack --out blob.xpwc --json blob.json
  # use the ExecuTorch Python environment (it imports executorch.exir) and put
  # flatc on PATH or pass --flatc. Without --ptd the constants are taken from
  # the .pte's own named data (older exports).
  # --strip-ptd-out stripped.ptd  also writes the .ptd without the raw weights the
  # blob replaces (keys used only as weight/bias of covered linears); the sidecar
  # JSON lists them under "stripped_ptd".

Target parameters (defaults are the Penguin board)
--------------------------------------------------
  --vlenb 32   VLEN 256 bits = 32 bytes. XNNPACK takes nr = 4 * vlenb / 4 = 32
               for bf16 and f32 gemm (src/configs/gemm-config.c, RISC-V branches
               of init_bf16_f32_gemm_config / init_f32_gemm_config_impl) and
               nr2 = 2 * vlenb / 4 = 16 (init_f32_gemm_nr2_config_impl).
  --zvfbfa 1   the build has XNNPACK_ENABLE_RISCV_ZVFBFA=ON and the hardware has
               bf16 vector arithmetic. It only selects the gemm microkernel
               (5x4v__rvvzvfbfa instead of 5x4v__rvv); nr, kr and the packers
               are the same either way, so it does not change any packed byte.
               It is recorded in the JSON only.

Format of the output (read from forks/executorch
backends/xnnpack/runtime/XNNWeightsCache.cpp, save_packed_index() and
load_packed_cache(), commit 5a527f1771): packed entries, then per entry
[name_len:u32][name][file_offset:u64][data_size:u64][seed:u32], then a 20-byte
footer [index_start:u64][entry_count:u32][magic:u32 = 0x58505743][version:u32 = 2],
all little endian. Each entry starts at a 64-byte aligned offset (zero padding
between entries). name = weight named key + bias named key (the bias part only
when the bias is a named constant), which is what XNNWeightsCache::look_up()
builds from the unpacked data pointers. data_size = aligned_total_weights_size.

Derived from
------------
  pandabon/XNNPACK (branch penguin, commit 7264d39dab): src/operators/
    fully-connected-nc.c, src/subgraph/fully-connected.c, src/configs/
    gemm-config.c, src/reference/packing.cc (see xnn_prepack.cc).
  pandabon/executorch (branch penguin-1.6, commit 5a527f1771): backends/xnnpack/
    runtime/{XNNHeader.cpp,XNNCompiler.cpp,XNNWeightsCache.cpp} (payload layout,
    named constants, cache format), backends/xnnpack/serialization/
    runtime_schema.fbs, extension/flat_tensor/serialize/serialize.py (.ptd).
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import OrderedDict, defaultdict

# XPWC constants: XNNWeightsCache.cpp kCacheMagic / kCacheVersion.
XPWC_MAGIC = 0x58505743
XPWC_VERSION = 2
ENTRY_ALIGN = 64  # requested by the task; the cache only needs file_offset < index_start

INVALID_ID = 0xFFFFFFFF  # XNN_INVALID_VALUE_ID (no bias)


def die(msg):
    print("prepack.py: error: " + msg, file=sys.stderr)
    sys.exit(1)


def git_head(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "log", "-1", "--format=%h"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reading the inputs
# ---------------------------------------------------------------------------

def load_pte(path):
    from executorch.exir._serialize._program import deserialize_pte_binary
    with open(path, "rb") as f:
        return deserialize_pte_binary(f.read())


class PtdSource:
    """Named constants from a .ptd (FlatTensor). Returns file references, no copies."""

    def __init__(self, path):
        from executorch.extension.flat_tensor.serialize.serialize import (
            FlatTensorHeader, _deserialize_to_flat_tensor)
        self.path = path
        with open(path, "rb") as f:
            data = f.read(1 << 20)  # header + flatbuffer are at the start
            hdr = FlatTensorHeader.from_bytes(data[8:])
            if not hdr.is_valid():
                die("%s: not a FlatTensor file (bad header)" % path)
            need = hdr.flatbuffer_offset + hdr.flatbuffer_size
            if need > len(data):
                f.seek(0)
                data = f.read(need)
            flat = _deserialize_to_flat_tensor(data[:need])
        self.entries = {}
        for nd in flat.named_data:
            seg = flat.segments[nd.segment_index]
            self.entries[nd.key] = (hdr.segment_base_offset + seg.offset, seg.size)

    def has(self, key):
        return key in self.entries

    def ref(self, key):
        off, size = self.entries[key]
        return (self.path, off, size)


class PteNamedSource:
    """Named constants stored inside the .pte (NamedDataStoreOutput.pte_data).

    The helper reads from files, so each needed constant is copied once into a
    scratch file on demand.
    """

    def __init__(self, ptefile, scratch_path):
        nd = ptefile.named_data
        self.map = {}
        if nd is not None:
            for key, entry in nd.pte_data.items():
                self.map[key] = nd.buffers[entry.buffer_index]
        self.scratch_path = scratch_path
        self.scratch = open(scratch_path, "wb")
        self.pos = 0
        self.done = {}

    def has(self, key):
        return key in self.map

    def ref(self, key):
        if key not in self.done:
            data = bytes(self.map[key])
            self.scratch.write(data)
            self.done[key] = (self.scratch_path, self.pos, len(data))
            self.pos += len(data)
        return self.done[key]

    def close(self):
        self.scratch.close()


def xnn_payload_flatbuffer(payload):
    """XNNHeader (XNNHeader.cpp::Parse): magic at 4, then u16 header length,
    u32 flatbuffer offset, u32 flatbuffer size, u32 constant data offset,
    u64 constant data size. Returns (flatbuffer bytes, constant data bytes)."""
    if payload[4:8] != b"XH00":
        die("delegate payload has no XNNPACK header (magic %r)" % payload[4:8])
    fb_off, fb_size, cd_off = struct.unpack_from("<III", payload, 10)
    cd_size = struct.unpack_from("<Q", payload, 22)[0]
    return payload[fb_off:fb_off + fb_size], payload[cd_off:cd_off + cd_size]


def flatbuffer_to_json(flatc, schema, fb_bytes, workdir, tag):
    binp = os.path.join(workdir, tag + ".bin")
    with open(binp, "wb") as f:
        f.write(fb_bytes)
    # --raw-binary: the XNNPACK graph's file identifier is "XN01" in the
    # payload, while the schema says "XN00".
    subprocess.check_call(
        [flatc, "--json", "--strict-json", "--defaults-json", "--raw-binary",
         "-o", workdir, schema, "--", binp])
    with open(os.path.join(workdir, tag + ".json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Walking the graphs
# ---------------------------------------------------------------------------

def tensor_of(graph, vid):
    """Return (XNNTensorValue dict, is_quantized) for a value id."""
    v = graph["xvalues"][vid]
    if v["xvalue_union_type"] == "XNNQuantizedTensorValue":
        return v["xvalue_union"]["tensor_value"], True
    return v["xvalue_union"], False


def const_info(graph, tv):
    """('none'|'named'|'inline'|'inline_buffer', key_or_None, size)."""
    idx = tv["constant_buffer_idx"]
    if idx == 0:
        return "none", None, 0
    cd = graph.get("constant_data") or []
    if idx < len(cd):
        ent = cd[idx]
        if ent["named_key"]:
            return "named", ent["named_key"], ent["size"]
        return "inline", (ent["offset"], ent["size"]), ent["size"]
    return "inline_buffer", None, 0


def dt(tv):
    return tv["datatype"].replace("xnn_datatype_", "")


def choose_fc_op(in_dt, filt_dt, out_dt, bias_dt, filt_static):
    """Mirror of get_fully_connected_op_type() (src/subgraph/fully-connected.c)
    for the cases this tool packs. Returns (op, reason_if_not).

    ExecuTorch (XNNCompiler.cpp defineFullyConnectedNode) defines a bf16 x bf16
    -> bf16 linear with an fp32 intermediate output, so XNNPACK sees
    bf16 x bf16 -> fp32 and picks fc_type_bf16_bf16_f32."""
    if in_dt == "bf16" and filt_dt == "bf16" and out_dt in ("bf16", "fp32"):
        if bias_dt not in (None, "fp32"):
            return None, "bf16 FC with %s bias (xnn_create_fully_connected_nc_bf16_f32 takes a float bias)" % bias_dt
        return "bf16_f32", None
    if in_dt == "fp32" and filt_dt == "fp32" and out_dt == "fp32":
        if not filt_static:
            return None, "dynamic fp32 weights"
        if bias_dt not in (None, "fp32"):
            return None, "fp32 FC with %s bias" % bias_dt
        return "f32", None
    return None, "FC datatypes in=%s filter=%s out=%s are not packed by this tool" % (in_dt, filt_dt, out_dt)


def scan_method(method, graph, cd_bytes, jobs, uncovered, fc_issues):
    """Collect FC jobs and uncovered static-weight nodes of one method."""
    for ni, node in enumerate(graph["xnodes"]):
        ntype = node["xnode_union_type"]
        x = node["xnode_union"]
        where = {"method": method, "node_index": ni, "debug_handle": node.get("debug_handle")}

        if ntype in ("XNNConv2d", "XNNDepthwiseConv2d", "XNNConvTranspose2d"):
            tv, _ = tensor_of(graph, x["filter_id"])
            kind, key, _ = const_info(graph, tv)
            if kind != "none":
                uncovered.append(dict(where, op=ntype, reason="static filter; stays packed on the board "
                                      "(convolution-nhwc.c / deconvolution-nhwc.c use the weights cache)",
                                      filter_key=key if kind == "named" else None))
            continue

        if ntype == "XNNBatchMatrixMultiply":
            for which in ("input1_id", "input2_id"):
                tv, _ = tensor_of(graph, x[which])
                kind, key, _ = const_info(graph, tv)
                if kind != "none":
                    uncovered.append(dict(where, op=ntype, reason="static %s; stays packed on the board "
                                          "(batch-matrix-multiply-nc.c uses the weights cache)" % which,
                                          key=key if kind == "named" else None))
            continue

        if ntype != "XNNFullyConnected":
            continue

        in_tv, in_q = tensor_of(graph, x["input1_id"])
        f_tv, f_q = tensor_of(graph, x["filter_id"])
        o_tv, o_q = tensor_of(graph, x["output_id"])
        f_kind, f_key, f_size = const_info(graph, f_tv)
        has_bias = x["bias_id"] != INVALID_ID
        b_tv = b_kind = b_key = None
        if has_bias:
            b_tv, _ = tensor_of(graph, x["bias_id"])
            b_kind, b_key, b_size = const_info(graph, b_tv)

        info = dict(where, flags=x["flags"], filter_dims=f_tv["dims"],
                    input_dtype=dt(in_tv), filter_dtype=dt(f_tv), output_dtype=dt(o_tv),
                    bias_dtype=dt(b_tv) if b_tv else None)

        if f_kind == "none":
            fc_issues.append(dict(info, problem="filter is not a constant (dynamic weights, not cached)"))
            continue
        if in_q or f_q or o_q:
            uncovered.append(dict(info, op=ntype, reason="quantized FC; not packed by this tool"))
            continue
        if f_kind != "named":
            fc_issues.append(dict(info, problem="weight is not a named constant (%s): the cache cannot hit by name" % f_kind))
            continue
        if has_bias and b_kind == "none":
            uncovered.append(dict(info, op=ntype, reason="bias is not a constant; the FC is not static"))
            continue

        op, why = choose_fc_op(dt(in_tv), dt(f_tv), dt(o_tv), dt(b_tv) if b_tv else None, True)
        if op is None:
            uncovered.append(dict(info, op=ntype, reason=why))
            continue
        if len(f_tv["dims"]) != 2:
            uncovered.append(dict(info, op=ntype, reason="filter is not 2-D"))
            continue

        bias_named = has_bias and b_kind == "named"
        if has_bias and not bias_named:
            fc_issues.append(dict(info, problem="bias is an inline constant (not named): cache name is the weight key only; "
                                  "two layers sharing a weight but not a bias would collide"))
        jobs.append(dict(
            method=method, node_index=ni, op=op, flags=x["flags"],
            d0=f_tv["dims"][0], dlast=f_tv["dims"][-1],
            wkey=f_key, wsize=f_size,
            has_bias=has_bias, bkey=b_key if bias_named else None,
            binline=(b_key if (has_bias and b_kind == "inline") else None),
            bsize=b_size if has_bias else 0, cdata=cd_bytes,
            filter_dtype=dt(f_tv), input_dtype=dt(in_tv), output_dtype=dt(o_tv),
            # where this job reads its constants, for --strip-ptd-out (graph object + (node, field) slots)
            graph=graph, slots=[(ni, "filter_id")] + ([(ni, "bias_id")] if bias_named else [])))


# ---------------------------------------------------------------------------
# Dropping the raw weights the blob replaces from the .ptd (--strip-ptd-out)
# ---------------------------------------------------------------------------

def _value_ids(obj):
    """All value ids a node (or graph) refers to: every field named *_id or *_ids."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("_id") and isinstance(v, int):
                yield k, v
            elif k.endswith("_ids") and isinstance(v, list):
                for i in v:
                    yield k, i
            elif isinstance(v, (dict, list)):
                yield from _value_ids(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _value_ids(v)


def strippable_keys(graphs, jobs):
    """Named constants whose EVERY use (all graphs, all methods) is the weight or the named
    bias of a covered fully-connected node. Any other use (a convolution, a batch matmul,
    another op, a graph input or output) keeps the key. Returns the set of keys."""
    covered = defaultdict(set)  # id(graph) -> {(node_index, field)}
    for j in jobs:
        covered[id(j["graph"])].update(j["slots"])
    uses, cov_uses = defaultdict(int), defaultdict(int)
    for graph in graphs:
        def note(vid, slot):
            if vid == INVALID_ID or vid >= len(graph["xvalues"]):
                return
            tv, _ = tensor_of(graph, vid)
            kind, key, _ = const_info(graph, tv)
            if kind == "named":
                uses[key] += 1
                if slot in covered[id(graph)]:
                    cov_uses[key] += 1
        for ni, node in enumerate(graph["xnodes"]):
            for field, vid in _value_ids(node["xnode_union"]):
                note(vid, (ni, field))
        for field, vid in _value_ids({"input_ids": graph.get("input_ids") or [],
                                      "output_ids": graph.get("output_ids") or []}):
            note(vid, None)
    return {k for k, c in cov_uses.items() if c > 0 and uses[k] == c}


def strip_ptd(src_path, dst_path, strip):
    """Write src_path without the keys in `strip`, with ExecuTorch's own FlatTensor
    serializer, then read both back and check every kept key. Returns a summary dict."""
    from math import gcd
    from executorch.exir._serialize._cord import Cord
    from executorch.exir._serialize.data_serializer import DataEntry, DataPayload
    from executorch.extension.flat_tensor.serialize.serialize import (
        FlatTensorConfig, FlatTensorHeader, FlatTensorSerializer, _deserialize_to_flat_tensor)

    with open(src_path, "rb") as f:
        raw = f.read()
    hdr = FlatTensorHeader.from_bytes(raw[8:])
    flat = _deserialize_to_flat_tensor(raw[:hdr.flatbuffer_offset + hdr.flatbuffer_size])
    # The reader drops alignment (every entry comes back with 1), so recover the writer's
    # segment_alignment: the segments are placed at multiples of it, and so is the base.
    g = hdr.segment_base_offset
    for seg in flat.segments:
        g = gcd(g, seg.offset)
    align = g & -g if g else 128
    ser = FlatTensorSerializer(FlatTensorConfig(segment_alignment=align))
    old = ser.deserialize(Cord(raw))
    missing = [k for k in strip if k not in old.named_data]
    if missing:
        die("--strip-ptd-out: %d keys to strip are not in %s" % (len(missing), src_path))
    kept = {k: e for k, e in old.named_data.items() if k not in strip}
    used = sorted({e.buffer_index for e in kept.values()})
    new_index = {b: i for i, b in enumerate(used)}
    new = DataPayload(
        buffers=[old.buffers[b] for b in used],
        named_data={k: DataEntry(buffer_index=new_index[e.buffer_index], alignment=1,
                                 tensor_layout=e.tensor_layout) for k, e in kept.items()})
    blob = ser.serialize(new)
    with open(dst_path, "wb") as f:
        f.write(bytes(blob))
    del blob

    # check: read the new file back with the same deserializer
    with open(dst_path, "rb") as f:
        back = ser.deserialize(Cord(f.read()))
    if set(back.named_data) != set(kept):
        die("stripped .ptd: key set differs from the expected kept set")
    for k, e in kept.items():
        b = back.named_data[k]
        if bytes(back.buffers[b.buffer_index]) != bytes(old.buffers[e.buffer_index]):
            die("stripped .ptd: bytes of kept key %s differ from the original" % k)
        if b.tensor_layout != e.tensor_layout:
            die("stripped .ptd: tensor layout of kept key %s differs from the original" % k)
    kept_bytes = sum(len(old.buffers[b]) for b in used)
    all_bytes = sum(len(b) for b in old.buffers)
    return dict(path=os.path.abspath(dst_path), segment_alignment=align,
                keys_total=len(old.named_data), keys_kept=len(kept), keys_stripped=len(strip),
                bytes_kept=kept_bytes, bytes_stripped=all_bytes - kept_bytes,
                file_bytes_before=len(raw), file_bytes_after=os.path.getsize(dst_path),
                stripped_keys=sorted(strip), kept_keys_checked=len(kept))


# ---------------------------------------------------------------------------
# Self-check: unpack a packed entry independently of the C++ packer
# ---------------------------------------------------------------------------

def unpack_entry(np, buf, ent):
    """Rebuild (weights [oc, ic] as raw uint16/uint32, bias float32 or None)
    from the packed bytes. Layout for kr = sr = 1 (both RVV recipes):
    per nr block: nr f32 biases, then kc rows of nr weights (only the first
    `n` of each row are meaningful in the last, partial block)."""
    nr, ic, oc = ent["nr"], ent["ic"], ent["oc"]
    elt = 2 if ent["op"] == "bf16_f32" else 4
    wdt = np.uint16 if elt == 2 else np.uint32
    block = nr * 4 + ic * nr * elt
    W = np.zeros((oc, ic), dtype=wdt)
    B = np.zeros(oc, dtype=np.float32)
    nblocks = (oc + nr - 1) // nr
    for bi in range(nblocks):
        n = min(nr, oc - bi * nr)
        base = bi * block
        B[bi * nr:bi * nr + n] = np.frombuffer(buf, dtype=np.float32, count=n, offset=base)
        rows = np.frombuffer(buf, dtype=wdt, count=ic * nr, offset=base + nr * 4).reshape(ic, nr)
        W[bi * nr:bi * nr + n, :] = rows[:, :n].T
    return W, B


def selfcheck(np, blob_path, entries, sources_fn, count):
    """Read the blob back through the same index ExecuTorch reads and compare."""
    with open(blob_path, "rb") as f:
        f.seek(-20, os.SEEK_END)
        index_start, n, magic, ver = struct.unpack("<QIII", f.read(20))
        assert magic == XPWC_MAGIC and ver == XPWC_VERSION, "bad footer"
        f.seek(index_start)
        raw = f.read()
    cur = 0
    index = {}
    for _ in range(n):
        (nl,) = struct.unpack_from("<I", raw, cur); cur += 4
        name = raw[cur:cur + nl].decode(); cur += nl
        off, size, seed = struct.unpack_from("<QQI", raw, cur); cur += 20
        index[name] = (off, size, seed)
    assert cur == len(raw) - 20, "index does not end where the footer starts"
    mm = np.memmap(blob_path, dtype=np.uint8, mode="r")
    results = []
    todo = entries if count is None else pick_check_entries(entries, count)
    for ent in todo:
        off, size, seed = index[ent["name"]]
        assert (off, size, seed) == (ent["offset"], ent["data_size"], ent["seed"]), "index mismatch"
        buf = bytes(mm[off:off + size])
        W, B = unpack_entry(np, buf, ent)
        wbytes, bbytes = sources_fn(ent)
        wdt = np.uint16 if ent["op"] == "bf16_f32" else np.uint32
        orig = np.frombuffer(wbytes, dtype=wdt)
        if ent["transpose"]:  # stored [ic, oc]
            orig = orig.reshape(ent["ic"], ent["oc"]).T
        else:
            orig = orig.reshape(ent["oc"], ent["ic"])
        ok_w = bool(np.array_equal(W, orig))
        if ent["has_bias"]:
            ok_b = bytes(B.tobytes()) == bbytes
        else:
            ok_b = bool(not B.any())
        # bytes after packed_weights_size must be zero padding
        pad_ok = not any(buf[ent["packed_weights_size"]:])
        results.append(dict(name=ent["name"], op=ent["op"], ic=ent["ic"], oc=ent["oc"],
                            partial_nr_block=ent["partial_nr_block"],
                            weights_equal=ok_w, bias_equal=ok_b, tail_padding_zero=pad_ok))
    return results


def pick_check_entries(entries, count):
    """First entry, then the first partial-nr one, the first f32 one, then fill."""
    chosen = []

    def add(e):
        if e is not None and e not in chosen:
            chosen.append(e)
    add(entries[0] if entries else None)
    add(next((e for e in entries if e["partial_nr_block"]), None))
    add(next((e for e in entries if e["op"] == "f32"), None))
    for e in entries:
        if len(chosen) >= count:
            break
        add(e)
    return chosen[:count]


# ---------------------------------------------------------------------------

def find_default_schema():
    try:
        import executorch
        for base in list(executorch.__path__):
            for rel in ("backends/xnnpack/serialization/runtime_schema.fbs",
                        "../../backends/xnnpack/serialization/runtime_schema.fbs"):
                p = os.path.normpath(os.path.join(base, rel))
                if os.path.exists(p):
                    return p
    except Exception:
        pass
    return None


def find_default_flatc():
    p = shutil.which("flatc")
    if p:
        return p
    try:
        import executorch
        for base in list(executorch.__path__):
            c = os.path.join(base, "data", "bin", "flatc")
            if os.path.exists(c):
                return c
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pte", required=True)
    ap.add_argument("--ptd", help="external constants (FlatTensor); default: named data inside the .pte")
    ap.add_argument("--xnnpack-dir", required=True, help="XNNPACK fork checkout (used only for the commit id in the JSON; the helper was built from it)")
    ap.add_argument("--helper", required=True, help="built xnn_prepack binary")
    ap.add_argument("--out", required=True, help="XPWC v2 blob to write")
    ap.add_argument("--json", required=True, help="sidecar JSON to write")
    ap.add_argument("--vlenb", type=int, default=32, help="vector register bytes (VLEN/8); Penguin: 32")
    ap.add_argument("--zvfbfa", type=int, default=1, choices=(0, 1), help="Zvfbfa kernels on (recorded only; no effect on packing)")
    ap.add_argument("--flatc", default=None, help="flatc binary (default: PATH, then the executorch package)")
    ap.add_argument("--schema", default=None, help="runtime_schema.fbs (default: from the executorch package)")
    ap.add_argument("--strip-ptd-out", default=None, metavar="PATH",
                    help="also write a copy of --ptd without the keys that only the covered linears use "
                         "(the blob replaces them); the board then holds one copy of each weight")
    ap.add_argument("--check", default="all", help="self-check: 'all', a number, or 0 to skip (default all)")
    ap.add_argument("--keep-work", action="store_true", help="keep the scratch directory")
    args = ap.parse_args()
    if args.strip_ptd_out and not args.ptd:
        die("--strip-ptd-out needs --ptd")

    flatc = args.flatc or find_default_flatc() or die("flatc not found; pass --flatc")
    schema = args.schema or find_default_schema() or die("runtime_schema.fbs not found; pass --schema")

    import numpy as np

    work = tempfile.mkdtemp(prefix="xnn_prepack_")
    try:
        # --- constants as the helper sees them (from the fork's headers) ---
        consts = {}
        for line in subprocess.check_output(
                [args.helper, "--vlenb", str(args.vlenb), "--zvfbfa", str(args.zvfbfa),
                 "--print-constants"], text=True).splitlines():
            k, v = line.split("=")
            consts[k] = int(v)

        # --- load the program ---
        ptef = load_pte(args.pte)
        program = ptef.program
        if args.ptd:
            src = PtdSource(args.ptd)
            src_desc = "ptd:" + args.ptd
            scratch = None
        else:
            scratch = os.path.join(work, "pte_named_data.bin")
            src = PteNamedSource(ptef, scratch)
            src_desc = "pte named data"

        methods = OrderedDict()
        for plan in program.execution_plan:
            for di, d in enumerate(plan.delegates):
                if d.id != "XnnpackBackend":
                    continue
                ref = d.processed
                payload = bytes(program.backend_delegate_data[ref.index].data)
                fb, cdata = xnn_payload_flatbuffer(payload)
                tag = "%s_d%d" % (plan.name, di)
                graph = flatbuffer_to_json(flatc, schema, fb, work, tag)
                methods.setdefault(plan.name, []).append((graph, cdata))

        # --- find layers ---
        all_jobs, uncovered, fc_issues = [], [], []
        n_nodes = defaultdict(int)
        for mname, graphs in methods.items():
            for graph, cdata in graphs:
                for node in graph["xnodes"]:
                    n_nodes[(mname, node["xnode_union_type"])] += 1
                scan_method(mname, graph, cdata, all_jobs, uncovered, fc_issues)

        # --- missing constants in the data source ---
        for j in all_jobs:
            for key in (j["wkey"], j["bkey"]):
                if key and not src.has(key):
                    die("constant %s (method %s) is not in the %s" % (key, j["method"], src_desc))
            if args.ptd and src.entries[j["wkey"]][1] != j["wsize"]:
                die("constant %s: size in the .ptd differs from the graph" % j["wkey"])

        # --- de-duplicate: one packing per distinct input of the cache seed ---
        uniq = OrderedDict()
        for j in all_jobs:
            pre = (j["op"], j["flags"] & consts["XNN_FLAG_TRANSPOSE_WEIGHTS"], j["d0"], j["dlast"],
                   j["wkey"], j["bkey"], j["binline"], j["has_bias"])
            u = uniq.setdefault(pre, dict(j, methods=[]))
            if j["method"] not in u["methods"]:
                u["methods"].append(j["method"])
            u["count"] = u.get("count", 0) + 1
        jobs = list(uniq.values())

        # --- job file for the helper ---
        extra_path = os.path.join(work, "inline_bias.bin")
        extra = open(extra_path, "wb")
        extra_pos = 0
        jobs_tsv = os.path.join(work, "jobs.tsv")
        with open(jobs_tsv, "w") as f:
            for idx, j in enumerate(jobs):
                wfile, woff, wlen = src.ref(j["wkey"])
                if j["has_bias"]:
                    if j["bkey"]:
                        bfile, boff, blen = src.ref(j["bkey"])
                    else:  # inline bias: copy it out of the delegate payload
                        off, size = j["binline"]
                        extra.write(j["cdata"][off:off + size]); extra.flush()
                        bfile, boff, blen = extra_path, extra_pos, size
                        extra_pos += size
                else:
                    bfile, boff, blen = "-", 0, 0
                j["_wref"] = (wfile, woff, wlen)
                j["_bref"] = (bfile, boff, blen)
                f.write("\t".join(map(str, [idx, j["op"], j["flags"], j["d0"], j["dlast"],
                                            wfile, woff, wlen, bfile, boff, blen])) + "\n")
        extra.close()
        if scratch:
            src.close()

        # --- run the helper ---
        data_bin = os.path.join(work, "packed.bin")
        res_tsv = os.path.join(work, "result.tsv")
        subprocess.check_call([args.helper, "--vlenb", str(args.vlenb), "--zvfbfa", str(args.zvfbfa),
                               "--jobs", jobs_tsv, "--out-data", data_bin, "--out-tsv", res_tsv])
        rows = []
        with open(res_tsv) as f:
            hdr = f.readline().rstrip("\n").split("\t")
            for line in f:
                rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
        assert len(rows) == len(jobs)

        # --- entries ---
        OPNAME = {"bf16_f32": "fully_connected_nc_bf16_f32", "f32": "fully_connected_nc_f32"}
        entries = []
        for j, r in zip(jobs, rows):
            name = j["wkey"] + (j["bkey"] or "")
            entries.append(dict(
                name=name, weight_key=j["wkey"], bias_key=j["bkey"], has_bias=j["has_bias"],
                bias_named=bool(j["bkey"]),
                methods=j["methods"], uses=j["count"],
                op=j["op"], op_type=OPNAME[j["op"]], operator_type_value=int(r["operator_type"]),
                ic=int(r["ic"]), oc=int(r["oc"]), transpose=bool(int(r["transpose"])),
                nr=int(r["nr"]), kr=int(r["kr"]), sr=int(r["sr"]),
                extra_weights_bytes=int(r["extra_weights_bytes"]),
                use_nr2=bool(int(r["use_nr2"])),
                packed_weights_size=int(r["packed_weights_size"]),
                data_size=int(r["aligned_total_weights_size"]),
                offset=int(r["data_offset"]), seed=int(r["seed"]),
                fingerprint_id=int(r["fingerprint_id"]), fingerprint_name=r["fingerprint_name"],
                partial_nr_block=bool(int(r["partial_nr_block"])),
                filter_dtype=j["filter_dtype"], input_dtype=j["input_dtype"],
                _job=j))

        # Two entries with the same (name, seed) but different shapes would
        # collide in the cache (name is the map key there).
        seen = {}
        warnings = []
        for e in entries:
            k = e["name"]
            if k in seen:
                warnings.append("cache name %s used by two entries (seeds %08x and %08x): ExecuTorch keeps only one per name"
                                % (k[:16] + "...", seen[k]["seed"], e["seed"]))
            seen[k] = e

        # --- write the blob ---
        with open(data_bin, "rb") as f:
            data = f.read()
        with open(args.out, "wb") as f:
            f.write(data)
            index_start = len(data)
            idx = bytearray()
            for e in entries:
                nb = e["name"].encode()
                idx += struct.pack("<I", len(nb)) + nb
                idx += struct.pack("<QQI", e["offset"], e["data_size"], e["seed"])
            idx += struct.pack("<QIII", index_start, len(entries), XPWC_MAGIC, XPWC_VERSION)
            f.write(idx)
        blob_size = os.path.getsize(args.out)

        # --- self-check ---
        def sources_fn(ent):
            j = ent["_job"]
            wf, wo, wl = j["_wref"]
            with open(wf, "rb") as f:
                f.seek(wo); wb = f.read(wl)
            bb = b""
            if j["has_bias"]:
                bf, bo, bl = j["_bref"]
                with open(bf, "rb") as f:
                    f.seek(bo); bb = f.read(bl)
            return wb, bb
        check = None
        if args.check != "0":
            cnt = None if args.check == "all" else int(args.check)
            check = selfcheck(np, args.out, entries, sources_fn, cnt)

        # --- stripped .ptd ---
        stripped = None
        if args.strip_ptd_out:
            graphs = [g for gl in methods.values() for g, _ in gl]
            stripped = strip_ptd(args.ptd, args.strip_ptd_out, strippable_keys(graphs, all_jobs))

        # --- sidecar JSON ---
        per_method = OrderedDict()
        for m in methods:
            per_method[m] = defaultdict(int)
        for e in entries:
            for m in e["methods"]:
                per_method[m][e["op_type"]] += 1
        info = dict(
            tool="prepack.py",
            inputs=dict(pte=os.path.abspath(args.pte), constants_from=src_desc),
            target=dict(vlenb=args.vlenb, zvfbfa=bool(args.zvfbfa),
                        note="zvfbfa selects the gemm kernel only; it does not change packed bytes"),
            helper_constants=consts,
            sources=dict(xnnpack_commit=git_head(args.xnnpack_dir)),
            totals=dict(entries=len(entries), data_bytes=index_start, blob_bytes=blob_size,
                        sum_data_size=sum(e["data_size"] for e in entries),
                        entries_with_partial_nr_block=sum(e["partial_nr_block"] for e in entries)),
            entries_per_method={m: dict(v) for m, v in per_method.items()},
            entries_per_method_total={m: sum(v.values()) for m, v in per_method.items()},
            node_type_counts={m: {t: c for (mm, t), c in sorted(n_nodes.items()) if mm == m} for m in methods},
            entries=[{k: v for k, v in e.items() if not k.startswith("_")} for e in entries],
            uncovered_static_weight_nodes=uncovered,
            fc_not_cacheable_by_name=fc_issues,
            warnings=warnings,
            stripped_ptd=stripped,
            selfcheck=check,
            note_partial_nr_block=("In a partial last nr block the packer skips the padding (it does not write it). "
                                   "This blob has zeros there; on the board those bytes are whatever the allocator "
                                   "returned (extra_weights_bytes is 0, so create_fully_connected_nc does not memset). "
                                   "They are never read as results for real output channels."),
        )
        with open(args.json, "w") as f:
            json.dump(info, f, indent=1)

        # --- console summary ---
        print("entries: %d, blob: %d bytes (%s)" % (len(entries), blob_size, args.out))
        for m, v in per_method.items():
            print("  %-13s %s" % (m, dict(v)))
        print("uncovered static-weight nodes: %d; FCs not cacheable by name: %d; warnings: %d"
              % (len(uncovered), len(fc_issues), len(warnings)))
        if stripped:
            print("stripped .ptd: %d of %d keys dropped, %d bytes stripped, %d kept (%d -> %d file bytes); %d kept keys re-read identical (%s)"
                  % (stripped["keys_stripped"], stripped["keys_total"], stripped["bytes_stripped"], stripped["bytes_kept"],
                     stripped["file_bytes_before"], stripped["file_bytes_after"], stripped["kept_keys_checked"], args.strip_ptd_out))
        if check is not None:
            bad = [c for c in check if not (c["weights_equal"] and c["bias_equal"] and c["tail_padding_zero"])]
            print("self-check: %d entries read back, %d mismatches" % (len(check), len(bad)))
            if bad:
                sys.exit(2)
    finally:
        if args.keep_work:
            print("scratch kept in", work)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
