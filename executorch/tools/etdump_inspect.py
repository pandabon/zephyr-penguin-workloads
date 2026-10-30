#!/usr/bin/env python3
"""Turn the ETDump a Penguin run printed on the console into per-operator tables (ExecuTorch Inspector). Generic for any ExecuTorch
app built with an event tracer + ETDumpGen that prints the dump as the SmolVLA runner does (executorch/smolvla, -DEXEC_ETDUMP=ON).

  etdump_inspect.py RUN.log [--etrecord X.etrecord] [--out PREFIX] [--csv] [--cpu-hz 50000000] [--per-run]

1. cuts the base64 block between "ETDUMP BEGIN ..." and "ETDUMP END crc32=..." out of RUN.log and writes the binary dump to PREFIX.etdp.
   Console noise is tolerated (CR line ends, pty/screen header and footer lines, stray text between the markers); the decoded length is
   checked against size= on the BEGIN line and the zlib CRC32 against crc32= on the END line (logs without crc32 still work), and a
   mismatch stops the script;
2. loads it with ExecuTorch's Inspector (devtools/inspector), with the ETRecord when given, reading the times as CYCLES (the runner's
   PAL clock is mcycle: core cycles on the FPGA / FireSim, instructions on Spike);
3. writes PREFIX.csv (one row per event of every method and run) and PREFIX.md (summary, top rows, one table per method), and prints
   the same Markdown.

One dump can hold many methods and runs (step and chain modes of the runner). The dump does not name the method of a run: the runner
prints "ETDUMP RUNS <label> <label> ..." (one label per execute(), in order; a trailing _<n> is the run number). The Inspector groups
runs with identical events into one block (mean/min/max over the runs); this script gives every block the method of the labels it
covers. Without a RUNS line the dump is one method, called "forward" (or the only method of the ETRecord).

CSV columns: method, run ("1", or "mean" for a block of several runs; --per-run adds one row per run), runs, row_kind, event, ticks (mean),
ticks_min, ticks_max, ticks_per_run, lowered_to, aten_ops, debug_handles, python_source, linked.
row_kind: "portable op" (OPERATOR_CALL: a portable kernel call incl. the framework's overhead), "portable kernel" (the native_call_<op>
row just before it: the kernel alone, inside the op row), "xnnpack operator" (an operator the XNNPACK backend logged itself, nested in
the delegate call after it), "delegate call" (one whole XNNPACK delegate), "method total" (Method::execute), "load" (Method::init etc.).
lowered_to: "portable:<kernel name>" or "xnnpack:<XNNPACK operator name>" ("xnnpack:(delegate call)" for the whole delegate row).
aten_ops / debug_handles / python_source come from the ETRecord (the delegate map links "[n<i>]" XNNPACK rows to ATen nodes); an XNNPACK row
without "[n<i>]" is an operator XNNPACK inserted itself (no ATen op). python_source: the last stack frame outside torch / site-packages
(the model file, e.g. expert_layer.py), else the last frame of the stack; --source-hint SUBSTR prefers frames whose path contains it.

Needs the python of the ExecuTorch checkout the .pte was exported with, e.g. smolvla-explore/.venv-main/bin/python.
The ETRecord comes from the export (smolvla-explore/export_*.py --etrecord); a multi-method ETRecord is split per method here
(the Inspector itself only reads the method "forward").

Credits: the base64 block and the "inspector_cli ... --source_time_scale cycles --target_time_scale cycles" call follow ExecuTorch
examples/arm/executor_runner/arm_executor_runner.cpp (write_etdump) and devtools/inspector/inspector_cli.py @ 681c3edb47
(Copyright Meta Platforms and Arm Limited, BSD-style licence). The per-method resolution mirrors Inspector._consume_etrecord
(devtools/inspector/_inspector.py @ ExecuTorch penguin-1.6). Written for this repo otherwise.

Reading the numbers: with an Inspector time scale of CYCLES nothing is converted, so every time is in mcycle ticks. Divide by
the core clock for seconds (50 MHz on the FPGA board: 1 tick = 20 ns). On Spike mcycle == minstret, so a "time" is an instruction count.
"""
import argparse
import base64
import re
import sys
import zlib

import pandas as pd

from executorch.devtools import Inspector
from executorch.devtools.etrecord import ETRecord, parse_etrecord
from executorch.devtools.inspector import TimeScale
from executorch.devtools.inspector._inspector_utils import (EDGE_DIALECT_GRAPH_KEY, FORWARD, create_debug_handle_to_op_node_mapping,
                                                           gen_graphs_from_etrecord)

KIND = {"operator": "portable op", "kernel": "portable kernel", "xnn-op": "xnnpack operator", "delegate": "delegate call",
        "execute": "method total", "load": "load"}


def extract_etdump(log_text):
    """Return (bytes, header line, run labels, crc32 or None) of the last ETDUMP block in the log."""
    log_text = log_text.replace("\r", "")   # console logs from a pty / the UART have CRLF line ends
    m = list(re.finditer(r"^ETDUMP BEGIN ([^\n]*)\n(.*?)^ETDUMP END(?: crc32=([0-9a-fA-F]{8}))?[ \t]*$", log_text, re.S | re.M))
    if not m:
        sys.exit("no 'ETDUMP BEGIN ... ETDUMP END' block in the log (was the app built with -DEXEC_ETDUMP=ON, and did the run finish?)")
    hdr, body, crc = m[-1].group(1), m[-1].group(2), m[-1].group(3)
    pre = log_text[:m[-1].start()]
    runs = re.findall(r"^ETDUMP RUNS ([^\n]*)$", pre, re.M)
    labels = runs[-1].split() if runs else []
    m_size = re.search(r"size=(\d+)", hdr)
    if not m_size:
        sys.exit("ETDUMP BEGIN line has no size=")
    size = int(m_size.group(1))
    need = (size + 2) // 3 * 4                      # base64 characters of the dump
    # data lines are pure base64 and 120 characters long (the last one shorter); anything else inside the block is console noise
    lines = [ln.strip() for ln in body.split("\n")]
    lines = [ln for ln in lines if ln and re.fullmatch(r"[A-Za-z0-9+/=]+", ln)]
    b64, k = "", 0
    while len(b64) < need and k < len(lines):
        if len(lines[k]) == 120 or need - len(b64) == len(lines[k]):
            b64 += lines[k]
        k += 1
    try:
        data = base64.b64decode(b64)
    except Exception as e:  # noqa
        sys.exit("ETDump base64 does not decode: %s" % e)
    if len(data) != size:
        sys.exit("ETDump size mismatch: header says %d bytes, decoded %d (lost or added console characters?)" % (size, len(data)))
    if crc is not None and zlib.crc32(data) & 0xFFFFFFFF != int(crc, 16):
        sys.exit("ETDump CRC32 mismatch: END line says %s, decoded bytes give %08x (corrupted console text)" % (crc, zlib.crc32(data) & 0xFFFFFFFF))
    return data, hdr, labels, crc


def best_frame(stack_traces, hint=""):
    """Python source of the op: the last 'File "...", line N, in fn' of its stack that is not torch's / site-packages' own (the model
    file); a frame whose path contains `hint` wins; else the last frame of the stack."""
    best = None
    for tr in (stack_traces or {}).values():
        frames = re.findall(r'File "([^"]+)", line (\d+), in (\S+)\n\s*(.*)', tr)
        if not frames:
            continue
        fmt = lambda f: "%s:%s %s" % (f[0].rsplit("/", 1)[-1], f[1], f[3].strip())
        mine = [f for f in frames if hint and hint in f[0]] or [f for f in frames if "site-packages" not in f[0] and "/torch/" not in f[0]]
        cand = fmt(mine[-1] if mine else frames[-1])
        if mine:
            return cand
        best = best or cand
    return best or ""


def kind_of(e):
    """What a row is (Inspector naming): OPERATOR_CALL = a portable kernel call incl. the framework's overhead (the
    `native_call_<op>.out` row just before it is the kernel alone); DELEGATE_CALL = one whole delegate (here XNNPACK);
    rows with is_delegated_op are the operators the XNNPACK backend logged itself, nested in the DELEGATE_CALL after them."""
    if e.is_delegated_op:
        return "xnn-op"
    if e.name == "DELEGATE_CALL":
        return "delegate"
    if e.name == "OPERATOR_CALL":
        return "operator"
    if e.name.startswith("native_call_"):
        return "kernel"
    if e.name == "Method::execute":
        return "execute"
    return "load"


def xnn_op_name(name):
    return re.sub(r"\s*#\d+$", "", re.sub(r"^\[n\d+\]\s*", "", name))


def short_ops(ops, n=3):
    ops = [o.replace("aten.", "").replace("dim_order_ops.", "") for o in ops]
    return ",".join(ops[:n]) + (",...(%d)" % len(ops) if len(ops) > n else "")


def split_etrecord(rec):
    """{method: single-method ETRecord} (the Inspector only knows the method 'forward')."""
    dh = rec._debug_handle_map or {}
    names = list(dh.keys())
    out = {}
    for n in names:
        ep = rec.edge_dialect_program[n] if isinstance(rec.edge_dialect_program, dict) else rec.edge_dialect_program
        out[n] = ETRecord(edge_dialect_program=ep, _debug_handle_map={FORWARD: dh[n]},
                          _delegate_map={FORWARD: rec._delegate_map[n]} if rec._delegate_map else None,
                          _instruction_id_to_num_outs_map={FORWARD: rec._instruction_id_to_num_outs_map[n]})
    return out


def attach(blocks, rec1):
    """Same three steps as Inspector._consume_etrecord, for the given blocks and one single-method ETRecord."""
    for b in blocks:
        b._gen_resolve_debug_handles(rec1._debug_handle_map[FORWARD], rec1._delegate_map[FORWARD] if rec1._delegate_map else None,
                                     rec1._instruction_id_to_num_outs_map[FORWARD])
    graphs = gen_graphs_from_etrecord(etrecord=rec1, enable_module_hierarchy=False)
    mp = create_debug_handle_to_op_node_mapping(graphs[EDGE_DIALECT_GRAPH_KEY])
    for b in blocks:
        for e in b.events:
            e._associate_with_op_graph_nodes(debug_handle_to_op_node_map=mp)


def label_method(label):
    m = re.sub(r"_\d+$", "", label)
    return "denoise_step" if m == "denoise" else m


def assign_methods(blocks, labels):
    """Method (and run labels) of each Execute block: the blocks come in order of first appearance, and the runs of one block are
    consecutive in time, so a block of n runs covers the next n labels."""
    exec_blocks = [b for b in blocks if b.name == "Execute"]
    res, pos = {}, 0
    for b in exec_blocks:
        ex = [e for e in b.events if e.name == "Method::execute"]
        n = len(ex[0].perf_data.raw) if ex and ex[0].perf_data else 1
        lab = labels[pos:pos + n]
        pos += n
        if labels and len(lab) != n:
            sys.exit("ETDUMP RUNS has %d labels, the dump has more runs (%d+)" % (len(labels), pos))
        res[id(b)] = (label_method(lab[0]) if lab else None, lab)
    if labels and pos != len(labels):
        sys.exit("ETDUMP RUNS has %d labels, the dump has %d runs" % (len(labels), pos))
    return res


def build_table(blocks, meth, per_run, hint):
    rows = []
    for blk in blocks:
        if blk.name == "Execute":
            method, labels = meth[id(blk)]
            method = method or "forward"
        else:
            method, labels = "(load)", []
        # the portable kernel row (native_call_<op>) sits just before its OPERATOR_CALL row: pair them for the kernel name
        prev_kernel = None
        for e in blk.events:
            p = e.perf_data
            kind = kind_of(e)
            ops = list(e.op_types or [])
            if kind == "kernel":
                prev_kernel = re.sub(r"^native_call_", "", e.name).replace(".out", "")
            if kind == "operator":
                lowered = "portable:" + (prev_kernel or (ops[0].replace("aten.", "") if ops else "?"))
                prev_kernel = None
            elif kind == "kernel":
                lowered = "portable:" + prev_kernel
            elif kind == "xnn-op":
                lowered = "xnnpack:" + xnn_op_name(e.name)
            elif kind == "delegate":
                lowered = "xnnpack:(delegate call)"
            else:
                lowered = ""
            base = {"method": method, "row_kind": KIND[kind], "event": e.name, "lowered_to": lowered, "aten_ops": ";".join(ops),
                    "debug_handles": ";".join(str(h) for h in (e.debug_handles if isinstance(e.debug_handles, (list, tuple)) else
                                                               ([e.debug_handles] if e.debug_handles is not None else []))),
                    "python_source": best_frame(e.stack_traces, hint), "linked": bool(ops) if kind in ("operator", "xnn-op") else None,
                    "_kind": kind}
            raw = list(p.raw) if p else []
            n = len(raw) if raw else 0
            rows.append(dict(base, run="mean" if n > 1 else "1", runs=n, ticks=p.avg if p else None, ticks_min=p.min if p else None,
                             ticks_max=p.max if p else None, ticks_per_run=";".join("%.0f" % x for x in raw)))
            if per_run and n > 1:
                for i, x in enumerate(raw):
                    lab = labels[i] if i < len(labels) else str(i + 1)
                    rows.append(dict(base, run=lab, runs=1, ticks=x, ticks_min=x, ticks_max=x, ticks_per_run="%.0f" % x))
    cols = ["method", "run", "runs", "row_kind", "event", "ticks", "ticks_min", "ticks_max", "ticks_per_run", "lowered_to", "aten_ops",
            "debug_handles", "python_source", "linked", "_kind"]
    return pd.DataFrame(rows, columns=cols)


def md_table(df):
    return df.to_markdown(index=False, floatfmt=".0f")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", help="console log of the run")
    ap.add_argument("--etrecord", help="ETRecord of the exported .pte (links events to ATen nodes and Python lines)")
    ap.add_argument("--out", help="prefix of the output files (default: the log name without .log)")
    ap.add_argument("--csv", action="store_true", help="print CSV instead of Markdown on stdout (PREFIX.csv and PREFIX.md are always written)")
    ap.add_argument("--cpu-hz", type=float, default=50e6, help="core clock, only for the seconds in the totals (default 50 MHz)")
    ap.add_argument("--max-rows", type=int, default=0, help="print only the first N rows of each table (the files have all)")
    ap.add_argument("--all-blocks", action="store_true", help="also print the load block ('Default': Method::init, load_method)")
    ap.add_argument("--hide", default="", help="comma list of row kinds to leave out of the printout, e.g. 'portable kernel,xnnpack operator'")
    ap.add_argument("--per-run", action="store_true", help="CSV: also one row per run for methods that ran several times")
    ap.add_argument("--source-hint", default="", help="prefer Python frames whose path contains this (default: any frame outside torch / site-packages)")
    ap.add_argument("--top", type=int, default=10, help="rows in the top-by-ticks table")
    a = ap.parse_args()

    raw_log = open(a.log, errors="replace").read()
    prefix = a.out or re.sub(r"\.log$", "", a.log)
    data, hdr, labels, crc = extract_etdump(raw_log)
    open(prefix + ".etdp", "wb").write(data)
    out = []
    P = lambda s="": (out.append(s), print(s))
    P("ETDump: %d bytes -> %s.etdp   (runner header: %s%s)" % (len(data), prefix, hdr, ", crc32 ok" if crc else ", no crc32 in this log"))

    insp = Inspector(etdump_path=prefix + ".etdp", etrecord=None, source_time_scale=TimeScale.CYCLES, target_time_scale=TimeScale.CYCLES)
    meth = assign_methods(insp.event_blocks, labels)
    if a.etrecord:
        recs = split_etrecord(parse_etrecord(a.etrecord))
        for b in insp.event_blocks:
            if b.name != "Execute":
                continue
            m = meth[id(b)][0]
            if m is None:                       # no RUNS line: a single-method dump
                if len(recs) != 1:
                    sys.exit("the ETRecord has methods %s but the log has no 'ETDUMP RUNS' line to say which one ran" % list(recs))
                m = next(iter(recs))
                meth[id(b)] = (m, meth[id(b)][1])
            if m not in recs:
                sys.exit("method %r of the dump is not in the ETRecord (methods: %s)" % (m, list(recs)))
            attach([b], recs[m])
    df = build_table(insp.event_blocks, meth, a.per_run, a.source_hint)
    df.drop(columns=["_kind"]).to_csv(prefix + ".csv", index=False)

    ex = df[df["method"] != "(load)"]
    ops = ex[ex["_kind"].isin(["operator", "xnn-op"])]
    P("event blocks: %s, rows: %d (Execute %d), ETRecord: %s" % ([b.name for b in insp.event_blocks], len(df), len(ex), a.etrecord or "no"))
    P("operator rows (portable ops + XNNPACK operators): %d, linked to ATen ops: %d, not linked: %d" %
      (len(ops), int(ops["linked"].sum()), int((~ops["linked"].astype(bool)).sum())))
    P("  by kind: " + ", ".join("%s %d" % (k, n) for k, n in ex["row_kind"].value_counts().items()))
    unl = ops[~ops["linked"].astype(bool)]
    if len(unl):
        P("  not linked, by lowered_to: " + ", ".join("%s x%d" % (k, n) for k, n in unl["lowered_to"].value_counts().items()))
    P()

    # totals per method
    P("## Totals per method (mean per run, ticks = mcycle; seconds at %.0f MHz)" % (a.cpu_hz / 1e6))
    tot = []
    for m, g in ex.groupby("method", sort=False):
        s = lambda k: g.loc[g["_kind"] == k, "ticks"].sum()
        runs = int(g.loc[g["_kind"] == "execute", "runs"].max()) if (g["_kind"] == "execute").any() else 0
        tot.append({"method": m, "runs": runs, "Method::execute": s("execute"), "sum portable ops": s("operator"),
                    "sum delegate calls": s("delegate"), "ops+delegates % of execute": 100 * (s("operator") + s("delegate")) / s("execute") if s("execute") else 0,
                    "rows": len(g), "seconds/run": "%.3f s" % (s("execute") / a.cpu_hz)})
    tdf = pd.DataFrame(tot)
    P(tdf.to_markdown(index=False, floatfmt=",.0f") if len(tdf) else "(none)")
    P()
    m_run = re.findall(r"(?:execute\(\)|CHAIN \S+|STEP \S+)[^\n]*?(?:mcycle delta|minstret delta) = (\d+)", raw_log.replace("\r", ""))
    if m_run:
        P("runner's own windows (tracer calls sit inside them): " + ", ".join(m_run[:16]) + (" ..." if len(m_run) > 16 else ""))
        P()

    P("## Top %d rows by ticks (mean per run)" % a.top)
    top = ops.sort_values("ticks", ascending=False).head(a.top)[["method", "event", "ticks", "lowered_to", "aten_ops", "python_source"]]
    P(md_table(top))
    P()

    hide = [h.strip() for h in a.hide.split(",") if h.strip()]
    for m, g in df.groupby("method", sort=False):
        if m == "(load)" and not a.all_blocks:
            continue
        g = g[~g["row_kind"].isin(hide)]
        shown = g[["run", "row_kind", "event", "ticks", "lowered_to", "aten_ops", "python_source"]].copy()
        shown["aten_ops"] = shown["aten_ops"].map(lambda s: short_ops(s.split(";")) if s else "")
        if a.max_rows:
            shown = shown.head(a.max_rows)
        P("## Method %s (%d rows%s)" % (m, len(g), "; runs: %s" % ",".join(map(str, g["runs"].unique())) if m != "(load)" else ""))
        P(shown.to_csv(index=False) if a.csv else md_table(shown))
        P()
    open(prefix + ".md", "w").write("\n".join(out) + "\n")
    print("wrote %s.md %s.csv %s.etdp" % (prefix, prefix, prefix))


if __name__ == "__main__":
    main()
