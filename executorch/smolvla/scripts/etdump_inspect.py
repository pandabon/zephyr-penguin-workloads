#!/usr/bin/env python3
"""Turn the ETDump a Penguin run printed on the console into a per-operator table (ExecuTorch Inspector).

  etdump_inspect.py RUN.log [--etrecord X.etrecord] [--out PREFIX] [--csv] [--cpu-hz 50000000]

1. cuts the base64 block between "ETDUMP BEGIN ..." and "ETDUMP END" out of RUN.log (printed by the runner built with
   -DEXEC_ETDUMP=ON, src/riscv_executor_runner.cpp) and writes the binary dump to PREFIX.etdp;
2. loads it with ExecuTorch's Inspector (devtools/inspector), with the ETRecord when given, reading the times as CYCLES
   (the runner's PAL clock is mcycle: core cycles on the FPGA, instructions on Spike);
3. prints one row per profiled event (operator calls, XNNPACK delegate operators), as a Markdown table or CSV.

Needs the python of the ExecuTorch checkout the .pte was exported with, e.g. smolvla-explore/.venv-main/bin/python.
The ETRecord comes from the export: smolvla-explore/export_expert_layer.py --etrecord.

Credits: the base64 block and the "inspector_cli ... --source_time_scale cycles --target_time_scale cycles" call follow ExecuTorch
examples/arm/executor_runner/arm_executor_runner.cpp (write_etdump) and devtools/inspector/inspector_cli.py @ 681c3edb47
(Copyright Meta Platforms and Arm Limited, BSD-style licence). Written for this repo otherwise.

Reading the numbers: with an Inspector time scale of CYCLES nothing is converted, so every time is in mcycle ticks. Divide by
the core clock for seconds (50 MHz on the FPGA board: 1 tick = 20 ns). On Spike mcycle == minstret, so a "time" is an instruction count.
"""
import argparse
import base64
import re
import sys

import pandas as pd

from executorch.devtools import Inspector
from executorch.devtools.inspector import TimeScale


def extract_etdump(log_text):
    """Return (bytes, header line) of the last ETDUMP block in the log."""
    log_text = log_text.replace("\r", "")   # the console log has CRLF line ends
    m = list(re.finditer(r"^ETDUMP BEGIN ([^\n]*)\n(.*?)^ETDUMP END", log_text, re.S | re.M))
    if not m:
        sys.exit("no 'ETDUMP BEGIN ... ETDUMP END' block in the log (was the app built with -DEXEC_ETDUMP=ON, and did the run finish?)")
    hdr, body = m[-1].group(1), m[-1].group(2)
    # console lines can carry a CR or a log prefix; keep only base64 characters per line
    b64 = "".join(re.sub(r"[^A-Za-z0-9+/=]", "", ln) for ln in body.splitlines())
    m_size = re.search(r"size=(\d+)", hdr)
    data = base64.b64decode(b64)
    if m_size and int(m_size.group(1)) != len(data):
        sys.exit("ETDump size mismatch: header says %s bytes, decoded %d (lost console characters?)" % (m_size.group(1), len(data)))
    return data, hdr


def last_frame(stack_traces):
    """Python source of the op: last 'File "...", line N, in fn' of its first stack trace, plus the code line."""
    for tr in (stack_traces or {}).values():
        frames = re.findall(r'File "([^"]+)", line (\d+), in (\S+)\n\s*(.*)', tr)
        if frames:
            f, ln, fn, code = frames[-1]
            return "%s:%s %s" % (f.rsplit("/", 1)[-1], ln, code.strip())
    return ""


def kind_of(e):
    """What a row is. The Inspector names them: OPERATOR_CALL = a portable kernel call incl. the framework's own overhead
    (the `native_call_<op>.out` row just before it is the kernel alone); DELEGATE_CALL = one whole delegate (here XNNPACK);
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


def short_ops(op_types, n=3):
    ops = [o.replace("aten.", "").replace("dim_order_ops.", "") for o in op_types]
    return ",".join(ops[:n]) + (",...(%d)" % len(ops) if len(ops) > n else "")


def build_table(insp):
    rows = []
    for blk in insp.event_blocks:
        for e in blk.events:
            p = e.perf_data
            rows.append({
                "block": blk.name,
                "kind": kind_of(e),
                "event": e.name,
                "avg_cycles": p.avg if p else None,
                "min_cycles": p.min if p else None,
                "max_cycles": p.max if p else None,
                "runs": len(p.raw) if p else 0,
                "backend": e.delegate_backend_name or "",
                "op_types": short_ops(e.op_types),
                "debug_handles": str(e.debug_handles) if e.debug_handles is not None else "",
                "python_source": last_frame(e.stack_traces),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", help="console log of the run")
    ap.add_argument("--etrecord", help="ETRecord of the exported .pte (links events to ATen nodes and Python lines)")
    ap.add_argument("--out", help="prefix of the output files (default: the log name without .log)")
    ap.add_argument("--csv", action="store_true", help="CSV instead of a Markdown table on stdout (PREFIX.csv is always written)")
    ap.add_argument("--cpu-hz", type=float, default=50e6, help="core clock, only for the seconds in the totals (default 50 MHz)")
    ap.add_argument("--max-rows", type=int, default=0, help="print only the first N rows (the files have all)")
    ap.add_argument("--all-blocks", action="store_true", help="also print the load block ('Default': Method::init, load_method)")
    ap.add_argument("--hide", default="", help="comma list of row kinds to leave out of the printout, e.g. kernel,xnn-op")
    a = ap.parse_args()

    raw_log = open(a.log, errors="replace").read()
    prefix = a.out or re.sub(r"\.log$", "", a.log)
    data, hdr = extract_etdump(raw_log)
    open(prefix + ".etdp", "wb").write(data)
    print("ETDump: %d bytes -> %s.etdp   (runner header: %s)" % (len(data), prefix, hdr))

    insp = Inspector(etdump_path=prefix + ".etdp", etrecord=a.etrecord,
                     source_time_scale=TimeScale.CYCLES, target_time_scale=TimeScale.CYCLES)
    df = build_table(insp)
    df.to_csv(prefix + ".csv", index=False)
    print("event blocks: %s, events: %d, ETRecord: %s" % ([b.name for b in insp.event_blocks], len(df), "yes" if a.etrecord else "no"))

    shown = df if a.all_blocks else df[df["block"] == "Execute"]
    if a.hide:
        shown = shown[~shown["kind"].isin(a.hide.split(","))]
    shown = shown.drop(columns=["min_cycles", "max_cycles"] + (["block"] if not a.all_blocks else []))
    if a.max_rows:
        shown = shown.head(a.max_rows)
    print(shown.to_csv(index=False) if a.csv else shown.to_markdown(index=False, floatfmt=".0f"))

    # Totals of the (last) Execute block, per run, in cycles.
    ex = df[df["block"] == "Execute"].reset_index(drop=True)
    tot = lambda k: ex.loc[ex["kind"] == k, "avg_cycles"].sum()
    method = tot("execute")
    print("\nTotals of the Execute block (avg per run, mcycle ticks; seconds at %.0f MHz):" % (a.cpu_hz / 1e6))
    print("  Method::execute row                       %12.0f" % method)
    print("  sum(operator rows)  portable kernels      %12.0f   (%d rows)" % (tot("operator"), (ex["kind"] == "operator").sum()))
    print("  sum(delegate rows)  whole XNNPACK calls   %12.0f   (%d rows)" % (tot("delegate"), (ex["kind"] == "delegate").sum()))
    print("  operators + delegates                     %12.0f   = %.1f%% of Method::execute" %
          (tot("operator") + tot("delegate"), 100 * (tot("operator") + tot("delegate")) / method if method else 0))
    # XNNPACK operator rows come out just before the DELEGATE_CALL row of their delegate: sum them per delegate.
    grp, acc, i = [], 0.0, 0
    for _, r in ex.iterrows():
        if r["kind"] == "xnn-op":
            acc += r["avg_cycles"]
        elif r["kind"] == "delegate":
            i += 1
            grp.append((i, acc, r["avg_cycles"]))
            acc = 0.0
    for i, xs, dc in grp:
        print("  delegate #%d: sum(xnn-op rows) %12.0f vs DELEGATE_CALL %12.0f" % (i, xs, dc))
    m = re.findall(r"execute\(\)[^\n]*?mcycle delta = (\d+)", raw_log.replace("\r", ""))
    if m:
        print("  runner's own execute() mcycle delta       %12s   (the tracer's calls sit inside this window)" % m[-1])
    print("  1 tick = %.1f ns at %.0f MHz" % (1e9 / a.cpu_hz, a.cpu_hz / 1e6))


if __name__ == "__main__":
    main()
