#!/usr/bin/env python3
"""Read results/results.csv and emit formatted comparison tables for the report."""
import csv, os, collections

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(HERE, "results", "results.csv")


def load():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def human(n):
    n = int(n)
    return {10000: "10k", 100000: "100k", 1000000: "1M", 10000000: "10M"}.get(n, str(n))


def sz(b):
    b = float(b)
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024 or u == "GB":
            return f"{b:.1f}{u}"
        b /= 1024


def main():
    rows = load()
    # index by (n,dim,engine,bit)
    R = {}
    for r in rows:
        R[(int(r["n"]), int(r["dim"]), r["engine"], int(r["bit_width"]))] = r

    ns = sorted({int(r["n"]) for r in rows})
    dims = sorted({int(r["dim"]) for r in rows})
    bits = sorted({int(r["bit_width"]) for r in rows})

    out = []
    for bw in bits:
        out.append(f"\n## {bw}-bit\n")
        # Build a table per dim
        for d in dims:
            out.append(f"\n### dim = {d}\n")
            out.append("| n | engine | params | build (s) | size | bytes/vec | QPS | p50 lat (ms) | R@1 | R@10 | R@100 |")
            out.append("|---|--------|--------|-----------|------|-----------|-----|--------------|-----|------|-------|")
            names = {"turbovec": "TurboVec", "faiss_opq": "FAISS OPQ (flat)",
                     "faiss_opq_ivf": "FAISS OPQ+IVF"}
            for n in ns:
                for eng in ("turbovec", "faiss_opq", "faiss_opq_ivf"):
                    r = R.get((n, d, eng, bw))
                    if not r:
                        continue
                    name = names[eng]
                    out.append(
                        f"| {human(n)} | {name} | {r['params']} | {fnum(r['build_s']):.2f} | "
                        f"{sz(r['size_bytes'])} | {fnum(r['bytes_per_vec']):.1f} | "
                        f"{fnum(r['qps']):.0f} | {fnum(r['lat_ms_p50']):.3f} | "
                        f"{r['recall@1']} | {r['recall@10']} | {r['recall@100']} |")
    # head-to-head ratios summary
    out.append("\n## Head-to-head ratios (TurboVec vs FAISS OPQ)\n")
    out.append("Ratio >1 favours TurboVec for QPS/build; <1 favours the FAISS variant. "
               "R@10 delta = TV minus variant (positive => TV higher recall).\n")
    for variant, vlabel in (("faiss_opq", "vs flat OPQ,PQ"), ("faiss_opq_ivf", "vs OPQ+IVF+PQ")):
        out.append(f"\n### TurboVec {vlabel}\n")
        out.append("| n | dim | bit | size TV/v | build TV/v | QPS TV/v | R@10 TV-v |")
        out.append("|---|-----|-----|-----------|------------|----------|-----------|")
        for bw in bits:
            for n in ns:
                for d in dims:
                    tv = R.get((n, d, "turbovec", bw))
                    op = R.get((n, d, variant, bw))
                    if not (tv and op):
                        continue
                    sr = fnum(tv["size_bytes"]) / fnum(op["size_bytes"])
                    br = fnum(tv["build_s"]) / max(fnum(op["build_s"]), 1e-9)
                    qr = fnum(tv["qps"]) / max(fnum(op["qps"]), 1e-9)
                    rd = fnum(tv["recall@10"]) - fnum(op["recall@10"])
                    out.append(f"| {human(n)} | {d} | {bw} | {sr:.2f}x | {br:.3f}x | {qr:.1f}x | {rd:+.3f} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
