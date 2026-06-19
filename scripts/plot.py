#!/usr/bin/env python3
"""Plot benchmark figures from results/results.csv into report/figures/*.png.

Four figures, each faceted by dim (10/100/500), 3 engines, 4-bit rate (the
representative rate; 2-bit is in the CSV):
  fig_build_time.png   build time vs n        (log-log)
  fig_qps.png          serving QPS vs n       (log-log)
  fig_recall.png       recall@10 vs n         (x log)
  fig_size.png         bytes/vector by dim    (grouped bars)
Plus fig_qps_vs_recall.png — QPS/recall tradeoff scatter (all cells, 4-bit).
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "results.csv")
OUT = os.path.join(HERE, "report", "figures")
os.makedirs(OUT, exist_ok=True)

ENGINES = [
    ("turbovec", "TurboVec", "#0072B2", "o"),
    ("faiss_opq", "FAISS OPQ (flat)", "#D55E00", "s"),
    ("faiss_opq_ivf", "FAISS OPQ+IVF", "#009E73", "^"),
]
DIMS = [10, 100, 500]
BIT = 4  # representative rate for the figures
SIZE_REF_N = 1_000_000


def load():
    with open(CSV) as f:
        rows = list(csv.DictReader(f))
    R = {}
    for r in rows:
        R[(int(r["n"]), int(r["dim"]), r["engine"], int(r["bit_width"]))] = r
    ns = sorted({int(r["n"]) for r in rows})
    return R, ns


def series(R, ns, d, eng, field, bit=BIT):
    xs, ys = [], []
    for n in ns:
        r = R.get((n, d, eng, bit))
        if r and r.get(field) not in (None, ""):
            xs.append(n); ys.append(float(r[field]))
    return xs, ys


def faceted(R, ns, field, title, ylabel, fname, logy=True, bit=BIT):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, d in zip(axes, DIMS):
        for eng, label, color, mk in ENGINES:
            xs, ys = series(R, ns, d, eng, field, bit=bit)
            if xs:
                ax.plot(xs, ys, marker=mk, color=color, label=label, lw=2, ms=7)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"dim = {d}")
        ax.set_xlabel("vectors (n)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle(f"{title}  ({bit}-bit)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p)


def size_bars(R, ns, bit=BIT, ref_n=SIZE_REF_N):
    # bytes/vec is approximately n-independent; use a configurable reference cell.
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(DIMS)); w = 0.26
    for i, (eng, label, color, _) in enumerate(ENGINES):
        vals = []
        for d in DIMS:
            r = R.get((ref_n, d, eng, bit))
            vals.append(float(r["bytes_per_vec"]) if r else 0.0)
        ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
    ax.set_xticks(x); ax.set_xticklabels([f"dim={d}" for d in DIMS])
    ax.set_ylabel("bytes / vector")
    ax.set_title(f"Index size per vector ({ref_n:,}, {bit}-bit)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_size.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p)


def qps_vs_recall(R, ns, bit=BIT):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for eng, label, color, mk in ENGINES:
        xs, ys = [], []
        for n in ns:
            for d in DIMS:
                r = R.get((n, d, eng, bit))
                if r:
                    ys.append(float(r["qps"])); xs.append(float(r["recall@10"]))
        ax.scatter(xs, ys, color=color, marker=mk, label=label, s=55, alpha=0.75, edgecolors="k", lw=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("recall@10"); ax.set_ylabel("QPS (single-thread)")
    ax.set_title(f"Serving speed vs recall — all cells ({bit}-bit)\nup-and-right is better",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_qps_vs_recall.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit", type=int, default=BIT)
    ap.add_argument("--size-ref-n", type=int, default=SIZE_REF_N)
    args = ap.parse_args()

    R, ns = load()
    faceted(R, ns, "build_s", "Index build time vs corpus size", "build time (s, log)",
            "fig_build_time.png", logy=True, bit=args.bit)
    faceted(R, ns, "qps", "Serving throughput vs corpus size", "QPS (log)",
            "fig_qps.png", logy=True, bit=args.bit)
    faceted(R, ns, "recall@10", "Recall@10 vs corpus size", "recall@10",
            "fig_recall.png", logy=False, bit=args.bit)
    size_bars(R, ns, bit=args.bit, ref_n=args.size_ref_n)
    qps_vs_recall(R, ns, bit=args.bit)


if __name__ == "__main__":
    main()
