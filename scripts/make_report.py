#!/usr/bin/env python3
"""Read results/results.csv and emit formatted comparison tables for the report."""
import csv, os, collections, statistics

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


def group_rows(rows):
    grouped = collections.defaultdict(list)
    for r in rows:
        if not r.get("n"):
            continue
        key = (int(r["n"]), int(r["dim"]), r["engine"], int(r["bit_width"]), r.get("params", ""))
        grouped[key].append(r)
    return grouped


def groups_for(grouped, n, d, eng, bw):
    return sorted(
        ((key, rows) for key, rows in grouped.items() if key[:4] == (n, d, eng, bw)),
        key=lambda item: item[0][4],
    )


def values(rows, field):
    out = []
    for r in rows:
        v = fnum(r.get(field))
        if v is not None:
            out.append(v)
    return out


def mean(rows, field):
    vals = values(rows, field)
    return (sum(vals) / len(vals)) if vals else None


def std(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def fmt_fixed(x, digits=2, integer=False):
    return f"{x:.0f}" if integer else f"{x:.{digits}f}"


def fmt_stat(rows, field, digits=2, integer=False, size=False):
    vals = values(rows, field)
    if not vals:
        return ""
    m = sum(vals) / len(vals)
    if size:
        text = sz(m)
    else:
        text = fmt_fixed(m, digits=digits, integer=integer)
    if len(vals) > 1:
        s = std(vals)
        if size:
            return f"{text} ± {sz(s)}"
        return f"{text} ± {fmt_fixed(s, digits=digits, integer=integer)}"
    return text


def seed_label(params, rows):
    seeds = {r.get("seed", "") for r in rows if r.get("seed", "") != ""}
    return params if len(seeds) <= 1 else f"{params} ({len(seeds)} seeds)"


def ratio(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def main():
    rows = load()
    grouped = group_rows(rows)

    ns = sorted({int(r["n"]) for r in rows if r.get("n")})
    dims = sorted({int(r["dim"]) for r in rows if r.get("n")})
    bits = sorted({int(r["bit_width"]) for r in rows if r.get("n")})

    out = [
        "\nMean ± sample standard deviation is shown when multiple seed rows exist for the same "
        "(n, dim, engine, bit_width, params) cell.\n"
    ]
    for bw in bits:
        out.append(f"\n## {bw}-bit\n")
        for d in dims:
            out.append(f"\n### dim = {d}\n")
            out.append("| n | engine | params | build (s) | size | bytes/vec | QPS | p50 lat (ms) | R@1 | R@10 | R@100 |")
            out.append("|---|--------|--------|-----------|------|-----------|-----|--------------|-----|------|-------|")
            names = {"turbovec": "TurboVec", "faiss_opq": "FAISS OPQ (flat)",
                     "faiss_opq_ivf": "FAISS OPQ+IVF"}
            for n in ns:
                for eng in ("turbovec", "faiss_opq", "faiss_opq_ivf"):
                    for key, cell_rows in groups_for(grouped, n, d, eng, bw):
                        name = names[eng]
                        params = seed_label(key[4], cell_rows)
                        out.append(
                            f"| {human(n)} | {name} | {params} | {fmt_stat(cell_rows, 'build_s', 2)} | "
                            f"{fmt_stat(cell_rows, 'size_bytes', size=True)} | {fmt_stat(cell_rows, 'bytes_per_vec', 1)} | "
                            f"{fmt_stat(cell_rows, 'qps', integer=True)} | {fmt_stat(cell_rows, 'lat_ms_p50', 3)} | "
                            f"{fmt_stat(cell_rows, 'recall@1', 4)} | {fmt_stat(cell_rows, 'recall@10', 4)} | "
                            f"{fmt_stat(cell_rows, 'recall@100', 4)} |")
    out.append("\n## Head-to-head ratios (TurboVec vs FAISS OPQ)\n")
    out.append("Ratio >1 favours TurboVec for QPS/build; <1 favours the FAISS variant. "
               "R@10 delta = TV minus variant (positive => TV higher recall). Ratios use seed means.\n")
    for variant, vlabel in (("faiss_opq", "vs flat OPQ,PQ"), ("faiss_opq_ivf", "vs OPQ+IVF+PQ")):
        out.append(f"\n### TurboVec {vlabel}\n")
        out.append("| n | dim | bit | variant params | size TV/v | build TV/v | QPS TV/v | R@10 TV-v |")
        out.append("|---|-----|-----|----------------|-----------|------------|----------|-----------|")
        for bw in bits:
            for n in ns:
                for d in dims:
                    tv_cells = groups_for(grouped, n, d, "turbovec", bw)
                    if not tv_cells:
                        continue
                    _, tv = tv_cells[0]
                    for key, op in groups_for(grouped, n, d, variant, bw):
                        sr = ratio(mean(tv, "size_bytes"), mean(op, "size_bytes"))
                        br = ratio(mean(tv, "build_s"), mean(op, "build_s"))
                        qr = ratio(mean(tv, "qps"), mean(op, "qps"))
                        tr = mean(tv, "recall@10")
                        or_ = mean(op, "recall@10")
                        if None in (sr, br, qr, tr, or_):
                            continue
                        out.append(
                            f"| {human(n)} | {d} | {bw} | {key[4]} | {sr:.2f}x | {br:.3f}x | "
                            f"{qr:.1f}x | {tr - or_:+.3f} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
