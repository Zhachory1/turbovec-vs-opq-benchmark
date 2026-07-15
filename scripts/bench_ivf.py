#!/usr/bin/env python3
"""Pass 2: add FAISS OPQ+IVF+PQ (production-realistic OPQ deployment) to the grid.

Pass 1 (bench.py) compared TurboVec (flat exhaustive SIMD scan) against flat
FAISS OPQ,PQ (also exhaustive) — an apples-to-apples scan-everything baseline.
But OPQ in production is paired with IVF so search only probes `nprobe` of
`nlist` lists instead of scanning the whole set. This pass adds
`OPQ{m},IVF{nlist},PQ{m}` and appends rows (engine=faiss_opq_ivf) to the same
results.csv. Same data/seed/queries/recall protocol as pass 1, so rows line up.

IVF params (documented, NOT exhaustively tuned — IVF QPS/recall is sensitive to
these):
  nlist  = min(4*sqrt(n), 4096), further capped so the training sample gives
           >=39 points/centroid (k-means needs that for stable coarse centroids)
  nprobe = max(1, nlist // 16)   (~6% of lists scanned)
  train  = min(n, 100_000)       (bigger than pass 1's 12k flat cap: IVF coarse
           quantizer needs enough points to populate nlist centroids)
"""
import os, sys, time, gc, csv, math, argparse
import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import (  # noqa: E402
    SIZES, DIMS, SKIP, BITS, KMAX, RECALL_KS, FIELDS,
    pad8, opq_m, gen_data, recall_at, time_search, RESULTS_DIR, IDX_DIR, CSV_PATH,
    load_existing_keys, write_environment, write_row, failure_row, ensure_csv_header, seed_values,
)

faiss.omp_set_num_threads(1)

IVF_TRAIN_SAMPLE = 100_000
NLIST_CAP = 4096


def pick_nlist(n, train):
    nlist = min(int(4 * math.sqrt(n)), NLIST_CAP)
    nlist = min(nlist, max(1, train // 39))  # >=39 training pts per centroid
    return max(1, nlist)


def run_cell(n, d, writer, fh, seed, bits=None, nlists=None, nprobes=None, n_queries_override=None, repeats_override=None, seen=None):
    rng = np.random.default_rng(seed)  # same seed as pass 1 -> same data
    bits = bits or BITS
    seen = seen if seen is not None else set()
    n_queries = n_queries_override or (500 if n >= 10_000_000 else 1000)
    search_repeats = repeats_override or (1 if n >= 1_000_000 else 3)
    print(f"\n=== IVF cell n={n} d={d} seed={seed} ===", flush=True)
    X, Q = gen_data(n, d, n_queries, rng)
    flat = faiss.IndexFlatIP(d)
    flat.add(X)
    _, gt = flat.search(Q, KMAX)
    del flat
    gc.collect()

    train = min(n, IVF_TRAIN_SAMPLE)
    default_nlist = pick_nlist(n, train)
    nlists = nlists or [default_nlist]

    for bw in bits:
      for nlist in nlists:
        probe_values = nprobes or [max(1, nlist // 16)]
        for nprobe in probe_values:
          try:
            m = opq_m(d, bw)
            fac = f"OPQ{m},IVF{nlist},PQ{m}"
            t = time.perf_counter()
            fi = faiss.index_factory(d, fac, faiss.METRIC_INNER_PRODUCT)
            tr = X if n <= train else X[rng.choice(n, train, replace=False)]
            fi.train(tr)
            fi.add(X)
            build_s = time.perf_counter() - t
            faiss.ParameterSpace().set_index_parameter(fi, "nprobe", nprobe)
            path = os.path.join(IDX_DIR, f"opqivf_{n}_{d}_{bw}_{seed}.faiss")
            faiss.write_index(fi, path)
            size = os.path.getsize(path)
            qps, p50, p99, out = time_search(lambda q, k: fi.search(q, k), Q, KMAX, search_repeats)
            fidx = np.asarray(out[1])
            row = dict(n=n, dim=d, engine="faiss_opq_ivf", bit_width=bw,
                       params=f"{fac} nprobe={nprobe}", seed=seed, build_s=round(build_s, 3),
                       size_bytes=size, bytes_per_vec=round(size / n, 2),
                       qps=round(qps, 1), lat_ms_p50=round(p50, 4), lat_ms_p99=round(p99, 4),
                       n_queries=n_queries)
            for k in RECALL_KS:
                row[f"recall@{k}"] = round(recall_at(fidx, gt, k), 4)
            write_row(row, writer, fh, seen)
            print(f"  OPQ+IVF{bw}({fac} nprobe={nprobe}): build={build_s:.2f}s "
                  f"size={size/1e6:.1f}MB qps={qps:.0f} r@10={row['recall@10']}", flush=True)
            os.remove(path)
            del fi
            gc.collect()
          except Exception as e:
            print(f"  OPQ+IVF{bw} nlist={nlist} nprobe={nprobe} FAILED: {e}", flush=True)
            write_row(failure_row(n, d, "faiss_opq_ivf", bw, f"OPQ+IVF nlist={nlist} nprobe={nprobe}", e, n_queries, seed), writer, fh, seen)

    del X, Q, gt
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    ap.add_argument("--dims", type=int, nargs="*", default=DIMS)
    ap.add_argument("--bits", type=int, nargs="*", default=BITS)
    ap.add_argument("--nlist", type=int, nargs="*", default=None)
    ap.add_argument("--nprobe", type=int, nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true", help="Run a tiny CI-friendly IVF cell.")
    ap.add_argument("--rerun", action="store_true", help="Append rows even when matching result rows already exist.")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="Explicit RNG seeds to run for every benchmark cell.")
    ap.add_argument("--seed-count", type=int, default=1, help="Run this many deterministic seeds per cell when --seeds is omitted.")
    args = ap.parse_args()
    if args.seed_count < 1:
        ap.error("--seed-count must be >= 1")
    if args.seeds is not None and not args.seeds:
        ap.error("--seeds requires at least one seed value")

    sizes = args.sizes or SIZES
    n_queries = None
    repeats = None
    if args.smoke:
        sizes = [1_000]
        args.dims = [10]
        args.bits = [2]
        args.nlist = [16]
        args.nprobe = [1, 4]
        n_queries = 25
        repeats = 1

    write_environment()
    ensure_csv_header(CSV_PATH)
    seen = set() if args.rerun else load_existing_keys(CSV_PATH)

    new_file = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
    if new_file:
        writer.writeheader(); fh.flush()
    for n in sizes:
        for d in args.dims:
            if (n, d) in SKIP:
                continue
            for seed in seed_values(args, n, d):
                run_cell(
                    n,
                    d,
                    writer,
                    fh,
                    seed=seed,
                    bits=args.bits,
                    nlists=args.nlist,
                    nprobes=args.nprobe,
                    n_queries_override=n_queries,
                    repeats_override=repeats,
                    seen=seen,
                )
    fh.close()
    print("\nIVF DONE. appended ->", CSV_PATH, flush=True)


if __name__ == "__main__":
    main()
