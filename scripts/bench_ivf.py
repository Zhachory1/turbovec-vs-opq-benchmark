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
    SIZES, DIMS, SKIP, BITS, KMAX, RECALL_KS, SEED, FIELDS,
    pad8, opq_m, gen_data, recall_at, time_search, RESULTS_DIR, IDX_DIR, CSV_PATH,
)

faiss.omp_set_num_threads(1)

IVF_TRAIN_SAMPLE = 100_000
NLIST_CAP = 4096


def pick_nlist(n, train):
    nlist = min(int(4 * math.sqrt(n)), NLIST_CAP)
    nlist = min(nlist, max(1, train // 39))  # >=39 training pts per centroid
    return max(1, nlist)


def run_cell(n, d, writer, fh, bits=None, nlists=None, nprobes=None, n_queries_override=None, repeats_override=None):
    rng = np.random.default_rng(SEED + n + d)  # same seed as pass 1 -> same data
    bits = bits or BITS
    n_queries = n_queries_override or (500 if n >= 10_000_000 else 1000)
    search_repeats = repeats_override or (1 if n >= 1_000_000 else 3)
    print(f"\n=== IVF cell n={n} d={d} ===", flush=True)
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
            path = os.path.join(IDX_DIR, f"opqivf_{n}_{d}_{bw}.faiss")
            faiss.write_index(fi, path)
            size = os.path.getsize(path)
            qps, p50, p99, out = time_search(lambda q, k: fi.search(q, k), Q, KMAX, search_repeats)
            fidx = np.asarray(out[1])
            row = dict(n=n, dim=d, engine="faiss_opq_ivf", bit_width=bw,
                       params=f"{fac} nprobe={nprobe}", build_s=round(build_s, 3),
                       size_bytes=size, bytes_per_vec=round(size / n, 2),
                       qps=round(qps, 1), lat_ms_p50=round(p50, 4), lat_ms_p99=round(p99, 4),
                       n_queries=n_queries)
            for k in RECALL_KS:
                row[f"recall@{k}"] = round(recall_at(fidx, gt, k), 4)
            writer.writerow(row); fh.flush()
            print(f"  OPQ+IVF{bw}({fac} nprobe={nprobe}): build={build_s:.2f}s "
                  f"size={size/1e6:.1f}MB qps={qps:.0f} r@10={row['recall@10']}", flush=True)
            os.remove(path)
            del fi
            gc.collect()
          except Exception as e:
            print(f"  OPQ+IVF{bw} nlist={nlist} nprobe={nprobe} FAILED: {e}", flush=True)

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
    args = ap.parse_args()

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

    fh = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    for n in sizes:
        for d in args.dims:
            if (n, d) in SKIP:
                continue
            run_cell(
                n,
                d,
                writer,
                fh,
                bits=args.bits,
                nlists=args.nlist,
                nprobes=args.nprobe,
                n_queries_override=n_queries,
                repeats_override=repeats,
            )
    fh.close()
    print("\nIVF DONE. appended ->", CSV_PATH, flush=True)


if __name__ == "__main__":
    main()
