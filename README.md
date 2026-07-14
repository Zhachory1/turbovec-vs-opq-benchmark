# TurboVec vs FAISS OPQ — benchmark

A reproducible benchmark comparing [**TurboVec**](https://github.com/RyanCodrai/turbovec)
(Google TurboQuant, Rust + SIMD, no-train flat index) against **FAISS OPQ** —
both the flat `OPQ,PQ` (exhaustive) and the production-realistic `OPQ,IVF,PQ`
(list-pruned) configurations — on **index size, build time, serving speed (QPS +
latency), and recall@1/@10/@100**.

**Grid:** {10k, 100k, 1M, 10M} vectors × {10, 100, 500} dims (10M×500 skipped —
exceeds 24 GB RAM), at 2-bit and 4-bit quantization rates. 66 configurations.

## Results

See [`report/REPORT.md`](report/REPORT.md) for the full write-up, per-cell
tables, head-to-head ratios, methodology, and caveats. Raw numbers:
[`results/results.csv`](results/results.csv).

**TL;DR:** TurboVec wins build time by 15×–3000× (no training step) and wins
serving QPS against flat OPQ by 2.7×–154×. Against the realistic OPQ+IVF it's
closer — TurboVec wins most cells but IVF's list-pruning overtakes it at very
large n + low dimension. Recall: TurboVec beats flat OPQ everywhere; vs OPQ+IVF
it's mixed. Recall is on **synthetic clustered data** and is directional only —
rerun on real embeddings before drawing conclusions.

## Figures

Serving speed vs recall — every cell, 4-bit (up-and-right is better):

![QPS vs recall](report/figures/fig_qps_vs_recall.png)

| | |
|---|---|
| ![build time](report/figures/fig_build_time.png) | ![QPS](report/figures/fig_qps.png) |
| ![recall@10](report/figures/fig_recall.png) | ![size](report/figures/fig_size.png) |

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install numpy faiss-cpu turbovec matplotlib
.venv/bin/python scripts/bench.py       # TurboVec + flat OPQ  (writes results/results.csv)
.venv/bin/python scripts/bench_ivf.py   # OPQ+IVF              (appends)
.venv/bin/python scripts/make_report.py # regenerate tables from the CSV
```

Small smoke mode for CI/local sanity checks:

```bash
.venv/bin/python scripts/bench.py --smoke
.venv/bin/python scripts/bench_ivf.py --smoke
```

Benchmark scripts write `results/environment.json` with Python/platform/library versions. Runs are idempotent by default: rows with the same `(n, dim, engine, bit_width, params, seed)` are skipped instead of duplicated. Use `--rerun` to append duplicate comparison rows intentionally.

Run multiple seeds to estimate variance; generated report tables show mean ± sample standard deviation for metrics when a cell has multiple seed rows, and plots render error bars:

```bash
.venv/bin/python scripts/bench.py --smoke --seed-count 3
.venv/bin/python scripts/bench_ivf.py --smoke --seeds 111 222 333
```

Tune IVF frontiers explicitly:

```bash
.venv/bin/python scripts/bench_ivf.py --sizes 100000 --dims 100 --bits 4 --nlist 128 256 512 --nprobe 1 4 16 64
```

Plot a different bit-width or size-bar reference cell:

```bash
.venv/bin/python scripts/plot.py --bit 2 --size-ref-n 100000
```

Single-threaded; vectors unit-normalized (cosine); exact ground truth via
`IndexFlatIP`. TurboVec requires dim be a multiple of 8, so vectors are
zero-padded (does not change cosine ranking). See the report for full
methodology and the fairness controls (rate matching, OPQ training subsample,
IVF `nlist`/`nprobe`).

## License

MIT — see [`LICENSE`](LICENSE).
