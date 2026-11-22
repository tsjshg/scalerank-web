# ScaleRank research code

The Python side of ScaleRank's development: the prototypes the Rust core was
ported from, the scripts behind the experiments, and the source of the
published [research notes](https://tsjshg.github.io/scalerank-web/docs/).
Everything here runs on [drbench](https://github.com/tsjshg/drbench), the
benchmark harness.

| Path | Contents |
|---|---|
| `src/scalerank_research/scalerank.py` | First prototype, with N × N rank tables |
| `src/scalerank_research/scalerank_fast.py` | The N²-free version that `crates/scalerank-core` ports |
| `src/scalerank_research/methods.py` | Registers `scalerank`, `scalerank_fast` and the Rust ports (`scalerank_rs`, `tsne_rs`, `umap_rs`) as drbench methods |
| `scripts/` | Experiment scripts (parameter sweeps, outlier and force diagnostics, comparisons with tuned t-SNE/UMAP and between implementations) and `build_overview.py`, which builds the research notes |
| `docs/` | Templates of the research notes (English and Japanese) |
| `tests/` | Tests of the prototypes |

## Setup

Install this package into drbench's environment. It declares a
`drbench.methods` entry point, so drbench picks the methods up by itself.

```bash
cd path/to/drbench
uv sync
uv pip install -e path/to/scalerank-web/research
# optional, for scalerank_rs / tsne_rs / umap_rs: the Rust bindings
uv pip install path/to/scalerank-web/crates/scalerank-py
uv run drbench list   # now includes scalerank, scalerank_fast, ...
```

## Running

Run from the drbench folder, so the results land in drbench's `results/`:

```bash
uv run drbench run -m scalerank:gamma=1.0 scalerank:gamma=0.5 scalerank:gamma=0.0
uv run python path/to/scalerank-web/research/scripts/compare_rs.py
uv run pytest path/to/scalerank-web/research/tests
```

The research notes are rebuilt from those results with

```bash
uv run python path/to/scalerank-web/research/scripts/build_overview.py
```

which writes `web/public/docs/index.html` (English) and `ja.html`
(Japanese); pushing them to `main` publishes them. Each script's docstring
lists the runs it needs. For the stability section, run drbench's
`scripts/stability.py` with the ScaleRank specs added, e.g.
`--seed-methods ... scalerank:gamma=1.0` (see its `--help`).

The code and experiments up to November 2026 were developed on the
`proto/scalerank` branch of drbench (tag `scalerank-prototype-2026-11`).
