# scalerank-web

[日本語](README_ja.md)

**Try it: <https://tsjshg.github.io/scalerank-web/>**

A dimensionality-reduction tool for tabular data that runs entirely in the
browser. Files are read locally and nothing is uploaded. It is in development;
the goal is 100,000 points.

## What ScaleRank does

ScaleRank turns a table (rows are items, numeric columns are features) into a
2-D map in which items that are similar in the data sit close together.

No 2-D map can keep every distance of many-dimensional data, so every method
chooses what to keep. t-SNE and UMAP keep each point's closest neighbours but
tend to lose how the groups are arranged relative to each other. PCA and MDS
keep the large-scale layout but blur local detail.

ScaleRank makes this choice a setting. It works with neighbour *ranks*: for
each point, which points are its 1st, 2nd, …, K-th nearest. During the
optimisation it repeatedly takes a point, a neighbourhood size K, one point
inside that neighbourhood and several outside it. Then it moves the map so the
inside point ends up closer than the outside ones.

The **scale** setting (γ) controls how often each K is drawn, with probability
proportional to K^−γ:

- **γ = 0** weights all neighbourhood sizes equally, which favours the global
  layout.
- **Larger γ** concentrates on small neighbourhoods, which favours local
  detail.

It never builds the N × N table of ranks. Exact ranks come from each point's
100 nearest neighbours, and larger ranks are estimated from random subsamples.

### Reading the quality table

| Column | Meaning |
|---|---|
| **Neighbours** | How many of each point's 10 nearest neighbours in the data are also among its 10 nearest in the map (R_NX at K = 10). 0 = no better than a random map, 1 = all kept. |
| **Scales** | The same score averaged over every neighbourhood size, from the closest neighbour to the whole data set, with small sizes weighted more (log-scaled AUC of R_NX). |
| **Distances** | Rank correlation between distances in the data and in the map: is the global arrangement of groups kept? |
| **Time** | Time to compute the map in the browser. |

Higher is better in every column but Time. No map wins every column, because
keeping local neighbours and keeping the global layout pull against each other.

### t-SNE and UMAP

Both are included for comparison and run only when you ask for them. They
follow the defaults and update rules of openTSNE and umap-learn, with two
differences:

- UMAP starts from PCA instead of a spectral layout.
- t-SNE uses Barnes-Hut throughout, without FFT acceleration.

## Repository layout

| Path | Contents |
|---|---|
| `crates/scalerank-core` | The core in Rust; its only dependency is the optional rayon. kNN (brute force, NN-descent), PCA, N²-free rank tables, ScaleRank, UMAP, t-SNE (Barnes-Hut) and quality metrics |
| `crates/scalerank-py` | Python bindings (PyO3), used to check the Rust code against the Python prototype, umap-learn and openTSNE in [drbench](https://github.com/tsjshg/madr) |
| `crates/scalerank-wasm` | WebAssembly bindings for the browser (wasm-bindgen, single-threaded, SIMD128) |
| `web` | The web app (Vite + TypeScript). Computation runs in a Web Worker and drawing uses WebGL2 |

## Development

```bash
cargo test -p scalerank-core
# install the Python bindings into drbench's environment
uv pip install --python ../madr/drbench/.venv/bin/python ./crates/scalerank-py

# web app (needs the wasm32-unknown-unknown target and wasm-bindgen-cli 0.2.129)
cd web
npm install
npm run wasm   # builds crates/scalerank-wasm into web/src/wasm
npm run dev    # http://localhost:5173
```

Every push to `main` builds the app and deploys it to GitHub Pages
(`.github/workflows/pages.yml`).
