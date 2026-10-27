# scalerank-web

ブラウザの中だけで動く次元削減ツール（開発中）。データは外に出ない。
表形式のデータを読み込み、ScaleRank（近傍の順位を、ユーザーが選んだスケールの範囲で保つ手法）と、
明示的に指示されたときには t-SNE / UMAP で 2 次元に埋め込む。目標は 10 万点。

## 構成

| パス | 内容 |
|---|---|
| `crates/scalerank-core` | 本体（Rust、依存は任意の rayon のみ）。kNN（総当たり、NN-descent）、PCA、N² を使わない順位テーブル、ScaleRank、UMAP、t-SNE（Barnes-Hut）、評価指標 |
| `crates/scalerank-py` | 検証用の Python バインディング（PyO3）。[drbench](https://github.com/tsjshg/madr) で Python の試作、umap-learn、openTSNE と比べる |

UMAP は umap-learn、t-SNE は openTSNE の既定値と更新規則に合わせている。
違いは、UMAP の初期化がスペクトルではなく PCA であることと、t-SNE に FFT 加速（openTSNE が 1 万点超で使う）がないこと。

WASM 版と Web UI はこのあと追加する。

## 開発

```bash
cargo test -p scalerank-core
# drbench の環境に Python バインディングを入れる
uv pip install --python ../madr/drbench/.venv/bin/python ./crates/scalerank-py
```
