# scalerank-web

ブラウザの中だけで動く次元削減ツール（開発中）。データは外に出ない。
表形式のデータを読み込み、ScaleRank（近傍の順位を、ユーザーが選んだスケールの範囲で保つ手法）と、
明示的に指示されたときには t-SNE / UMAP で 2 次元に埋め込む。目標は 10 万点。

## 構成

| パス | 内容 |
|---|---|
| `crates/scalerank-core` | 本体（Rust、依存は任意の rayon のみ）。kNN、PCA、N² を使わない順位テーブル、ScaleRank の最適化 |
| `crates/scalerank-py` | 検証用の Python バインディング（PyO3）。[drbench](https://github.com/tsjshg/madr) で Python の試作と比べる |

WASM 版と Web UI はこのあと追加する。

## 開発

```bash
cargo test -p scalerank-core
# drbench の環境に Python バインディングを入れる
uv pip install --python ../madr/drbench/.venv/bin/python ./crates/scalerank-py
```
