# scalerank-web

[English](README.md)

**公開ページ：<https://tsjshg.github.io/scalerank-web/>**

研究ノート（手法ができるまでの経緯、ベンチマーク、これまでの主な実験）：<https://tsjshg.github.io/scalerank-web/docs/ja.html>（[English](https://tsjshg.github.io/scalerank-web/docs/)）

ブラウザの中だけで動く、表形式データ向けの次元削減ツール（開発中）。
ファイルは手元で読み込まれ、どこにも送信されない。目標は 10 万点。

## ScaleRank がすること

ScaleRank は、表（行が対象、数値の列が特徴量）を 2 次元の地図にする。データ上で似ている対象ほど、地図上でも近くに置かれる。

高次元データのすべての距離を 2 次元で保つことはできないので、どの手法も何を残すかを選んでいる。
t-SNE と UMAP は各点のすぐ近くの近傍を残すが、グループ同士の配置は失われやすい。
PCA や MDS は大きな配置を残すが、局所の細部はぼやける。

ScaleRank は、この選択を設定として明示する。扱うのは近傍の**順位**、つまり各点から見て 1 番目、2 番目、…、K 番目に近いのはどの点か、という情報である。
最適化では次の手順を繰り返す。

1. ある点と、近傍の大きさ K を選ぶ。
2. その近傍の内側から 1 点、外側から数点を選ぶ。
3. 内側の点が外側の点より近くなるように地図を動かす。

**スケール**（γ）は、各 K を選ぶ頻度を決める。K は K^−γ に比例した確率で選ばれる。

- **γ = 0** はすべての近傍の大きさを等しく扱い、大域的な配置を重視する。
- **γ が大きいほど**小さな近傍に集中し、局所の細部を重視する。

N × N の順位表は作らない。順位は、各点の近傍 100 点までは正確に求め、それより大きな順位はランダムな部分標本から推定する。

### 品質表の読み方

| 列 | 意味 |
|---|---|
| **Neighbours** | 各点の近傍 10 点が、地図上でもどれだけ近傍 10 点に入っているか（K = 10 の R_NX）。0 はランダムな地図と同じ、1 はすべて残っている。 |
| **Scales** | 同じ指標を、最も近い近傍からデータ全体までのあらゆる近傍の大きさで平均したもの。小さな近傍ほど重く扱う（R_NX の対数スケール AUC）。 |
| **Distances** | データと地図での距離の順位相関。グループの大域的な配置が保たれているか。 |
| **Time** | ブラウザでの計算時間。 |

Time 以外はすべて大きいほど良い。ただし局所の近傍と大域の配置はトレードオフなので、全列で一番になる地図はない。

### t-SNE と UMAP

比較のために含めており、ユーザーが指示したときだけ計算する。どちらも openTSNE と umap-learn の既定値と更新規則に合わせている。違いは次の 2 点。

- UMAP の初期配置はスペクトルではなく PCA。
- t-SNE は FFT による加速を使わず、常に Barnes-Hut で計算する。

## 構成

| パス | 内容 |
|---|---|
| `crates/scalerank-core` | 本体（Rust、依存は任意の rayon のみ）。kNN（総当たり、NN-descent）、PCA、N² を使わない順位テーブル、ScaleRank、UMAP、t-SNE（Barnes-Hut）、評価指標 |
| `crates/scalerank-py` | 検証用の Python バインディング（PyO3）。[drbench](https://github.com/tsjshg/madr) で Python の試作、umap-learn、openTSNE と比べる |
| `crates/scalerank-wasm` | ブラウザ用の WASM バインディング（wasm-bindgen、単一スレッド、SIMD128） |
| `web` | Web UI（Vite + TypeScript）。計算は Web Worker、描画は WebGL2 |

## 開発

```bash
cargo test -p scalerank-core
# drbench の環境に Python バインディングを入れる
uv pip install --python ../madr/drbench/.venv/bin/python ./crates/scalerank-py

# Web UI（wasm32-unknown-unknown ターゲットと wasm-bindgen-cli 0.2.129 が必要）
cd web
npm install
npm run wasm   # crates/scalerank-wasm をビルドして web/src/wasm に出力
npm run dev    # http://localhost:5173
```

`main` に push するたびにアプリがビルドされ、GitHub Pages に公開される（`.github/workflows/pages.yml`）。
`web/public/docs` の研究ノートは drbench の結果を使うので、[drbench](https://github.com/tsjshg/madr) 側で `scripts/build_overview.py --publish <web/public/docs のパス>` を実行して生成する。
