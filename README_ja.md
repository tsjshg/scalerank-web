# scalerank-web

[English](README.md)

**公開ページ：<https://tsjshg.github.io/scalerank-web/>**

研究ノート（手法ができるまでの経緯、ベンチマーク、これまでの主な実験）：<https://tsjshg.github.io/scalerank-web/docs/ja.html>（[English](https://tsjshg.github.io/scalerank-web/docs/)）

**グループの中も、グループどうしの関係も読める地図。**

ブラウザの中だけで動く、表形式データ向けの次元削減ツール（開発中）。
ファイルは手元で読み込まれ、どこにも送信されない。目標は 10 万点。
手法そのもの、Python パッケージ、ベンチマークは [tsjshg/scalerank](https://github.com/tsjshg/scalerank) にある。

## ScaleRank がすること

ScaleRank は、表（行が対象、数値の列が特徴量）を 2 次元の地図にする。データ上で似ている対象ほど、地図上でも近くに置かれる。

高次元データのすべての距離を 2 次元で保つことはできないので、どの手法も何を残すかを選んでいる。
t-SNE と UMAP は各点のすぐ近くの近傍を残すが、グループ同士の配置は失われやすい。
PCA や MDS は大きな配置を残すが、局所の細部はぼやける。

ScaleRank は、2 次元で保てる範囲でその両方を保つ。[drbench](https://github.com/tsjshg/drbench) の 8 データすべてで、局所と大域の両方で ScaleRank を上回る手法はなかった（パラメータを 30 通り調整した t-SNE・UMAP も含む）。全体の配置（データと地図の距離相関 0.79、t-SNE 0.52、UMAP 0.37）は、乱数の seed を変えてもほとんど変わらない。

さらに、両者の比重を設定として選べる。扱うのは近傍の**順位**、つまり各点から見て 1 番目、2 番目、…、K 番目に近いのはどの点か、という情報である。
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

手法そのもの、Python パッケージ、研究用のコードは [tsjshg/scalerank](https://github.com/tsjshg/scalerank) にあります。このレポジトリはブラウザのアプリです。

| パス | 内容 |
|---|---|
| `crates/scalerank-wasm` | `scalerank` と `scalerank-baselines`（UMAP、t-SNE）のブラウザ用 WASM バインディング（wasm-bindgen、単一スレッド、SIMD128） |
| `web` | Web UI（Vite + TypeScript）。計算は Web Worker、描画は WebGL2 |
| `web/public/docs` | 研究ノート。scalerank レポジトリの `research/scripts/build_overview.py` で生成する |

## 開発

```bash
# wasm32-unknown-unknown ターゲットと wasm-bindgen-cli 0.2.129 が必要
cd web
npm install
npm run wasm   # crates/scalerank-wasm をビルドして web/src/wasm に出力
npm run dev    # http://localhost:5173
```

`main` に push するたびにアプリがビルドされ、GitHub Pages に公開される（`.github/workflows/pages.yml`）。
`web/public/docs` の研究ノートは、scalerank レポジトリで drbench の結果から生成してここにコミットする（scalerank の `research/README.md`）。
