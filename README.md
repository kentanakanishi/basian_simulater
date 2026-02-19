# ベイズ最適化 インタラクティブ可視化

ランダムに生成した2変数関数 f(x,y) に対して、**GS-TS（Gaussian Process + Thompson Sampling）** アルゴリズムで最大値を対話的に探索・可視化するWebアプリです。

## デモ

| 操作 | 内容 |
|---|---|
| 🔄 関数をリセット | 新しいランダム関数を生成し、探索をリセット |
| 🎯 次の候補を探索 | GS-TS を1ステップ実行し、次の評価点を提案 |
| 表示レイヤー切替 | 真の関数 / GP平均 μ / 不確かさ σ をカラーマップで表示 |

## アルゴリズム

### 関数生成

以下の初等関数をランダムな係数で4〜7項、線形結合して f(x,y) を生成します。

| 項の種類 | 式 |
|---|---|
| 正弦波 | w·sin(ax + by + c) |
| 余弦波 | w·cos(ax + by + c) |
| ガウシアンバンプ | w·exp(-((x−cx)² + (y−cy)²) / σ²) |
| 双曲線正接 | w·tanh(ax + by) |
| 2次多項式 | w·(ax² + by² + cxy) |

定義域: x, y ∈ [−3, 3]、グリッド解像度: 60×60

### GS-TS（Gaussian Process + Thompson Sampling）

```
1. 現在の観測点 {(xᵢ, yᵢ, f(xᵢ,yᵢ))} で GP をフィット
   カーネル: ConstantKernel × RBF + WhiteKernel

2. グリッド全点で事後分布 (μ, σ) を予測

3. Thompson Sampling:
   sample(x) = μ(x) + ε·σ(x),  ε ~ N(0,1)

4. argmax(sample) を次の評価点として提案

5. 真の f(x,y) で評価 → 観測に追加 → 1 へ戻る
```

初期観測点: ランダム3点

## セットアップ

### 前提

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

### インストール・起動

```bash
# 依存関係のインストール
uv sync

# 起動 (http://localhost:5000)
uv run python app.py
```

## 技術スタック

| レイヤー | 技術 |
|---|---|
| バックエンド | Python / Flask |
| GP実装 | scikit-learn `GaussianProcessRegressor` |
| 数値計算 | NumPy / SciPy |
| 可視化 | Plotly.js |

## プロジェクト構成

```
basian_simulater/
├── app.py              # Flask サーバー・GP計算・関数生成
├── pyproject.toml      # プロジェクト定義（uv）
├── uv.lock             # 依存関係ロックファイル
├── templates/
│   └── index.html      # メイン画面
└── static/
    ├── main.js         # Plotly描画・API通信
    └── style.css       # ダークテーマUI
```
