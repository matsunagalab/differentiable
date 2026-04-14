# CLAUDE.md — `docking/` プロジェクト指針

このディレクトリは Julia による **microdifferentiable ZDOCK**（微分可能なタンパク質剛体ドッキング）の研究コードである。詳細は `README.md` と `master_thesis/ICS-25M-23MM336.pdf` を参照。

## 目的

- ZDOCK のドッキングスコア関数 $S = \alpha S_{SC} + S_{IFACE} + \beta S_{ELEC}$ を微分可能に再実装。
- Flux.jl + ChainRulesCore でパラメータ（$\alpha, \beta$, IFACE 144 個, 電荷 11 個 = 計 157）を Decoy データから勾配降下法で最適化する。

## コードを触るときの注意

- **コア実装は `docking.jl`**。MDToolbox の一部として書かれており、`TrjArray` 型を使う。CUDA/CUDA カーネル版が一部の関数（`spread_*!`, `rotate!`）に存在する。
- 学習対象スコアの本体は `docking_score(receptor_org, ligands_org, alpha, iface_ij)` で、その逆伝播は同ファイル内の `ChainRulesCore.rrule` で手書き定義されている。パラメータの形（IFACE はペアで対称、電荷は原子タイプ毎）に従って勾配が組まれているので、スコア関数を変更する際は rrule の同期更新が必須。
- グリッド生成は `generate_grid`（既定 spacing=1.2 Å）、FFT 探索は `compute_docking_score_with_fft`、エントリポイントは `docking(...)`。
- 原子タイプ ID 付与（`set_atomtype_id`）、vdW 半径（`set_radius`）、電荷（`set_charge`）はすべて残基/原子名の if-else で記述。新しい残基や原子型を扱う際はここを拡張する。

## ノートブックの役割分担

- 学習を回す → `train_param-apart.ipynb`（最終版）。古い実装や個別検討は `train.ipynb`, `train_elec.ipynb`。
- テスト（学習未使用タンパク質での評価）→ `train_1CGI.ipynb`, `train_1ZHI.ipynb`。
- 学習結果の可視化（IFACE / 電荷パラメータのヒートマップ等）→ `analyze.ipynb`。

## データ

- `protein/<PDBID>/` … 11 種のタンパク質ペアの `_r_u.pdb.ms`（受容体）/ `_l_u.pdb.ms`（リガンド）と ZDOCK 出力 `*.zd3.0.2.fg.fixed.out`、RMSD ファイル `*.rmsds`、`complex.<i>.pdb`。
- `decoys_bm4_zd3.0.2_6deg_fixed/` は ZDOCK 公式の Decoy データセット（6度刻み）。`makefile` を実行して `create_lig` をビルドし、構造ファイルを生成する想定。
- 評価指標：
  - **Hit** = スコア上位 100 構造のうち RMSD ≤ 2.5 Å のもの（recall 相当）
  - **Rank** = スコア降順で最初の Hit が現れる順位（precision 相当、低いほど良い）

## 学習設定の既定（修論時点）

- 初期値：$\alpha=0.01$, $\beta=3.0$、IFACE は経験表（`get_iface_ij()`）、電荷は `set_charge` の値。
- Optimizer は Adam、200 epoch。
- 損失関数：「最適化前のスコア分布」を Positive（最大値）と Negative（最小値）に並び替えた**理想スコア分布**との MSE。Positive/Negative の不均衡を補正するため、それぞれのデータ数で割って和を取る（`train_param-apart.ipynb` 内 `loss(...)` 参照）。

## 作業上の流儀

- スコア関数とその rrule、原子タイプ表は密結合。片方だけ編集して動かさないこと。
- パラメータ数の感覚値：IFACE は 12×12 だがペアで対称なので独立 78 個 + 対角 12 個程度の自由度。修論は対称性を考慮した微分定義を `rrule` に入れている。
- GPU パスは `CuArray`/`CuDeviceArray` 多重ディスパッチで実装されている。CPU 動作確認だけして PR にしない。
- `forcefield/` ディレクトリは別プロジェクト（MBAR 等）。本プロジェクトとは独立なので参照しない。
