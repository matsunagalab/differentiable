# Differentiable ZDOCK

Julia 実装による「微分可能なドッキングシミュレーション」。剛体タンパク質ドッキングソフトウェア ZDOCK のスコア関数を微分可能に再実装し、Decoy データセットから勾配降下法でスコアパラメータを最適化する研究プロジェクト。

詳しい背景・理論・結果は [`master_thesis/ICS-25M-23MM336.pdf`](master_thesis/ICS-25M-23MM336.pdf)（柿沼 孝紀, 埼玉大学 松永研究室 修士論文, 2025年2月）を参照。

## 研究概要

- ZDOCK [Chen and Weng, *Proteins*, 2003] のドッキングスコア関数

  $$ S = \alpha S_{SC} + S_{IFACE} + \beta S_{ELEC} $$

  を構成する 3 項（形状相補性 SC、Interface Atomic Contact Energies IFACE、静電エネルギー ELEC）を Julia で実装し、各パラメータについて自動微分可能にしている。
- 最適化対象は $\alpha$, $\beta$, IFACE スコアの 144 パラメータ（12×12 原子タイプペア）、電荷パラメータ 11 個の計 **157 パラメータ**。
- ZDOCK が提供する Decoy データセット（不正解構造と正解構造の混合, 6度刻み回転）から、Hit/Rank の指標が改善するよう勾配降下法（Adam）でパラメータを学習する。
- 微分は Julia の Flux.jl を用いて定義し、ChainRulesCore による rrule を `docking.jl` 内の `docking_score` に対して実装している。

## ディレクトリ構成

| パス | 内容 |
|------|------|
| `docking.jl` | MDToolbox 内のドッキング計算を行うコア実装。スコア関数（SC / IFACE / ELEC）、グリッド化、FFT ベースの探索、`docking_score` の微分（rrule）を含む |
| `train_param-apart.ipynb` | 本研究で**最終的に使用した**パラメータ学習コード |
| `train_elec.ipynb` | 静電エネルギーの学習を最初に実装した時のコード |
| `train.ipynb` | 卒業研究時のコード |
| `train_1CGI.ipynb`, `train_1ZHI.ipynb` | テストデータ（学習に不使用のタンパク質）で最適化前後のパラメータを評価するコード |
| `analyze.ipynb` | 最適化前後のパラメータのヒートマップを作成するコード |
| `protein/` | 研究で使用したタンパク質（Decoy）データセット。`1BJ1, 1CGI, 1EWY, 1F51, 1KXQ, 1Z5Y, 1ZHI, 2G77, 2H7V, 2VDB, 3D5S` |
| `decoys_bm4_zd3.0.2_6deg_fixed/` | ZDOCK 提供のベンチマークデータセット。`makefile` & `create_lig` でコンパイルし構造ファイルを生成する |
| `1KXQ/` | 1KXQ 用 Decoy をデコードした構造ファイル（`complex.*.pdb`）と ZDOCK 出力（`*.zd3.0.2.fg.fixed.out`）, RMSD ファイル |
| `master_thesis/` | 修士論文 PDF |

各 `protein/<PDBID>/` には以下が含まれる：
- `<PDBID>_r_u.pdb.ms` … 受容体（Receptor）構造
- `<PDBID>_l_u.pdb.ms` … リガンド（Ligand）構造
- `<PDBID>.zd3.0.2.fg.fixed.out` … ZDOCK の予測ポーズ出力
- `<PDBID>.zd3.0.2.fg.fixed.out.rmsds` … 各ポーズの正解構造からの RMSD
- `complex.<i>.pdb` … 上位ポーズを複合体として書き出した PDB

## ドッキングスコアの構成

論文 第3章 を要約：

- **形状相補性 $S_{SC}$**：受容体・リガンドの $N\times N\times N$ グリッド上で、表面/コアの離散関数を畳み込み、表面どうしは $-9$、コアどうしは $-81$、表面–コア間は $-27$ のペナルティで評価。
- **IFACE $S_{IFACE}$**：12 個の原子タイプ間のペア毎に経験的に求められた $144$ パラメータを用い、グリッド 6Å 以内の脱溶媒和エネルギーを $k/2$ FFT で評価。
- **静電エネルギー $S_{ELEC}$**：受容体側はグリッド 8Å 以内の電位、リガンド側は原子電荷 × $-1/r_{rl}$ 倍の畳み込み。電荷は 11 種類の原子タイプ毎に与える。

前処理として SASA（溶媒露出面積）を Golden Section Spiral 法で計算し、表面/コア原子を判定（水分子半径 1.40 Å、表面閾値 1 Å²）。

## 学習データと結果（修士論文より）

- 学習：1KXQ, 1F51, 2VDB の 3 タンパク質、Decoy 上位 100 構造、200 epoch、計算時間 13–14 時間。
- テスト（汎化性能評価）：1CGI, 1ZHI（学習に不使用）。
- 結果：学習に用いた 3 タンパク質では Positive 構造の Rank が改善（例：1KXQ Rank 42→10）。テストデータでも僅かながら改善が確認された。

今後の課題として、(1) 学習タンパク質種の拡大、(2) ペナルティ係数 $\rho$ 等の追加パラメータの最適化、(3) 交差エントロピー等のより高度な損失関数の導入、(4) 計算の並列化が挙げられている。
