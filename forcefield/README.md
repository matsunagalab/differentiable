# forcefield

このディレクトリは、微分可能アプローチによる力場パラメータ最適化のディレクトリである。

## ディレクトリ構成

- `archive/`: 手書きのメモなど
- `mbar_lj.ipynb`: 一次元のレナードジョーンズ系の力場パラメータ最適化に関するノートブック
- `tram`: tramに関するコード
- `mbar_alanine_dipeptide/`: alanine-dipeptideの二面角に関する力場パラメータのコード
    - `data`: amberの力場ファイル
    - `analysis.ipynb`: 学習後のトラジェクトリを解析するノートブック
    - `mbar_alanine_dipeptide.ipynb`: 二面角のパラメータを最適化するノートブック