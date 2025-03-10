# forcefield

このディレクトリは、微分可能なアプローチを用いた力場パラメータの最適化に関するものである。

## ディレクトリ構成

- `archive/`: 手書きのメモなど
- `mbar_lj.ipynb`: 一次元のレナードジョーンズ系の力場パラメータ最適化に関するノートブック
- `tram/`: tramに関するコード
- `mbar_alanine_dipeptide/`: alanine-dipeptideの二面角に関する力場パラメータのコード
    - `data`: amberの力場ファイル
    - `analysis.ipynb`: 学習後のトラジェクトリを解析するノートブック
    - `mbar_alanine_dipeptide.ipynb`: 二面角のパラメータを最適化するノートブック
    - `mbar_coulomb.ipynb`: クーロン相互作用のパラメータを最適化するノートブック
    - `sim_dihedral.ipynb`: `mbar_alanine_dipeptide.ipynb`の学習に使用するシミュレーションを流すノートブック
    - `sim_qoulomb.ipynb`: `mbar_coulomb.ipynb`の学習に使用するシミュレーションを流すノートブック
    - `sim_target.ipynb`: ターゲットのシミュレーションを流すノートブック
    - `alanine-dipeptide-nowater.pdb`: alanine-dipeptideのpdbファイル
    - `sim.py`: openmmを用いてシミュレーションを流すpythonファイル