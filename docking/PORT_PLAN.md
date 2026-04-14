# Plan: `docking/` の Julia バグ修正 + PyTorch 移植（テスト駆動開発）

## Context

`/Users/yasu/gdrive/work/differentiable/docking/docking.jl` + `train_param-apart.ipynb` で構成される微分可能 ZDOCK を PyTorch へ移植する。ただし、事前調査で **ノートブック側の「本番コード」に複数の深刻なバグ**があることが判明したため、作業を 2 段階に分ける：

**Stage A: Julia 正準化＆バグ修正**
1. ノートブックの関数定義を抽出し `docking/docking_canonical.jl` として独立した正準ソースにする
2. バグを修正し、物理的に妥当な出力に戻す
3. 修正前後のスコアを比較し、修論結果と突き合わせる

**Stage B: TDD で PyTorch へ移植**
1. 修正済み Julia を「ゴールデン」として JLD2 に出力
2. 関数毎に PyTorch 実装 → Julia と数値比較

## Julia 実行環境の調査結果（動作確認済み）

| 項目 | 状態 |
|---|---|
| `julia` 1.11.6, `MDToolbox` 0.1.2, `Flux` 0.16.5 | ✓ |
| `ChainRulesCore`, `CUDA`, `JLD2` | 本調査で `Pkg.add` 済み |
| macOS (CUDA 非機能) | CPU パスで参照出力を生成。`CUDA.functional() == false` |
| `docking.jl` の `include` | ✓ |
| 1KXQ での `docking_score` end-to-end 実行 | ✓ `-3311.858` を返す（4s、JIT 初回） |

## 発見されたバグ一覧

ノートブック cell 2, 4, 5, 6, 62（`train_param-apart.ipynb`）と `docking.jl` の精読で発見：

### 致命的（物理/学習シグナルを壊す）

**B1. `assign_Re_potential!` / `assign_Li_potential!` が silent no-op**（cell 2）
```julia
grid_imag = grid_charge ./ grid_dis   # ❌ ローカル変数への再束縛
# 正：grid_imag .= grid_charge ./ grid_dis
```
引数で受け取った `grid_imag` が**一切更新されない**。呼び出し側（`docking_score_elec` L775, `rrule` L1006, L1122, L1148）で `sum(grid_real .* grid_imag)` を計算するが、`grid_imag` は直前の IFACE 計算の残骸（stale）なので、**ELEC 項は実質デタラメ**。修論で β の絶対値が 3.0 → 3.197 と微増しかしていないのは、この項が正しく動いていないため勾配が弱いことと整合する。

**B2. `loss` 関数が最初の項しか使わない**（cell 62）
```julia
function loss(rl_1, rl_2, rl_3, t1_1, t1_2, t2_1, t2_2, t3_1, t3_2)
    l = sum(Flux.Losses.mse.(m.(rl_1), t1_1)) / 9      # ← これだけが l に代入
    + sum(Flux.Losses.mse.(m.(rl_1), t1_2)) / 91        # ← 行頭 + で新しい式、捨てられる
    + sum(Flux.Losses.mse.(m.(rl_2), t2_1)) / 10
    ...
    return l
end
```
Julia では行末に演算子が無いと改行で式が終わる。6 項のうち**最初の 1 項しか損失に寄与していない**。修正は各行末に `+` を置くか、`l = (項1 + 項2 + ...)`  で囲う。学習シグナルが 6 分の 1 に弱まっていた可能性。

**B3. `set_charge` のターミナル O 分岐がデッドコード**（cell 2, L20–23）
```julia
elseif ta.resname[iatom] == "O"    # ❌ resname は "ARG"/"ALA"/... なので "O" に一致しない
    charge[iatom] = 10
elseif ta.resname[iatom] == "OXT"  # ❌ 同上
    charge[iatom] = 2
# 正：atomname[iatom] == "O" / "OXT"
```
全てのバックボーン O（type 10、電荷 -0.5）とターミナル OXT（type 2、電荷 -1.0）は else 節に落ちて type 8（CA、電荷 0.0）に化ける。タンパク質電荷の偏りがほぼ消え、電場計算が無意味に近い。

### 物理的に疑わしい（判定要）

**B4. `calculate_distance!` が距離の和を返す**（cell 2, L152）
```julia
grid[ix, iy, iz] += sqrt(d)
```
結果として `grid_imag = Σq / Σr` になる（B1 を直せば）。クーロン電位 `Σ q_i / r_i` とは別物。修士論文（第3.5節）の意図は `Σ q / r` なので、正しくは
```julia
grid[ix, iy, iz] += weight[iatom] / sqrt(d)
```
とし、`weight` = `charge_score[l]`、そして `assign_Re_potential!` の除算を削除するのが自然。

### 非致命的（保守性の問題のみ）

**B5. `idx` 変数の重複利用**（cell 4 L756-767）：内側ループで `idx = receptor.atomtype_id .== j` と再束縛するが、外側ループ頭で `idx = ligands.atomtype_id .== i` が再初期化するので動作上は問題なし。読みづらいだけ。

**B6. `receptor.mass .= iface_ij[k]` の全体ブロードキャスト**（cell 4 L762 ほか）：scalar を全原子 mass に書き込んでから `[idx]` で部分取得。非効率だがバグではない。

**B7. `docking.jl` 側の `set_charge`（L880–895）の `ta.atomname` → `atomname` 未定義**：ノートブック版で上書きされるため実使用上は無害だが、docking.jl 単体では壊れている。

### 構造的な問題

**S1. ノートブック上書きが `docking.jl` と重複**：`set_charge`, `generate_grid`, `spread_*!`, `assign_sc_*`, `assign_Rij!`, `assign_Li!`, `docking_score` が両方に存在し、**ノートブック版が勝つ**が、どちらが真実かコード読みではわかりにくい。

**S2. docking.jl にノートブック限定の関数が無い**：`docking_score_elec`, `get_charge_score`, `assign_Li_charge!`, `assign_Re_charge!`, `calculate_distance!`, `assign_Re_potential!`, `assign_Li_potential!`, `assign_sc_*_plus!`, `assign_sc_*_minus!`, そして最重要の `rrule(::typeof(docking_score_elec))`。

## Stage A：Julia 正準化＆バグ修正

### A-1. ノートブックから正準 Julia ソースを抽出

新規ファイル `docking/docking_canonical.jl` を作成。内容：
1. `docking.jl` を include（`get_iface_ij`, `get_acescore`, `set_atomtype_id`, `set_radius`, `compute_sasa`, `golden_section_spiral`, `rotate!`, `generate_ligand`, `filter_tops!`, `docking` などバグの無い関数を継承）
2. ノートブック cell 2, 4, 5, 6 の関数定義を書き出し（`set_charge`, `get_charge_score`, `calculate_distance!`, `assign_*_potential!`, `assign_*_charge!`, `generate_grid`（上書き）, `spread_*!`（上書き）, `assign_sc_*_plus!`, `assign_sc_*_minus!`, `assign_Rij!`, `assign_Li!`, `docking_score`（上書き、spacing=3.0）, `docking_score_elec`, `rrule(docking_score_elec)`）

抽出はスクリプト化し、`tools/extract_notebook.py`（Python + json）で再現可能にする。

### A-2. バグ修正

優先順：**B1 → B2 → B3 → B4**（致命→物理的疑問）。各修正は独立コミット。

- **B1 修正**：`assign_Re_potential!` と `assign_Li_potential!` の `grid_imag = ...` を `grid_imag .= ...` に
- **B2 修正**：`loss` 関数を `l = (term1 + term2 + ... + term6)` の形に統一
- **B3 修正**：`set_charge` の `ta.resname[iatom] == "O"/"OXT"` を `atomname[iatom] == "O"/"OXT"` に
- **B4 判定**：ユーザと相談。修論記述（3.5 節：`Σ q / r`）と現コード（`Σq / Σr`）のどちらを正とするか

## Phase C: Literature-driven ELEC fixes (B10–B15)

論文精読（Chen & Weng 2002, Chen et al. 2003, Mintseris et al. 2007）で
判明した ELEC 実装の根本的な問題。B4 は単なる判断ではなく明確なバグと
確定。ユーザ指示で Julia・PyTorch 両側を修正。

### B10. ELEC が Coulomb でない (Σq/Σr pseudo-quantity)

- **Chen 2002 p284 明記**: "The electrostatics energy... described by the
  **Coulombic formula**. We adopted the approach by Gabb et al., except
  that we used the partial charges in the CHARMM19 potential."
- **修正**: `V(r) = Σⱼ qⱼ / |r − rⱼ|` を直接計算。
  - Julia: `docking_score_elec_coulomb`（`docking_canonical.jl`）
  - Python: `docking_score_elec(..., elec_mode="coulomb")`（デフォルト）

### B11. リガンド側電荷の符号抜け

- **Chen 2003 Eq 2**: `Im[L_PSC+DE+ELEC] = -1 × atom_charge` at nearest cell.
- **修正**: Coulombic path で自然に組み込む（`Σ V_rec × q_lig` の符号が
  相互作用エネルギーになる）。

### B12. ELEC を同原子タイプペアに限定していた

- **Chen 2002**: "all atom pairs" で Coulomb 和を取る。
- **現実装の間違い**: `for l in 1:11` で `ligands.atomtype_id == l` かつ
  `receptor.atomtype_id == l` のペアにのみ限定。LYS-NZ × ASP-OD のような
  **強い異種イオン引力が完全に欠落**していた。
- **修正**: Coulombic path はタイプでグループ化しない（全ペア和）。

### B13. 受容体コア内で V=0 処理がない

- **Chen 2002 p284 明記**: "grid points in the core of the receptor are
  assigned a value of 0 for the electric potential, to avoid contributions
  from non-physical receptor-core/ligand contacts."
- **修正**: `V_rec = V_rec * open_space_mask` で SC shape 内をゼロ化。

### B14. 電荷 LUT が粗すぎる (軽微)

- **論文**: CHARMM19 per-atom partial charges (≥20 残基 × 10+ 原子 = 100+ 値)。
- **現実装**: 11 値のみ（TERM-N, TERM-O, ARG-NH, GLU-OE, ASP-OD, LYS-NZ,
  PRO-N, 他ゼロ）。
- **一次対応**: `partial_charge_per_atom(charge_id, charge_score)` で
  軽量版 per-atom 電荷を生成。本格 CHARMM19 移植は後続課題。

### B15. ELEC 距離カットオフ (軽微)

論文は "all receptor atoms" だが、実装は rcut=8Å。実用上容認、文書化済み。

### 検証

`docking_torch/tests/test_physics.py` で 5 テスト全緑：
- 符号（正電荷ペアで斥力、異符号で引力）
- 1/r スケーリング（float64 rel_err 0）
- 重ね合わせ原理 (|V_A + V_B − V_AB| = 0)
- Coulomb vs legacy が実データで異なる (1KXQ Δ ~3.9)
- autograd が Coulomb パスを通過する

Julia と PyTorch の Coulomb 実装は 1KXQ 10 ポーズで **max rel err 0.00**
（bit-exact）。

### A-3. スコア妥当性チェック

1KXQ で B1 修正前後の以下を比較：
- `score_sc`（αS_SC）の range / 分布
- `score_iface` の range / 分布
- `score_elec` の range / 分布（B1 修正で 0 近辺から意味のある値に変わるはず）
- `docking_score_elec` と `docking_score` の数値差（ELEC 項の寄与が α=0.01, β=3.0 で何 % か）

B4 を修正するかはここで判定。仮に修論が `Σq/Σr` で学習されていたならその異常物理を維持する必要があるかもしれない（修論の数値再現優先）。

### A-4. 軽いスモーク（rrule の勾配の形が正しいか）

`Flux.gradient(m, receptor, ligand)` で α, β, iface, charge_score の勾配形状（scalar, 144-vec, 11-vec）が得られることと、`finite_difference_gradient` と数値的に一致することを確認。B1 修正前後で比較。

### A-5. マイルストーン

| # | 達成条件 | 工数 |
|---|---|---|
| A-1 | `docking_canonical.jl` + `tools/extract_notebook.py` 完成、`include` が macOS CPU で通る | 0.5 日 |
| A-2 | B1, B2, B3 修正（B4 は判定待ち） | 0.5 日 |
| A-3 | スコア妥当性チェックレポート（markdown） | 0.5 日 |
| A-4 | 勾配整合確認（rrule vs 数値微分） | 0.5 日 |

## Stage B：PyTorch 移植（TDD）

### B-1. 依存順の移植フェーズ（変更なし）

| Phase | 対象 |
|---|---|
| 1 | `get_acescore`, `get_iface_ij`, `get_charge_score`, `golden_section_spiral`, `set_atomtype_id`, `set_radius`, `set_charge`, `rotate!` |
| 2 | `compute_sasa`, `generate_grid` |
| 3 | `spread_nearest_*`, `spread_neighbors_*`, `calculate_distance!`（B4 を反映） |
| 4 | `assign_Li!`, `assign_Rij!`, `assign_Li_charge!`, `assign_Re_charge!`, `assign_Re_potential!`, `assign_Li_potential!`, `assign_sc_*_plus!`, `assign_sc_*_minus!` |
| 5 | `docking_score`, `docking_score_elec` 前方計算（autograd は PyTorch 任せ、rrule 不要） |
| 6 | 学習ループ（Adam、修正済み loss） |

### B-2. 参照出力ダンパ

`docking/tests/julia_ref/generate_refs.jl` が `docking_canonical.jl`（バグ修正済み）を include して 1KXQ で各関数を実行、JLD2 に出力。

```
tests/refs/1KXQ/
  phase1_{atomtype_id,radius,charge,iface_ij,charge_score,spiral,rotate}.jld2
  phase2_{sasa,grid}.jld2
  phase3_{spread_nearest,spread_neighbors,calculate_distance}.jld2
  phase4_{assign_Li,assign_Rij,assign_charge,assign_potential,assign_sc_plus,assign_sc_minus}.jld2
  phase5_{docking_score_fwd,docking_score_elec_fwd,docking_score_elec_grad}.jld2
```

### B-3. ディレクトリ構成

```
differentiable/
├── docking/                             # 既存（変更: docking_canonical.jl 追加のみ）
│   ├── docking.jl                       # 触らない
│   ├── docking_canonical.jl             # ← 新規（正準ソース）
│   ├── tools/extract_notebook.py        # ← 新規
│   └── tests/julia_ref/
│       ├── Project.toml
│       └── generate_refs.jl
└── docking_torch/                       # ← 新規（PyTorch プロジェクト）
    ├── pyproject.toml
    ├── src/zdock/
    │   ├── io.py        # PDB パーサ（stdlib のみ）
    │   ├── atomtypes.py # LUT 群
    │   ├── geom.py      # rotate, spiral, grid
    │   ├── sasa.py
    │   ├── spread.py    # scatter_add ベース
    │   ├── score.py     # docking_score / docking_score_elec
    │   └── train.py
    └── tests/
        ├── conftest.py
        ├── test_phase1_*.py
        ├── test_phase2_*.py
        ├── test_phase3_*.py
        ├── test_phase4_*.py
        └── test_phase5_*.py
```

### B-4. 許容誤差

- float64 統一、葉ノード `atol=1e-10 rtol=1e-8`、グリッド散布 `atol=1e-8 rtol=1e-6`、FFT 後 `atol=1e-4 rtol=1e-5`、勾配 `rtol=1e-4 atol=1e-6`。

### B-5. マイルストーン（Stage A 完了後）

| M# | 達成条件 | 工数 |
|---|---|---|
| B-0 | `docking_torch/` ひな形 + `generate_refs.jl` で参照 JLD2 出力完了 | 0.5 日 |
| B-1 | Phase 1 全部移植 + pytest 緑 | 1 日 |
| B-2 | Phase 2 | 1 日 |
| B-3 | Phase 3 | 1.5 日 |
| B-4 | Phase 4 | 2 日 |
| B-5 | Phase 5 前方一致 | 2 日 |
| B-6 | 勾配：`torch.autograd.gradcheck` + Julia 勾配と一致 | 1.5 日 |
| B-7 | 学習 200 epoch 実行、損失が Julia 側と定性的に一致 | 2 日 |

Stage A 2 日 + Stage B 11.5 日 = 約 13.5 日。

## 修正・新規作成するファイル

- 新規：`docking/docking_canonical.jl`（正準ソース、バグ修正込み）
- 新規：`docking/tools/extract_notebook.py`（notebook → .jl 抽出スクリプト）
- 新規：`docking/tests/julia_ref/{Project.toml,generate_refs.jl}`
- 新規：`differentiable/docking_torch/`（PyTorch プロジェクト一式）
- 新規：`docking/PORT_PLAN.md`（本プランのコピー、プロジェクト内にも配置）
- **既存 `docking/docking.jl` は触らない**（ゴールデン原典として温存）
- **既存 `docking/train_param-apart.ipynb` は触らない**（修論時の状態を保存）

## 検証方法

1. **Stage A 完了時**：
   - `julia generate_refs.jl` で 1KXQ の全中間出力 JLD2 が生成される
   - B1 修正前後で `score_elec` が定性的に変化することを確認
   - rrule の勾配と数値微分が一致
2. **Stage B 完了時**：
   - `cd docking_torch && pytest -q` 全緑
   - `α=0.01, β=3.0, 200 epoch` で学習実行、損失が B2 修正後の Julia 損失と定性的に一致
   - 1KXQ の Rank 改善が Julia・PyTorch 両側で得られる

## 未決事項

1. **B4（`calculate_distance!` のクーロン正規化）をどうするか**：
   - (α) 修論通り `Σq / Σr` を維持（現コード）→ バグは B1, B2, B3 のみ修正
   - (β) 物理的に正しい `Σ q/r` に直す → 修論再現は難しくなるが汎化性能は上がるはず
2. **PyTorch の実行先**：macOS CPU / MPS / 別マシンの Linux GPU
3. **PDB パーサ**：自前 50 行 vs Biopython 依存
4. **修論の数値再現をどこまで優先するか**：B2 を直すと損失構造が変わり、修論 Rank 42→10 は再現しない可能性がある。
5. **`docking.jl` の死にコード**（B7）を修正するか放置するか
