# B4 物理判定レポート: Σq/Σr vs Σq/r

## 発見

修士論文 第 3.5 節のクーロン電位記述は

$$V_{rec}(\mathrm{cell}) = \sum_{i \in \mathrm{rec}, |r_i - \mathrm{cell}| < r_{cut}} \frac{q_i}{|r_i - \mathrm{cell}|}$$

である。これは古典的なクーロン和で、物理的に正しい電位計算。

しかし `train_param-apart.ipynb` cell 2 の `assign_Re_potential!` 実装は

```julia
grid_charge .= 0
spread_nearest_add!(grid_charge, ..., charge_score)     # Σq at nearest cell
grid_dis   .= 0
calculate_distance!(grid_dis, ..., rcut=8)              # Σ sqrt(d) within rcut
grid_imag = grid_charge ./ grid_dis                      # ← B1 バグは修正済み
```

となっており、実際に計算されているのは

$$V_{\mathrm{bug}}(\mathrm{cell}) = \frac{\sum_{i \text{ nearest}(i)=\mathrm{cell}} q_i}{\sum_{i \in N_{rcut}(\mathrm{cell})} |r_i - \mathrm{cell}|}$$

「単一セルに入る電荷の和」を「rcut 内原子との距離の和」で割るという、物理的には意味不明な量。

## 影響量

A-3 レポートによれば B1 (grid_imag 非更新バグ) 修正前後で `docking_score_elec` の出力は 1KXQ 上位 10 ポーズで +1171〜+1455 のシフト。学習後の β も 3.0→3.197 と僅かしか動かなかった。

「Σq/Σr」を「Σq/r」に置き換えた場合、同じ物理量を計算する別式になるため:
- 各セルの電位値は一般に大きく変わる（分母 Σr >> 1 が多く、修論式では電位が小さめに出る）
- 勾配 dL/dβ, dL/dchs[l] が scale 変化
- 学習収束パラメータが変わる（β はより意味ある値に引かれるはず）

## 実装

Python 側は両方 `docking_torch/src/zdock/spread.py` に用意済み:

- `calculate_distance(...)`: Σ sqrt(d) — 修論と現コード
- `spread_neighbors_coulomb(grid, xyz, charges, rcut, ...)`: Σ q/d — 物理正版

`docking_score_elec` に `potential_mode: Literal["sum_q_over_sum_r", "coulomb"] = "sum_q_over_sum_r"` オプションを後述のパッチで追加可能。

## 選択肢

| | 修論再現優先 (A) | 物理正しさ優先 (B) |
|---|---|---|
| ELEC 実装 | 現行 Σq/Σr | Σq/r |
| 修論スコア数値 | ✓ 同一 | ✗ 異なる |
| 修論 Rank 42→10 再現 | ✓ 期待できる | ✗ 収束先が変わる |
| 汎化性能（tests 1CGI/1ZHI） | 修論同等 | 向上見込み |
| β パラメータの意味 | 弱い（学習中あまり動かない） | 強い（正しい寄与） |
| 学習後モデルを他研究者が使う場合 | 要内部注記 | そのまま使える |

## 推奨

個人的には **(B) 物理正しさ優先** を推奨:

1. 修論の numerical 結果再現は「docking.jl + notebook をそのまま走らせる」でいつでも可能
2. PyTorch 移植の意義は「今後の発展・汎化」なので、バグ入り物理より正しい物理の方が将来的に有益
3. B1 fix の時点で既に厳密な修論再現は崩れているため、B4 修正も含めて「修論の思想を実装した版」と位置付ける

ただしユーザ判断で (A) にする場合は、現行 (修正不要) のまま継続するだけで済む。

## この判定を保留するための構造

仮にユーザ判断を後日に持ち越す場合、`docking_score_elec(..., potential_mode="sum_q_over_sum_r"|"coulomb")` を切替可能にしておけば、将来 flag 切替で両方ベンチマーク可能。パッチ案は `docking_torch/src/zdock/score.py` の `# Precompute receptor ELEC potential slabs` セクションに `if potential_mode == "coulomb": ...` を追加するだけ。
