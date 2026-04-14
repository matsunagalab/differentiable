# Julia 参照実装テスト

`docking_canonical.jl`（`docking.jl` + notebook 上書き関数 + 本リポジトリで適用したバグ修正）が妥当な出力を返すこと、および将来の移植で同じ出力が再現されることを確認するためのテスト群。

全テストは `docking/` をカレントディレクトリとして実行する：

```bash
cd /Users/yasu/gdrive/work/differentiable/docking
julia tests/julia_ref/<script>.jl
```

## 一覧

| スクリプト | 目的 | 実行時間目安 | 対応レポート |
|---|---|---|---|
| `sanity_check.jl` | `docking_score_elec` の forward スコアを、バグ修正前の raw ノートブック版（`docking_canonical_overrides_buggy.jl`）と修正版（`docking_canonical.jl`）で比較 | ~15 秒 | [`sanity_report.md`](sanity_report.md) |
| `gradcheck.jl` | `rrule(docking_score_elec)` の勾配と中央差分の整合確認。α, β, iface 3 点, charge 3 点を比較 | ~45 秒 | [`gradcheck_report.md`](gradcheck_report.md) |

## 前提

- Julia 1.11.6、`MDToolbox.jl` 0.1.2、`Flux.jl` 0.16.5、`ChainRulesCore`、`CUDA`、`ProgressMeter`、`JLD2` がインストール済みであること
- `protein/1KXQ/complex.*.pdb` が存在すること（ノートブック側でコンパイルされる Decoy ファイル）
- `docking_canonical_overrides_buggy.jl` は `python3 tools/extract_notebook.py > docking_canonical_overrides_buggy.jl` で再生成可能

## 全部まとめて実行

```bash
cd /Users/yasu/gdrive/work/differentiable/docking
bash tests/julia_ref/run_all.sh
```

## スクリプトを追加する規約

- 新しいテストは `tests/julia_ref/` 配下に `test_<何>.jl` または `<何>_check.jl` として追加する
- ハッピーパス／エッジケース共に数値結果を markdown レポートにまとめる
- 参照出力を JLD2 にダンプする場合は `tests/refs/<proteinID>/phaseN_<何>.jld2` のパスで統一
