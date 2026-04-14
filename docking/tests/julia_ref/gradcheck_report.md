# Gradient Check レポート (A-4)

`rrule(docking_score_elec)`（ノートブック cell 6 の手書き逆伝搬）と中央差分（forward から直接計算）の整合性を 1KXQ top-5 ポーズで検証した結果。

## サマリ（B5 修正後）

| パラメータ | AD (rrule) | FD (central) | 相対誤差 | 判定 |
|---|---|---|---|---|
| α | -1.817e+05 | -1.817e+05 | 3.3e-13 | ✅ 完全一致 |
| β | +2.904 | +2.904 | 5.6e-11 | ✅ 完全一致 |
| iface[1] (i=1,j=1) | +107.0 | +107.0 | 1.7e-11 | ✅ 完全一致 |
| iface[13] (i=1,j=2) | +820.0 | +336.0 | 1.44 | ❌ 2.4x 過大 |
| iface[72] (i=12,j=6) | +482.0 | +482.0 | 3.7e-12 | ✅ 完全一致 |
| chs[1] (TERM-N) | +23.53 | +1.009 | 22.3 | ❌ 23x 過大 |
| chs[6] (LYS-NZ) | +12.84 | +1.426 | 8.0 | ❌ 9x 過大 |
| chs[11] (N) | 0.0 | 0.0 | - | ✅ 両者零 |

## 見つかった rrule のバグ

### B5. spacing mismatch（修正済み）

forward: `spacing=3.0`、rrule: `spacing=1.5`。別解像度のグリッドで勾配計算していたため、最も基礎的な α と β ですら 8 倍ずれていた。`docking_canonical_overrides.jl` 内の rrule を `spacing=3.0` に統一、ついでに `MDToolbox.generate_grid` 呼び出しを上書き版 `generate_grid` に切り替え、forward/backward が同じコードパスを通るようにした。

### B6. IFACE 行列の対称化取り違え

rrule の iface 勾配ループ（cell 6, L1022〜）は `data1[k, iframe] = tmp` と同時に `data1[k_dual, iframe] = tmp` を書き込む。これは iface_ij を対称行列（iface[i,j] = iface[j,i]）として扱い、パラメータが結合して動くと仮定した場合の勾配。

しかし forward（cell 4）は iface を 144 独立パラメータとして読む（`receptor.mass .= iface_ij[k]`）ため、FD は「iface[13] 単体を動かしたときの変化」を測る。rrule は（i=2,j=1）iteration で data1[13] を上書きするため、結果としてレポートされるのは「(i=2,j=1) の tmp」＝ dual ペアの寄与であり、求めるべき「(i=1,j=2) の tmp」とは一般に異なる。

**修正方針**（未適用）：対角要素 (i==j) 以外では data1[k] にのみ書き、data1[k_dual] には書かない。または、data1[k] += tmp; data1[k_dual] += tmp と加算にし、後段で 2 で割る。どちらが「物理的に正しい」かは iface_ij のパラメータ化方針に依存。

### B7. CHARGE 勾配の系統的過大（原因未特定）

chs[1] で 23 倍、chs[6] で 9 倍、rrule が FD より大きい値を返す。
解析によると rrule の計算式 `data2[l] + data3[l] = 2 × charge_score[l] × I_l` は数学的には forward の微分と一致するはずだが、実測ではずれが発生。L 毎に異なる倍率なので、単純なスケールミスではなく、`receptor.mass` / `ligands.mass` の累積や 共有 state による汚染が疑われるが、特定には更なる調査が要る。

調査の時間対効果を考え、本タスクでは **PyTorch 側で autograd を正として実装し、Julia rrule の勾配値に頼らない**方針に切り替える。

## PyTorch 移植への影響

- **Forward は Julia gold を信頼**：`docking_score_elec` の forward スカラは B1/B3/B5 修正版 Julia と数値一致させる（TDD のメインターゲット）。
- **Backward は PyTorch autograd で自動取得**：PyTorch 側で forward を正しく書けば勾配は自動的に正しい。Julia rrule は gold として使わず、Julia FD との突合を検証手段にする。
- **Julia rrule 自体を修正するかは別タスク**：B6/B7 は Julia 側のコードベースを直すべき問題だが、Stage B の PyTorch 移植を先に完了させた方が、Julia 修正時のリグレッションテストが組める。

## 次のタスク

**B-0**：PyTorch プロジェクト雛形を立て、`generate_refs.jl` で 1KXQ の参照 JLD2 を一括出力する。
