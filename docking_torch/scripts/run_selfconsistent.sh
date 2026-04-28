#!/usr/bin/env bash
# Self-consistent decoy + training loop: each round regenerates the
# FFT decoys under the previous round's trained params, then trains
# a fresh scorer on those decoys. The trained params of round N seed
# the decoy generation for round N+1.
#
# Resumable: if `params_rXX.pt` already exists for a given round, that
# round is skipped (allows restart after interruption).
#
# All env vars have sensible defaults; override by exporting before
# invoking. Key knobs:
#
#   N_ROUNDS=10           number of self-consistent iterations
#   N_PROTEINS=40         subset of proteins used for training each round
#                         (decoys are still built for all 129)
#   N_ROT=2048            FFT rotation grid size per decoy build
#   NTOP=1000             top-N decoys retained per protein
#   EPOCHS=25             training epochs per round
#   LR_GRID=0.01          comma-separated lr grid (single value = no lr search)
#   LOSS=dockq_rank       training objective
#   GPUS=1,2,3,4          GPUs used for parallel decoy generation
#   TRAIN_GPU=1           single GPU used for training
#   OUT_DIR=out/selfconsistent_<LOSS>

set -euo pipefail

N_ROUNDS=${N_ROUNDS:-10}
N_PROTEINS=${N_PROTEINS:-40}
# Stratified decoy filter parameters (default recommended values)
FILTER_MODE=${FILTER_MODE:-stratified}
N_ANCHOR=${N_ANCHOR:-200}
CONE_DEG=${CONE_DEG:-12}
N_HARD=${N_HARD:-1000}
N_CONTROL=${N_CONTROL:-200}
N_HARD_ROT=${N_HARD_ROT:-}   # rotation pool size (empty = default)
# For top-K fallback mode only:
N_ROT=${N_ROT:-2048}
NTOP=${NTOP:-1400}
EPOCHS=${EPOCHS:-25}
LR_GRID=${LR_GRID:-0.01}
LOSS=${LOSS:-dockq_rank}
DOCKQ_TEMPERATURE=${DOCKQ_TEMPERATURE:-0.2}
GPUS=${GPUS:-1,2,3,4}
TRAIN_GPU=${TRAIN_GPU:-1}

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT_DIR=${OUT_DIR:-$REPO_ROOT/out/selfconsistent_${LOSS}}
mkdir -p "$OUT_DIR"

MASTER_LOG="$OUT_DIR/master.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

cd "$REPO_ROOT"

log "===== self-consistent run: ${N_ROUNDS} rounds, loss=${LOSS} ====="
log "N_PROTEINS=${N_PROTEINS}  EPOCHS=${EPOCHS}  LR_GRID=${LR_GRID}"
log "FILTER_MODE=${FILTER_MODE}"
if [ "$FILTER_MODE" = "stratified" ]; then
    log "  N_ANCHOR=${N_ANCHOR}  CONE_DEG=${CONE_DEG}  N_HARD=${N_HARD}  N_CONTROL=${N_CONTROL}"
else
    log "  N_ROT=${N_ROT}  NTOP=${NTOP}"
fi
log "GPUS=${GPUS}  TRAIN_GPU=${TRAIN_GPU}"
log "OUT_DIR=${OUT_DIR}"

PREV_CKPT=""

for i in $(seq 1 "$N_ROUNDS"); do
    R=$(printf "r%02d" "$i")
    DECOYS="$OUT_DIR/decoys_${R}.h5"
    PARAMS="$OUT_DIR/params_${R}.pt"
    ROUND_LOG="$OUT_DIR/round_${R}.log"

    if [ -f "$PARAMS" ]; then
        log "--- Round $i: $PARAMS exists, skip"
        PREV_CKPT="$PARAMS"
        continue
    fi

    log "--- Round $i start"

    # Decoy generation (skip if the decoy h5 already exists, e.g. from
    # an earlier interrupted run that crashed during training).
    if [ ! -f "$DECOYS" ]; then
        log "    building decoys → $DECOYS (filter=$FILTER_MODE)"
        if [ -n "$PREV_CKPT" ]; then
            CKPT_FLAG="--params-ckpt $PREV_CKPT"
        else
            CKPT_FLAG=""
        fi
        if [ "$FILTER_MODE" = "stratified" ]; then
            N_HARD_ROT_FLAG=""
            if [ -n "$N_HARD_ROT" ]; then
                N_HARD_ROT_FLAG="--n-hard-rot $N_HARD_ROT"
            fi
            uv run python scripts/build_fft_decoys.py \
                --gpus "$GPUS" \
                --filter-mode stratified \
                --n-anchor "$N_ANCHOR" --cone-deg "$CONE_DEG" \
                --n-hard "$N_HARD" --n-control "$N_CONTROL" \
                $N_HARD_ROT_FLAG \
                --out "$DECOYS" $CKPT_FLAG \
                >>"$ROUND_LOG" 2>&1
        else
            uv run python scripts/build_fft_decoys.py \
                --gpus "$GPUS" \
                --filter-mode top_k \
                --n-rotations "$N_ROT" --ntop "$NTOP" \
                --out "$DECOYS" $CKPT_FLAG \
                >>"$ROUND_LOG" 2>&1
        fi
    else
        log "    decoys exist → $DECOYS (skip build)"
    fi

    SEED=42
    log "    training → $PARAMS (seed=$SEED, T=$DOCKQ_TEMPERATURE, epochs=$EPOCHS)"
    CUDA_VISIBLE_DEVICES=$TRAIN_GPU uv run python \
        examples/06_train_dockq_fft.py \
        --decoys "$DECOYS" \
        --n-proteins "$N_PROTEINS" \
        --epochs "$EPOCHS" \
        --lr-grid "$LR_GRID" \
        --loss "$LOSS" \
        --dockq-temperature "$DOCKQ_TEMPERATURE" \
        --device cuda \
        --seed "$SEED" \
        --out "$PARAMS" \
        >>"$ROUND_LOG" 2>&1

    PREV_CKPT="$PARAMS"

    # Quick progress peek: last val metric reported in the round log
    VAL_METRIC=$(grep -Eo 'val mean DockQ@top-[0-9]+ = [0-9.]+' "$ROUND_LOG" | tail -1 || true)
    log "--- Round $i done: $VAL_METRIC"
done

log "===== all ${N_ROUNDS} rounds complete ====="
log "checkpoints: $OUT_DIR/params_r*.pt"
log "master log: $MASTER_LOG"
