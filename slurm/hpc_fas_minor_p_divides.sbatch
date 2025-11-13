#!/usr/bin/env bash
#SBATCH --job-name=fas_p_divides
#SBATCH --partition=cpu              # adjust to your cluster
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

# HPC Slurm batch script to run hpc_fas_minor.py and compute
# the coefficient (in the divisible sense) of the base-A root product p.
#
# Usage example:
#   sbatch --export=ALL,\
#     TUPLES="3,1,5;3,1,4",\
#     ROW="0,0,1",\
#     OUT_PREFIX="runs/run1",\
#     EXPAND=0 \
#     slurm/hpc_fas_minor_p_divides.sbatch
#
# Environment variables (required unless defaulted):
#   TUPLES     Semicolon-separated characteristic tuples, e.g. "3,1,5;3,1,4"
#   ROW        Extra row spec "g,v,s", e.g. "0,0,1"
#   OUT_PREFIX Output file prefix, e.g. "runs/run1" (directories auto-created)
#   EXPAND     Optional (0/1). If 1, expand det(A_base)*S before Poly; may be slower.
#
# Notes:
# - Adjust partition/account/time/mem to your cluster.
# - Ensure your Python environment is available on compute nodes.

set -euo pipefail

# ------------------------- Environment / Modules -------------------------
module purge || true
module load intel-python/2022.1.0

# Activate your venv in $SCRATCH/fascalc (adjust if different)
cd "$SCRATCH/fascalc"
source venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

# ------------------------------ Parameters ------------------------------
# Optional: auto-load variables from a params.env file if present
# Example params.env contents:
#   TUPLES='3,1,5;3,1,4'
#   ROW='0,0,1'
#   OUT_PREFIX='runs/run1'
#   EXPAND=0
if [ -f "${SLURM_SUBMIT_DIR}/params.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${SLURM_SUBMIT_DIR}/params.env"
  set +a
fi

: "${TUPLES:?Set TUPLES like '3,1,5;3,1,4'}"
: "${ROW:?Set ROW like '0,0,1' (graph_idx,vertex,layer)}"
: "${OUT_PREFIX:?Set OUT_PREFIX like 'runs/run1'}"
EXPAND=${EXPAND:-0}

# Preflight validation to catch unquoted --export or bad formats early
if ! printf '%s' "$ROW" | grep -Eq '^[0-9]+,[0-9]+,[0-9]+$'; then
  echo "[ERROR] ROW must be 'g,v,s' (three integers). Got: '$ROW'" >&2
  exit 2
fi

# TUPLES: one or more tuples of 2+ integers separated by commas, tuples sep by semicolons
# e.g. 3,1,5;3,1,4 or 2,1,4
if ! printf '%s' "$TUPLES" | grep -Eq '^[0-9]+(,[0-9]+)+(;[0-9]+(,[0-9]+)+)*$'; then
  echo "[ERROR] TUPLES must look like '3,1,5;3,1,4' (integers separated by commas, tuples by semicolons). Got: '$TUPLES'" >&2
  echo "[HINT] If passing via --export, quote values: --export=ALL,TUPLES='3,1,5;3,1,4',ROW='0,0,1'" >&2
  exit 2
fi

# Normalize EXPAND to 0/1
case "${EXPAND}" in
  1|true|TRUE|yes|YES|on|ON) EXPAND=1 ;;
  0|false|FALSE|no|NO|off|OFF|'') EXPAND=0 ;;
  *) echo "[WARN] Unrecognized EXPAND='$EXPAND'; defaulting to 0" >&2; EXPAND=0 ;;
esac

mkdir -p "$(dirname -- "${OUT_PREFIX}")" logs

# ------------------------------- Run job --------------------------------
echo "[INFO] Starting hpc_fas_minor p-divides at $(date)"
echo "[INFO] Tuples:      ${TUPLES}"
echo "[INFO] Row:          ${ROW}"
echo "[INFO] Out prefix:   ${OUT_PREFIX}"
echo "[INFO] Expand flag:  ${EXPAND}"

set -x
srun python hpc_fas_minor.py \
  --tuples "${TUPLES}" \
  --row "${ROW}" \
  --out-prefix "${OUT_PREFIX}" \
  --coeff-p-divides \
  $( ((EXPAND)) && echo --expand )
set +x

# ------------------------------- Summary --------------------------------
echo "[INFO] Outputs written with prefix: ${OUT_PREFIX}"
echo "[INFO]   - ${OUT_PREFIX}.coeff_p_divides.txt"
echo "[INFO]   - ${OUT_PREFIX}.coeff_p_divides.srepr"
echo "[INFO]   - ${OUT_PREFIX}.coeff_p_divides.meta.json"
echo "[INFO]   - ${OUT_PREFIX}.minor.txt (plus .srepr/.meta.json, and .tex if enabled)"
echo "[INFO] Finished at $(date)"
