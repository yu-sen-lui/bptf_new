#!/usr/bin/env bash
###############################################################################
# File: run_work.sbatch
###############################################################################
#SBATCH --job-name=running_data_separately_and_combined         # anything short & descriptive
#SBATCH --partition=general            # = -p general
#SBATCH --gpus=a100:1              # ask for 1 GPU, any model
#SBATCH --cpus-per-task=16             # = --cpus-per-task=16
#SBATCH --mem=200G                     # = --mem 200000   (200 GB)
#SBATCH --time=10:00:00                # 600 minutes = 10 hours
#SBATCH --output=%x_%j.out             # stdout ➜ bptf_and_nb_<jobid>.out
#SBATCH --error=%x_%j.err              # stderr ➜ bptf_and_nb_<jobid>.err
# (optional) e-mail on end / fail
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=luiyusen@uchicago.edu
###############################################################################

set -euo pipefail
echo "Running on $(hostname)   at $(date)"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"

# ── 1. activate the exact Conda / module env you use interactively ──────────
# If you rely on modules, do it here, e.g.
#   module purge
#   module load miniconda cuda/12.3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bptf                         # <-- your env

# grab first positional arg (or fall back to 150)
N_COMBINED=${1:-150}

# limit NumPy / MKL threads to the cores we asked for
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── 2. run the jobs (each wrapped in its own srun) ──────────────────────────
# Using "srun" inside the allocation guarantees proper I/O buffering & GPU bind.
# They run SEQUENTIALLY here; put "&" after a command if you want concurrency.

echo "python running_data_separately_and_combined.py --n_components_combined '${N_COMBINED}'"
srun --exclusive python running_data_separately_and_combined.py --n_components_combined "${N_COMBINED}"

echo "Finished at $(date)"