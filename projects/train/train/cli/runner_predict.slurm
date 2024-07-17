#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=amplfisim
#SBATCH --account=bcse-delta-gpu
#SBATCH --output=/scratch/bcse/saggarwal/%x.out
#SBATCH --error=/scratch/bcse/saggarwal/%x.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=aggar104@umn.edu

PROJECT_DIR="/projects/bcse/"
export CONTAINER_PATH="/scratch/bcse/saggarwal/amplfi/delta.sif"

export AMPLFI_OUTDIR="/scratch/bcse/saggarwal/amplfi/out_dir"
export AMPLFI_DATADIR="/projects/bcse/saggarwal/amplfi-data-dir"

export BASE_DIR="$/scratch/bcse/saggarwal/pe-base-dir/"
export DATA_DIR="/projects/bcse/saggarwal/amplfi-data-dir"
export SCRATCH_DIR="/scratch/bcse/saggarwal/amplfi/"

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Running on `hostname`"
nvidia-smi

srun apptainer run --bind /scratch/bcse/saggarwal/amplfi/out_dir,/projects/bcse/saggarwal/amplfi-data-dir,/projects/bcse/saggarwal/binds --nv /scratch/bcse/saggarwal/amplfi/delta2.sif /usr/local/bin/python /projects/bcse/saggarwal/amplfi/projects/train/train/cli/similarity.py fit --config /projects/bcse/saggarwal/binds/cbc.yaml
