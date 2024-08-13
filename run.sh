#!/bin/bash

#SBATCH --job-name=videogs_loss_job    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mem=16G                       # Memory required per node
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications

# Load the necessary modules (if any, e.g., Python, CUDA, etc.)
# module load python/3.8.5  # Load Python module, adjust the version accordingly
# module load cuda/11.2     # Load CUDA module, adjust the version accordingly

# Activate the environment if needed
source activate torch     # Replace 'myenv' with the name of your conda environment

# Run the Python script
srun cd gsplat
srun pip install .[dev]
cd ../
pip install -r requirements.txt
