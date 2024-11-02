#!/bin/bash

#SBATCH --job-name=test    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=01:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

dataset_path="/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv"
model_path="/home/e/e1344641/GaussianVideo/models/Models/Beauty/GaussianImage_Cholesky_100000_30000/gmodels_state_dict.pth"
data_name="Beauty"
num_points=30000
savdir="test"
srun python loadmodel.py --dataset $dataset_path --data_name $data_name --num_points $num_points --savdir $savdir --model_path $model_path