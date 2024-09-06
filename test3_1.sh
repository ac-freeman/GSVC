#!/bin/bash

#SBATCH --job-name=videogs_loss_job    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications

# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

# Define datasets and their corresponding names
datasets=(
  "/home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv Jockey"
)

for dataset in "${datasets[@]}"; do
  dataset_path=$(echo $dataset | cut -d' ' -f1)
  data_name=$(echo $dataset | cut -d' ' -f2)
  for num_points in 5000 10000 15000 20000 25000; do
  # Run the training script for each dataset
    for iterations in 30000; do
      srun python train_video.py --dataset $dataset_path --data_name $data_name --num_points $num_points --iterations $iterations
      done
    done
done