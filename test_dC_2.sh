#!/bin/bash

#SBATCH --job-name=test_dC_2    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                       # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

srun python train_video_Combine.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --data_name HoneyBee --num_points 30000 --iterations 100000 --savdir result_density_Com2 --savdir_m models_density_Com2 --is_ad
srun python train_video_Combine.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --data_name HoneyBee --num_points 40000 --iterations 100000 --savdir result_density_Com2 --savdir_m models_density_Com2 --is_ad
srun python train_video_Combine.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --data_name HoneyBee --num_points 50000 --iterations 100000 --savdir result_density_Com2 --savdir_m models_density_Com2 --is_ad

