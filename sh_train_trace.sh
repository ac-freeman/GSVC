#!/bin/bash

#SBATCH --job-name=T_rep    # Job name
#SBATCH --output=T_rep_output.txt # Standard output and error log
#SBATCH --error=T_rep_error.txt  # Error log
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

# Define datasets and their corresponding names
# datasets=(
#   "/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv Beauty"
# )


# srun python train_video_Full_trace.py --loss_type L2 --dataset ./grey.yuv --data_name grey --num_points 10000 --iterations 100000 --savdir result_trace --savdir_m result_trace --is_ad --is_rm
# srun python train_video_Full_trace.py --loss_type L2 --dataset ./grid.yuv --data_name grid --num_points 10000 --iterations 100000 --savdir result_trace --savdir_m result_trace --is_ad --is_rm
# srun python train_video_Full_trace.py --loss_type L2 --dataset ./greygrid.yuv --data_name greygrid --num_points 10000 --iterations 100000 --savdir result_trace --savdir_m result_trace --is_ad --is_rm
srun python train_video_Full_trace.py --loss_type L2 --dataset ./greygrid.yuv --data_name greygrid --num_points 50000 --iterations 100000 --savdir result_trace --savdir_m result_trace --is_ad --is_rm


