#!/bin/bash

#SBATCH --job-name=T_rep    # Job name
#SBATCH --output=output/T_rep_output.txt # Standard output and error log
#SBATCH --error=output/T_rep_error.txt  # Error log
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source ~/.bashrc
conda activate torch  # Replace 'torch' with the name of your conda environment




srun python train_video_Represent.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --data_name HoneyBee --num_points 50000 --savdir GaussianVideo_results_pos --savdir_m GaussianVideo_models_pos --iterations 50000 --image_length 50 --is_rm --is_ad --is_pos --save_everyimgs
