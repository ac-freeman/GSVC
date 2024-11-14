#!/bin/bash

#SBATCH --job-name=test    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

# python train_video_DD_test.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 10000 --iterations 100000 --savdir result_dd --savdir_m models_dd --is_ad --is_rm
# python train_video_DD_test.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --data_name HoneyBee --num_points 10000 --iterations 100000 --savdir result_dd --savdir_m models_dd --is_ad --is_rm
python train_video_DD_test.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 10000 --iterations 100000 --savdir result_dd --savdir_m models_dd --is_ad --is_rm
#python train_video_DD_test.py --loss_type L2 --dataset /home/e/e1344641/GaussianVideo/Video/Mix_1920x1080_120fps_420_8bit_YUV.yuv --data_name Mix --num_points 10000 --iterations 100000 --savdir result_dd --savdir_m models_dd --is_ad --is_rm
