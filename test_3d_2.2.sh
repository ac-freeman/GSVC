#!/bin/bash

#SBATCH --job-name=test_3d_2    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment


python train_video_3dGS.py --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv  --data_name HoneyBee --num_points 18000 --savdir 3dGS2 --savdir_m m_3dGS2 --iterations 50000
python train_video_3dGS.py --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv  --data_name HoneyBee --num_points 27000 --savdir 3dGS2 --savdir_m m_3dGS2 --iterations 50000
python train_video_3dGS.py --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv  --data_name HoneyBee --num_points 36000 --savdir 3dGS2 --savdir_m m_3dGS2 --iterations 50000
python train_video_3dGS.py --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv  --data_name HoneyBee --num_points 45000 --savdir 3dGS2 --savdir_m m_3dGS2 --iterations 50000
