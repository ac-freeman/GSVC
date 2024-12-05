#!/bin/bash

#SBATCH --job-name=test_3d_1    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=72:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

python train_video_3dGS.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv  --data_name Jockey --num_points 10000 --savdir 3dGS --savdir_m m_3dGS --iterations 50000
python train_video_3dGS.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv  --data_name Jockey --num_points 20000 --savdir 3dGS --savdir_m m_3dGS --iterations 50000
python train_video_3dGS.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv  --data_name Jockey --num_points 30000 --savdir 3dGS --savdir_m m_3dGS --iterations 50000
python train_video_3dGS.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv  --data_name Jockey --num_points 40000 --savdir 3dGS --savdir_m m_3dGS --iterations 50000
python train_video_3dGS.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv  --data_name Jockey --num_points 50000 --savdir 3dGS --savdir_m m_3dGS --iterations 50000
