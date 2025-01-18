#!/bin/bash

#SBATCH --job-name=T_com   # Job name
#SBATCH --output=./output/rep_output.txt # Standard output and error log
#SBATCH --error=./output/rep_error.txt  # Error log
#SBATCH --time=72:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

# srun python train_Compress.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_10000/gmodels_state_dict.pth --data_name Beauty --num_points 10000 --savdir Compress1 --savdir_m Compress_modles1 --iterations 50000 --is_rm
# srun python train_video_Represent.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 10000 --iterations 10 --savdir test4 --savdir_m model4 --is_ad --is_rm

# srun python train_video_Represent_trace.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 10 --iterations 20000 --savdir result_iter --savdir_m model
# srun python train_video_Represent_trace.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 100 --iterations 20000 --savdir result_iter --savdir_m model 
# srun python train_video_Represent_trace.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 1000 --iterations 20000 --savdir result_iter --savdir_m model 
# srun python train_video_Represent_trace.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 10000 --iterations 20000 --savdir result_iter --savdir_m model 
# srun python train_video_Represent_trace.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 100000 --iterations 20000 --savdir result_iter --savdir_m model 

# srun python train_video_Represent_trace2.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 10 --iterations 20000 --savdir result_iter_2 --savdir_m model
# srun python train_video_Represent_trace2.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 100 --iterations 20000 --savdir result_iter_2 --savdir_m model 
# srun python train_video_Represent_trace2.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 1000 --iterations 20000 --savdir result_iter_2 --savdir_m model 
# srun python train_video_Represent_trace2.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 10000 --iterations 20000 --savdir result_iter_2 --savdir_m model 
# srun python train_video_Represent_trace2.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 1 --iterations 20000 --savdir result_iter_2 --savdir_m model 

# srun python train_video_Represent_trace2.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 1 --iterations 20000 --savdir result_iter_3 --savdir_m model 
# srun python train_video_Represent_trace2.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 10000 --iterations 20000 --savdir result_iter --savdir_m model 

srun python train_video_Represent_test.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 100000 --iterations 100000 --savdir result_high_2 --savdir_m model_high_2 --is_ad --is_rm
srun python train_video_Compress.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints/model_high_2/Beauty/GaussianVideo_100000_100000/gmodels_state_dict.pth --data_name Beauty --num_points 100000 --savdir Compress_high --savdir_m Compress_modles_high --iterations 100000 --is_rm
