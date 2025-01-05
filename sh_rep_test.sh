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

srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/Beauty/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name Beauty --num_points 10000 --savdir d_video --is_rm
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/Beauty/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name Beauty --num_points 20000 --savdir d_video --is_rm 
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/Beauty/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name Beauty --num_points 30000 --savdir d_video --is_rm 
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/Beauty/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name Beauty --num_points 40000 --savdir d_video --is_rm 
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/Beauty/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name Beauty --num_points 50000 --savdir d_video --is_rm  


srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/HoneyBee/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name HoneyBee --num_points 10000 --savdir d_video --is_rm 
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/HoneyBee/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name HoneyBee --num_points 20000 --savdir d_video --is_rm 
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/HoneyBee/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name HoneyBee --num_points 30000 --savdir d_video --is_rm 
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/HoneyBee/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name HoneyBee --num_points 40000 --savdir d_video --is_rm 
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/HoneyBee/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name HoneyBee --num_points 50000 --savdir d_video --is_rm 

srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/Jockey/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name Jockey --num_points 10000 --savdir d_video --is_rm 
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/Jockey/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name Jockey --num_points 20000 --savdir d_video --is_rm 
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/Jockey/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name Jockey --num_points 30000 --savdir d_video --is_rm 
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/Jockey/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name Jockey --num_points 40000 --savdir d_video --is_rm 
srun python test_rasterizer.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress_delta_4_modles/Jockey/GaussianVideo_50000_10000/gmodels_state_dict.pth --data_name Jockey --num_points 50000 --savdir d_video --is_rm 