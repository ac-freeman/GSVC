#!/bin/bash

#SBATCH --job-name=test_com_1    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=72:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

python Compress_train.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_10000/gmodels_state_dict.pth --data_name Beauty --num_points 10000 --savdir Compress --savdir_m Compress --iterations 50000
python Compress_test.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress/Beauty/GaussianVideo_100000_10000/gmodels_state_dict.pth --data_name Beauty --num_points 10000 --savdir Compress

python Compress_train.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_20000/gmodels_state_dict.pth --data_name Beauty --num_points 20000 --savdir Compress --savdir_m Compress --iterations 50000
python Compress_test.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress/Beauty/GaussianVideo_100000_20000/gmodels_state_dict.pth --data_name Beauty --num_points 20000 --savdir Compress

python Compress_train.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_30000/gmodels_state_dict.pth --data_name Beauty --num_points 30000 --savdir Compress --savdir_m Compress --iterations 50000
python Compress_test.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress/Beauty/GaussianVideo_100000_30000/gmodels_state_dict.pth --data_name Beauty --num_points 30000 --savdir Compress

python Compress_train.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_40000/gmodels_state_dict.pth --data_name Beauty --num_points 40000 --savdir Compress --savdir_m Compress --iterations 50000
python Compress_test.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress/Beauty/GaussianVideo_100000_40000/gmodels_state_dict.pth --data_name Beauty --num_points 40000 --savdir Compress

python Compress_train.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_50000/gmodels_state_dict.pth --data_name Beauty --num_points 50000 --savdir Compress --savdir_m Compress --iterations 50000
python Compress_test.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints_quant/Compress/Beauty/GaussianVideo_100000_50000/gmodels_state_dict.pth --data_name Beauty --num_points 50000 --savdir Compress