#!/bin/bash

#SBATCH --job-name=test    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=01:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

srun python Interpolation.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 27000 --savdir itest_kl --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_kl/Jockey/GaussianImage_Cholesky_100000_30000/gmodels_state_dict.pth
srun python Interpolation.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 27000 --savdir itest_kl --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_kl/Beauty/GaussianImage_Cholesky_100000_30000/gmodels_state_dict.pth

srun python loadmodel.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 27000 --savdir test_kl --model_path /home/e/e1344641/GaussianVideo/modcheckpointsels/models_kl/Jockey/GaussianImage_Cholesky_100000_30000/gmodels_state_dict.pth
srun python loadmodel.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 27000 --savdir test_kl --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_kl/Beauty/GaussianImage_Cholesky_100000_30000/gmodels_state_dict.pth

srun python Interpolation.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 72000 --savdir itest_kl --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_kl/Jockey/GaussianImage_Cholesky_100000_80000/gmodels_state_dict.pth
srun python Interpolation.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 72000 --savdir itest_kl --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_kl/Beauty/GaussianImage_Cholesky_100000_80000/gmodels_state_dict.pth

srun python loadmodel.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 72000 --savdir test_kl --model_path /home/e/e1344641/GaussianVideo/modcheckpointsels/models_kl/Jockey/GaussianImage_Cholesky_100000_80000/gmodels_state_dict.pth
srun python loadmodel.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 72000 --savdir test_kl --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_kl/Beauty/GaussianImage_Cholesky_100000_80000/gmodels_state_dict.pth

