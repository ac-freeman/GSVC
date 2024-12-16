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

srun python train_test.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --data_name HoneyBee --num_points 50000 --iterations 100000 --savdir test_result --savdir_m test_models --is_ad --is_rm --is_pos --save_everyimgs
srun python train_test.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv --data_name HoneyBee --num_points 45000 --iterations 100000 --savdir testGI_result --savdir_m testGI_models --is_pos --save_everyimgs

srun python train_test.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 50000 --iterations 100000 --savdir test_result --savdir_m test_models --is_ad --is_rm --is_pos --save_everyimgs
srun python train_test.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 45000 --iterations 100000 --savdir testGI_result --savdir_m testGI_models --is_pos --save_everyimgs

srun python train_test.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 50000 --iterations 100000 --savdir test_result --savdir_m test_models --is_ad --is_rm --is_pos --save_everyimgs
srun python train_test.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --data_name Jockey --num_points 45000 --iterations 100000 --savdir testGI_result --savdir_m testGI_models --is_pos --save_everyimgs