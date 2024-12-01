#!/bin/bash

#SBATCH --job-name=test_dd_1    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment


data_path=/home/e/e1344641/data/kodak

for num_points in  9000
do
CUDA_VISIBLE_DEVICES=0 python train_quantize.py -d $data_path \
--data_name kodak --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000 \
# --model_path ./checkpoints/kodak/GaussianImage_Cholesky_50000_$num_points
done

for num_points in 9000
do
CUDA_VISIBLE_DEVICES=0 python test_quantize.py -d $data_path \
--data_name kodak --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000 \
--model_path ./checkpoints_quant/kodak/GaussianImage_Cholesky_50000_$num_points
done



