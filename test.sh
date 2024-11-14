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

python train_video_densification.py --loss_type L2 --dataset /home/e/e1344641/GaussianVideo/exp/GMM_exp/test_video.yuv --data_name test1 --num_points 10000 --iterations 25000 --savdir test1 --savdir_m test1 --is_rm --save_everyimgs --is_pos
python train_video_densification.py --loss_type L2 --dataset /home/e/e1344641/GaussianVideo/exp/GMM_exp/test_video.yuv --data_name test2 --num_points 10000 --iterations 25000 --savdir test1 --savdir_m test1 --is_ad --is_rm --save_everyimgs --is_pos
python train_video_densification.py --loss_type L2 --dataset /home/e/e1344641/GaussianVideo/exp/GMM_exp2/output_video1.yuv --data_name test3 --num_points 10000 --iterations 25000 --savdir test1 --savdir_m test1 --is_ad --is_rm --save_everyimgs --is_pos
python train_video_DG.py --loss_type L2 --dataset /home/e/e1344641/GaussianVideo/exp/GMM_exp2/output_video1.yuv --data_name test4 --num_points 10000 --iterations 25000 --savdir test1 --savdir_m test1 --is_ad --is_rm --save_everyimgs --is_pos

python train_video_DG.py --loss_type L2 --dataset /home/e/e1344641/GaussianVideo/Video/Mix_1920x1080_120fps_420_8bit_YUV.yuv --data_name Mix --num_points 10000 --iterations 100000 --savdir result_DG --savdir_m models_DG --is_ad --is_rm
