#!/bin/bash

#SBATCH --job-name=T_com_D_4   # Job name
#SBATCH --output=T_com_output.txt # Standard output and error log
#SBATCH --error=T_com_error.txt  # Error log
#SBATCH --time=72:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

python train_Compress_delta3.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_10000/gmodels_state_dict.pth --data_name Beauty --num_points 10000 --savdir Compress_delta_3 --savdir_m Compress_delta_3_modles --iterations 50000 --is_rm
python train_Compress_delta3.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_20000/gmodels_state_dict.pth --data_name Beauty --num_points 20000 --savdir Compress_delta_3 --savdir_m Compress_delta_3_modles --iterations 50000 --is_rm
python train_Compress_delta3.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_30000/gmodels_state_dict.pth --data_name Beauty --num_points 30000 --savdir Compress_delta_3 --savdir_m Compress_delta_3_modles --iterations 50000 --is_rm
python train_Compress_delta3.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_40000/gmodels_state_dict.pth --data_name Beauty --num_points 40000 --savdir Compress_delta_3 --savdir_m Compress_delta_3_modles --iterations 50000 --is_rm
python train_Compress_delta3.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_50000/gmodels_state_dict.pth --data_name Beauty --num_points 50000 --savdir Compress_delta_3 --savdir_m Compress_delta_3_modles --iterations 50000 --is_rm



# python train_Compress_delta.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_10000/gmodels_state_dict.pth --data_name Beauty --num_points 10000 --savdir Compress_delta_1 --savdir_m Compress_delta_1_modles --iterations 50000 --is_rm
# python train_Compress_delta.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_20000/gmodels_state_dict.pth --data_name Beauty --num_points 20000 --savdir Compress_delta_1 --savdir_m Compress_delta_1_modles --iterations 50000 --is_rm
# python train_Compress_delta.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_30000/gmodels_state_dict.pth --data_name Beauty --num_points 30000 --savdir Compress_delta_1 --savdir_m Compress_delta_1_modles --iterations 50000 --is_rm
# python train_Compress_delta.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_40000/gmodels_state_dict.pth --data_name Beauty --num_points 40000 --savdir Compress_delta_1 --savdir_m Compress_delta_1_modles --iterations 50000 --is_rm
# python train_Compress_delta.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_50000/gmodels_state_dict.pth --data_name Beauty --num_points 50000 --savdir Compress_delta_1 --savdir_m Compress_delta_1_modles --iterations 50000 --is_rm

# python train_Compress_delta.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_none/Beauty/GaussianVideo_100000_9000/gmodels_state_dict.pth --data_name Beauty --num_points 9000 --savdir Compress_delta_none --savdir_m Compress_delta_none_modles --iterations 50000 
# python train_Compress_delta.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_none/Beauty/GaussianVideo_100000_18000/gmodels_state_dict.pth --data_name Beauty --num_points 18000 --savdir Compress_delta_none --savdir_m Compress_delta_none_modles --iterations 50000 
# python train_Compress_delta.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_none/Beauty/GaussianVideo_100000_27000/gmodels_state_dict.pth --data_name Beauty --num_points 27000 --savdir Compress_delta_none --savdir_m Compress_delta_none_modles --iterations 50000 
# python train_Compress_delta.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_none/Beauty/GaussianVideo_100000_36000/gmodels_state_dict.pth --data_name Beauty --num_points 36000 --savdir Compress_delta_none --savdir_m Compress_delta_none_modles --iterations 50000 
# python train_Compress_delta.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_none/Beauty/GaussianVideo_100000_45000/gmodels_state_dict.pth --data_name Beauty --num_points 45000 --savdir Compress_delta_none --savdir_m Compress_delta_none_modles --iterations 50000 


# python train_Compress.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_10000/gmodels_state_dict.pth --data_name Beauty --num_points 10000 --savdir Compress_test --savdir_m Compress_modles_test --iterations 50000 --is_rm