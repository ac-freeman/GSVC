#!/bin/bash

#SBATCH --job-name=T_com_D_1   # Job name
#SBATCH --output=T_com_output.txt # Standard output and error log
#SBATCH --error=T_com_error.txt  # Error log
#SBATCH --time=72:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

datasets=(
  "/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv Beauty"
)


# Define additional parameters
savdir="result_K10"
savdir_m="models_K10"
is_pos=False
is_ad=True
is_rm=True
loss_type="L2"
for dataset in "${datasets[@]}"; do
  dataset_path=$(echo $dataset | cut -d' ' -f1)
  data_name=$(echo $dataset | cut -d' ' -f2)
  for num_points in  10000  20000 30000 40000 50000; do
    for iterations in 100000; do
      pos_flag=""
      ad_flag=""
      rm_flag=""
      if [ "$is_pos" = True ]; then
        pos_flag="--is_pos"
      fi


      if [ "$is_ad" = True ]; then
        ad_flag="--is_ad"
      fi

      if [ "$is_rm" = True ]; then
        rm_flag="--is_rm"
      fi

      srun python train_video_Full_test2.py --loss_type $loss_type --dataset $dataset_path \
        --data_name $data_name --num_points $num_points --iterations $iterations \
        --savdir $savdir --savdir_m $savdir_m \
        $pos_flag $ad_flag $rm_flag
    done
  done
done


# python train_video_Full_test2.py --loss_type L2 --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 10000 --savdir result_K10 --savdir_m models_K10 --iterations 100000 --is_rm --is_ad

python train_Compress_delta2.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_10000/gmodels_state_dict.pth --data_name Beauty --K_frame_path /home/e/e1344641/GaussianVideo/checkpoints/result_K10/Beauty/K_frames.txt --num_points 10000 --savdir Compress_delta_2 --savdir_m Compress_delta_2_modles --iterations 50000 --is_rm
python train_Compress_delta2.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_20000/gmodels_state_dict.pth --data_name Beauty --K_frame_path /home/e/e1344641/GaussianVideo/checkpoints/result_K10/Beauty/K_frames.txt --num_points 20000 --savdir Compress_delta_2 --savdir_m Compress_delta_2_modles --iterations 50000 --is_rm
python train_Compress_delta2.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_30000/gmodels_state_dict.pth --data_name Beauty --K_frame_path /home/e/e1344641/GaussianVideo/checkpoints/result_K10/Beauty/K_frames.txt --num_points 30000 --savdir Compress_delta_2 --savdir_m Compress_delta_2_modles --iterations 50000 --is_rm
python train_Compress_delta2.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_40000/gmodels_state_dict.pth --data_name Beauty --K_frame_path /home/e/e1344641/GaussianVideo/checkpoints/result_K10/Beauty/K_frames.txt --num_points 40000 --savdir Compress_delta_2 --savdir_m Compress_delta_2_modles --iterations 50000 --is_rm
python train_Compress_delta2.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/models/models_dd/Beauty/GaussianImage_Cholesky_100000_50000/gmodels_state_dict.pth --data_name Beauty --K_frame_path /home/e/e1344641/GaussianVideo/checkpoints/result_K10/Beauty/K_frames.txt --num_points 50000 --savdir Compress_delta_2 --savdir_m Compress_delta_2_modles --iterations 50000 --is_rm



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