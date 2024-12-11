#!/bin/bash

#SBATCH --job-name=test_com_3   # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=72:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment
datasets=(
  "/home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv Jockey"
)

savdir="GaussianImage"
savdir_m="models_GaussianImage"
is_pos=False
is_ad=False
is_rm=False
loss_type="L2"
for dataset in "${datasets[@]}"; do
  dataset_path=$(echo $dataset | cut -d' ' -f1)
  data_name=$(echo $dataset | cut -d' ' -f2)
  # for num_points in  90000  18000 27000 36000 45000; do
  for num_points in  9000; do
    for iterations in 100000; do
      pos_flag=""
      ad_flag=""

      if [ "$is_pos" = True ]; then
        pos_flag="--is_pos"
      fi


      if [ "$is_ad" = True ]; then
        ad_flag="--is_ad"
      fi

      if [ "$is_rm" = True ]; then
        rm_flag="--is_rm"
      fi

      srun python train_video_Full_test.py --loss_type $loss_type --dataset $dataset_path \
        --data_name $data_name --num_points $num_points --iterations $iterations \
        --savdir $savdir --savdir_m $savdir_m \
        $pos_flag $ad_flag $rm_flag
    done
  done
done

python Compress_train.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_GaussianImage/Jockey/GaussianVideo_100000_9000/gmodels_state_dict.pth --data_name Jockey --num_points 10000 --savdir Compress_GI --savdir_m Compress_GI --iterations 50000
# python Compress_train.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_GaussianImage/Jockey/GaussianVideo_100000_18000/gmodels_state_dict.pth --data_name Jockey --num_points 20000 --savdir Compress_GI --savdir_m Compress_GI --iterations 50000
# python Compress_train.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_GaussianImage/Jockey/GaussianVideo_100000_27000/gmodels_state_dict.pth --data_name Jockey --num_points 30000 --savdir Compress_GI --savdir_m Compress_GI --iterations 50000
# python Compress_train.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_GaussianImage/Jockey/GaussianVideo_100000_36000/gmodels_state_dict.pth --data_name Jockey --num_points 40000 --savdir Compress_GI --savdir_m Compress_GI --iterations 50000
# python Compress_train.py --dataset /home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv --model_path /home/e/e1344641/GaussianVideo/checkpoints/models_GaussianImage/Jockey/GaussianVideo_100000_45000/gmodels_state_dict.pth --data_name Jockey --num_points 50000 --savdir Compress_GI --savdir_m Compress_GI --iterations 50000