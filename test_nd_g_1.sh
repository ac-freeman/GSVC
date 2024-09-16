#!/bin/bash

#SBATCH --job-name=videogs_loss_job    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                       # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

# Define datasets and their corresponding names
datasets=(
  "/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv Beauty"
  "/home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv HoneyBee"
  "/home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv Jockey"
)

# Define additional parameters
savdir="result_grad"
savdir_m="models_grad"
is_pos=True
is_warmup=False
is_ad=False
for dataset in "${datasets[@]}"; do
  dataset_path=$(echo $dataset | cut -d' ' -f1)
  data_name=$(echo $dataset | cut -d' ' -f2)
  for num_points in 10000; do
    for iterations in 50000; do
      pos_flag=""
      warmup_flag=""
      ad_flag=""

      # 检查布尔值并构建相应的命令行参数
      if [ "$is_pos" = True ]; then
        pos_flag="--is_pos"
      fi

      if [ "$is_warmup" = True ]; then
        warmup_flag="--is_warmup"
      fi

      if [ "$is_ad" = True ]; then
        ad_flag="--is_ad"
      fi
      # Run the training script for each dataset with additional parameters
      srun python train_video_grad.py --dataset $dataset_path \
        --data_name $data_name --num_points $num_points --iterations $iterations \
        --savdir $savdir --savdir_m $savdir_m \
        $pos_flag $warmup_flag $ad_flag
    done
  done
done
