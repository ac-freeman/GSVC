#!/bin/bash

#SBATCH --job-name=test_dW_3    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=8G                       # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

# Define datasets and their corresponding names
datasets=(
  "/home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv Jockey"
)

# Define additional parameters
savdir="result_density_rgbW"
savdir_m="models_Opacity"
is_pos=False
is_warmup=False
is_ad=True
is_clip=False
loss_type="L2"
for dataset in "${datasets[@]}"; do
  dataset_path=$(echo $dataset | cut -d' ' -f1)
  data_name=$(echo $dataset | cut -d' ' -f2)
  # for num_points in 4000 6000 8000 10000 20000 30000 40000 50000 60000 70000; do
  for num_points in 27000 36000 45000 ; do
    for iterations in 100000; do
      pos_flag=""
      warmup_flag=""
      ad_flag=""
      clip_flag=""

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

      if [ "$is_clip" = True ]; then
        clip_flag="--is_clip"
      fi
      # Run the training script for each dataset with additional parameters
      srun python train_video_rgbW.py --loss_type $loss_type --dataset $dataset_path \
        --data_name $data_name --num_points $num_points --iterations $iterations \
        --savdir $savdir --savdir_m $savdir_m \
        $pos_flag $warmup_flag $ad_flag $clip_flag
    done
  done
done

