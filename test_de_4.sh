#!/bin/bash

#SBATCH --job-name=test_de_4    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                       # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

# Define datasets and their corresponding names
datasets=(
  "/home/e/e1344641/GaussianVideo/Video/Mix_1920x1080_120fps_420_8bit_YUV.yuv Mix"
)

# Define additional parameters
savdir="result_de"
savdir_m="models_de"
is_pos=False
is_ad=True
is_rm=True
loss_type="L2"
for dataset in "${datasets[@]}"; do
  dataset_path=$(echo $dataset | cut -d' ' -f1)
  data_name=$(echo $dataset | cut -d' ' -f2)
  # for num_points in 30000 40000 50000; do
  # for num_points in 30000 40000 50000 60000 70000 80000; do
  for num_points in  10000  20000 30000 40000 50000 5000 15000 25000 60000 70000 80000; do
    for iterations in 100000; do
      pos_flag=""
      ad_flag=""

      # 检查布尔值并构建相应的命令行参数
      if [ "$is_pos" = True ]; then
        pos_flag="--is_pos"
      fi


      if [ "$is_rm" = True ]; then
        rm_flag="--is_rm"
      fi

      # Run the training script for each dataset with additional parameters
      srun python train_video_densification.py --loss_type $loss_type --dataset $dataset_path \
        --data_name $data_name --num_points $num_points --iterations $iterations \
        --savdir $savdir --savdir_m $savdir_m \
        $pos_flag $ad_flag $rm_flag
    done
  done
done
