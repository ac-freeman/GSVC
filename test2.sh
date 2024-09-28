#!/bin/bash

#SBATCH --job-name=videogs_loss_job    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=8:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=24G                       # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

# Define datasets and their corresponding names
# datasets=(
#   "/home/e/e1344641/data/UVG/A1/A1_1920x1080_120fps_420_8bit_YUV.yuv A1"
#   "/home/e/e1344641/data/UVG/A2/A2_1920x1080_120fps_420_8bit_YUV.yuv A2"
#   "/home/e/e1344641/data/UVG/A3/A3_1920x1080_120fps_420_8bit_YUV.yuv A3"
# )
# datasets=(
#   "/home/e/e1344641/data/UVG/A1/A1_1920x1080_120fps_420_8bit_YUV.yuv A1"
#   "/home/e/e1344641/data/UVG/A2/A2_1920x1080_120fps_420_8bit_YUV.yuv A2"
#   "/home/e/e1344641/data/UVG/A3/A3_1920x1080_120fps_420_8bit_YUV.yuv A3"
#   "/home/e/e1344641/data/UVG/B1/B1_1920x1080_120fps_420_8bit_YUV.yuv B1"
#   "/home/e/e1344641/data/UVG/B2/B2_1920x1080_120fps_420_8bit_YUV.yuv B2"
#   "/home/e/e1344641/data/UVG/B3/B3_1920x1080_120fps_420_8bit_YUV.yuv B3"
#   "/home/e/e1344641/data/UVG/B1/B1_1920x1080_120fps_420_8bit_YUV.yuv C1"
#   "/home/e/e1344641/data/UVG/B2/B2_1920x1080_120fps_420_8bit_YUV.yuv C2"
#   "/home/e/e1344641/data/UVG/B3/B3_1920x1080_120fps_420_8bit_YUV.yuv C3"
# )
datasets=(
  "/home/e/e1344641/data/UVG/C3/C3_1920x1080_120fps_420_8bit_YUV.yuv C3"
  "/home/e/e1344641/data/UVG/D3/D3_1920x1080_120fps_420_8bit_YUV.yuv D3"
)

# Define additional parameters
savdir="result_density_C/grad"
savdir_m="models_density_C/grad"
savdir_f="result_density_C/f"
savdir_m_f="models_density_C/f"
is_pos=True
is_warmup=False
is_ad=True
loss_type="L2"
is_clip=False
for dataset in "${datasets[@]}"; do
  dataset_path=$(echo $dataset | cut -d' ' -f1)
  data_name=$(echo $dataset | cut -d' ' -f2)
  for num_points in 50000; do
    for iterations in 30000; do
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
      srun python train_video_grad.py --loss_type $loss_type --dataset $dataset_path \
        --data_name $data_name --num_points $num_points --iterations $iterations \
        --savdir $savdir --savdir_m $savdir_m \
        $pos_flag $warmup_flag $ad_flag $clip_flag
      srun python train_video_frame.py --loss_type $loss_type --dataset $dataset_path \
        --data_name $data_name --num_points $num_points --iterations $iterations \
        --savdir $savdir_f --savdir_m $savdir_m_f \
        $pos_flag $warmup_flag $ad_flag $clip_flag
    done
  done
done
