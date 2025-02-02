#!/bin/bash

#SBATCH --job-name=T_com   # Job name
#SBATCH --output=output/T_com_output.txt # Standard output and error log
#SBATCH --error=output/T_com_error.txt  # Error log
#SBATCH --time=72:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

# datasets=(
#   "/home/e/e1344641/data/USTC/USTC_Badminton.yuv  Badminton"
#   "/home/e/e1344641/data/USTC/USTC_BasketballPass.yuv BasketballPass"
#   "/home/e/e1344641/data/USTC/USTC_Dancing.yuv Dancing"
#   "/home/e/e1344641/data/USTC/USTC_ParkWalking.yuv ParkWalking"
#   "/home/e/e1344641/data/USTC/USTC_ShakingHands.yuv ShakingHands"
#   "/home/e/e1344641/data/USTC/USTC_BasketballDrill.yuv BasketballDrill"
#   "/home/e/e1344641/data/USTC/USTC_BicycleDriving.yuv BicycleDriving"
#   "/home/e/e1344641/data/USTC/USTC_FourPeople.yuv FourPeople"
#   "/home/e/e1344641/data/USTC/USTC_Running.yuv Running"
#   "/home/e/e1344641/data/USTC/USTC_Snooker.yuv Snooker"
# )
datasets=(
  "/home/e/e1344641/data/USTC/USTC_Badminton.yuv  Badminton"
)

savdir="GaussianVideo_results_USTC"
savdir_m="GaussianVideo_models_USTC"
is_pos=False
is_ad=True
is_rm=True

loss_type="L2"
for dataset in "${datasets[@]}"; do
  dataset_path=$(echo $dataset | cut -d' ' -f1)
  data_name=$(echo $dataset | cut -d' ' -f2)
  for num_points in  10000  20000 30000 40000 50000; do
    for iterations in 100000; do
      model_path="/home/e/e1344641/GaussianVideo/checkpoints/GaussianVideo_models_USTC/${data_name}/GaussianVideo_${iterations}_${num_points}/gmodels_state_dict.pth"
    
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

      srun python train_video_Represent.py \
      --loss_type $loss_type \
      --dataset $dataset_path \
      --data_name $data_name\
      --num_points $num_points\
      --iterations $iterations \
      --savdir $savdir\
      --savdir_m $savdir_m \
      $pos_flag $ad_flag $rm_flag

      srun python train_video_Compress.py \
      --dataset "$dataset_path" \
      --model_path "$model_path" \
      --data_name "$data_name" \
      --num_points "$num_points" \
      --iterations "$iterations" \
      --savdir "$savdir" \
      --savdir_m "$savdir_m" \
       $rm_flag 
    done
  done
done
