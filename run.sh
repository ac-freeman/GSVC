#!/bin/bash

#SBATCH --job-name=videogs_loss_job    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications



# Activate the environment if needed
source activate torch     # Replace 'myenv' with the name of your conda environment

<<<<<<< HEAD
data_path="/home/e/e1344641/data/kodak"

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 800 1000 3000 5000 7000 9000
do
srun python train.py -d $data_path \
--data_name kodak --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000
done
=======
>>>>>>> 3978ed3ad653c5c51bbb2780069bce1ad1c86c11
