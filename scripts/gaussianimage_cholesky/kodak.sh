#!/bin/bash

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
