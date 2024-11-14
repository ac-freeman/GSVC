#!/bin/bash

#SBATCH --job-name=test    # Job name
#SBATCH --output=mix_output.txt # Standard output and error log
#SBATCH --error=mix_error.txt  # Error log
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=16G                    # Request 16GB of memory
# Activate the environment if needed
source activate torch  # Replace 'torch' with the name of your conda environment

# 加载 ffmpeg（如果需要的话，具体路径请根据集群环境调整）
module load ffmpeg

# 设置视频输入和输出路径
VIDEO_PATH=/home/e/e1344641/data/UVG
OUTPUT_PATH=/home/e/e1344641/data/UVG/Mix
OUTPUT_FILE=Mix_1920x1080_120fps_420_8bit_YUV_960x540.yuv

# 提取 Beauty 的前90帧
ffmpeg -s 960x540 -pix_fmt yuv420p -t 3 -i $VIDEO_PATH/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV_960x540.yuv -frames:v 90 $VIDEO_PATH/Beauty/Beauty_90frames.yuv

# 提取 HoneyBee 的前90帧
ffmpeg -s 960x540 -pix_fmt yuv420p -t 3 -i $VIDEO_PATH/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV_960x540.yuv -frames:v 90 $VIDEO_PATH/HoneyBee/HoneyBee_90frames.yuv

# 提取 Jockey 的前90帧
ffmpeg -s 960x540 -pix_fmt yuv420p -t 3 -i $VIDEO_PATH/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV_960x540.yuv -frames:v 90 $VIDEO_PATH/Jockey/Jockey_90frames.yuv

# 创建文件列表以拼接视频
echo "file '$VIDEO_PATH/Beauty/Beauty_90frames.yuv'" > $OUTPUT_PATH/filelist.txt
echo "file '$VIDEO_PATH/HoneyBee/HoneyBee_90frames.yuv'" >> $OUTPUT_PATH/filelist.txt
echo "file '$VIDEO_PATH/Jockey/Jockey_90frames.yuv'" >> $OUTPUT_PATH/filelist.txt

# 拼接视频并保存为输出文件
ffmpeg -f concat -safe 0 -i $OUTPUT_PATH/filelist.txt -c:v copy -pix_fmt yuv420p $OUTPUT_PATH/$OUTPUT_FILE

# 清理临时文件
rm $VIDEO_PATH/Beauty/Beauty_90frames.yuv
rm $VIDEO_PATH/HoneyBee/HoneyBee_90frames.yuv
rm $VIDEO_PATH/Jockey/Jockey_90frames.yuv
rm $OUTPUT_PATH/filelist.txt

echo "Video processing completed successfully!"