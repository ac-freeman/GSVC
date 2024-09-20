import os
import cv2
import numpy as np
from tqdm import tqdm  # 导入进度条

# 创建保存目录
save_dir_1 = '/home/e/e1344641/data/UVG/output_frames_1'
os.makedirs(save_dir_1, exist_ok=True)
save_dir_2 = '/home/e/e1344641/data/UVG/output_frames_2'
os.makedirs(save_dir_2, exist_ok=True)

# 视频参数
width, height = 1920, 1080
total_frames = 2
grid_size = 5  # 每个彩色格子的长宽

# 彩色格子的生成
def generate_color_grid(rows, cols, grid_size):
    grid = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(0, rows, grid_size):
        for j in range(0, cols, grid_size):
            # 为每个彩色格子生成随机颜色
            color = np.random.randint(0, 255, 3, dtype=np.uint8)
            # 将该颜色填充到 grid_size x grid_size 的格子中
            grid[i:i+grid_size, j:j+grid_size] = color
    return grid

# 生成偶数帧（黑色背景+彩色矩形）
def generate_center_colorful_frame():
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    color_grid = generate_color_grid(height // 2, width // 2, grid_size)
    frame[height//4:3*height//4, width//4:3*width//4] = color_grid
    return frame

# 生成奇数帧（彩色背景+黑色矩形）
def generate_center_black_frame():
    frame = generate_color_grid(height, width, grid_size)
    frame[height//4:3*height//4, width//4:3*width//4] = 0  # 黑色矩形
    return frame

# 保存每一帧并显示进度条
for i in tqdm(range(total_frames), desc="Saving frames", unit="frame"):
    if i % 2 == 0:  # 偶数帧
        frame = generate_center_colorful_frame()
    else:  # 奇数帧
        frame = generate_center_black_frame()
    
    frame_filename = os.path.join(save_dir_1, f'frame_{i:03d}.png')
    cv2.imwrite(frame_filename, frame)

print(f"所有帧已保存到: {save_dir_1}")

# 保存每一帧并显示进度条
for i in tqdm(range(total_frames), desc="Saving frames", unit="frame"):
    if i % 2 == 0:  # 偶数帧
        frame = generate_center_black_frame()
    else:  # 奇数帧
        frame = generate_center_colorful_frame()
    
    frame_filename = os.path.join(save_dir_2, f'frame_{i:03d}.png')
    cv2.imwrite(frame_filename, frame)

print(f"所有帧已保存到: {save_dir_2}")
