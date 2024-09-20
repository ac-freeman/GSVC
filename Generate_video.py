import os
import cv2
import numpy as np

# 创建保存目录
save_dir = '/home/e/e1344641/data/UVG/output_frames'
os.makedirs(save_dir, exist_ok=True)

# 视频参数
width, height = 1920, 1080
total_frames = 20

# 彩色格子的生成
def generate_color_grid(rows, cols):
    grid = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            grid[i, j] = np.random.randint(0, 255, 3)
    return grid

# 生成偶数帧（黑色背景+彩色矩形）
def generate_even_frame():
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    color_grid = generate_color_grid(height // 2, width // 2)
    frame[height//4:3*height//4, width//4:3*width//4] = color_grid
    return frame

# 生成奇数帧（彩色背景+黑色矩形）
def generate_odd_frame():
    frame = generate_color_grid(height, width)
    frame[height//4:3*height//4, width//4:3*width//4] = 0  # 黑色矩形
    return frame

# 保存每一帧
for i in range(total_frames):
    if i % 2 == 0:  # 偶数帧
        frame = generate_even_frame()
    else:  # 奇数帧
        frame = generate_odd_frame()
    
    frame_filename = os.path.join(save_dir, f'frame_{i:03d}.png')
    cv2.imwrite(frame_filename, frame)

print(f"所有帧已保存到: {save_dir}")
