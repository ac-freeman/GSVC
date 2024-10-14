import numpy as np
import cv2
import os

# 参数设置
width, height = 960, 540  # 视频分辨率
frames = 2  # 每个视频的帧数

# 保存路径
output_dir_a1 = '/home/e/e1344641/data/UVG/E1'
os.makedirs(output_dir_a1, exist_ok=True)

# 定义黑白格子图案（黑白相间）
def generate_bw_grid(w, h, block_size=10):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            # 根据格子位置决定颜色：交替设置黑色和白色
            if (x // block_size + y // block_size) % 2 == 0:
                color = (255, 255, 255)  # 白色
            else:
                color = (0, 0, 0)  # 黑色
            image[y:y+block_size, x:x+block_size] = color
    return image

# 生成视频帧
def create_frame(even_frame, odd_frame, video_type='E1'):
    for i in range(1, frames + 1):
        if i % 2 == 0:  # 偶数帧
            frame = even_frame
        else:  # 奇数帧
            frame = odd_frame

        if video_type == 'E1':
            file_path = os.path.join(output_dir_a1, f'frame_{i:02d}.jpg')

        # 保存为jpg文件
        cv2.imwrite(file_path, frame)

# 生成 A1 视频的帧
def generate_a1_frames():
    # 偶数帧和奇数帧都为全屏的黑白格子
    even_frame = generate_bw_grid(width, height)
    odd_frame = generate_bw_grid(width, height)

    create_frame(even_frame, odd_frame, video_type='E1')

# 生成并保存为jpg文件
generate_a1_frames()

