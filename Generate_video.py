import numpy as np
import cv2
import os
import subprocess

# 参数设置
width, height = 1920, 1080  # 视频分辨率
frames = 2  # 每个视频的帧数

# 保存路径
output_dir_a1 = '/home/e/e1344641/data/UVG/A1'
output_dir_a2 = '/home/e/e1344641/data/UVG/A2'
os.makedirs(output_dir_a1, exist_ok=True)
os.makedirs(output_dir_a2, exist_ok=True)

# 定义彩色格子图案
def generate_color_grid(w, h, block_size=40):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            image[y:y+block_size, x:x+block_size] = color
    return image

# 生成视频帧
def create_frame(even_frame, odd_frame, video_type='A1'):
    for i in range(1, frames+1):
        if i % 2 == 0:  # 偶数帧
            frame = even_frame
        else:  # 奇数帧
            frame = odd_frame
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        if video_type == 'A1':
            file_path = os.path.join(output_dir_a1, f'frame_{i:02d}.yuv')
        else:
            file_path = os.path.join(output_dir_a2, f'frame_{i:02d}.yuv')
        frame_yuv.tofile(file_path)

# 生成 A1 视频的帧
def generate_a1_frames():
    # 偶数帧：黑色背景，彩色格子矩形
    even_frame = np.zeros((height, width, 3), dtype=np.uint8)
    grid = generate_color_grid(width // 2, height // 2)
    even_frame[height//4:3*height//4, width//4:3*width//4] = grid

    # 奇数帧：彩色背景，黑色矩形
    odd_frame = generate_color_grid(width, height)
    cv2.rectangle(odd_frame, (width//4, height//4), (3*width//4, 3*height//4), (0, 0, 0), -1)

    create_frame(even_frame, odd_frame, video_type='A1')

# 生成 A2 视频的帧
def generate_a2_frames():
    # 奇数帧：黑色背景，彩色格子矩形
    odd_frame = np.zeros((height, width, 3), dtype=np.uint8)
    grid = generate_color_grid(width // 2, height // 2)
    odd_frame[height//4:3*height//4, width//4:3*width//4] = grid

    # 偶数帧：彩色背景，黑色矩形
    even_frame = generate_color_grid(width, height)
    cv2.rectangle(even_frame, (width//4, height//4), (3*width//4, 3*height//4), (0, 0, 0), -1)

    create_frame(even_frame, odd_frame, video_type='A2')

# 合成视频
def generate_file_list(frame_dir, file_list_path):
    # 获取文件夹中的所有帧文件并按自然顺序排序
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.yuv')])
    
    # 将帧文件写入临时文件，供 ffmpeg 使用
    with open(file_list_path, 'w') as f:
        for frame in frames:
            f.write(f"file '{os.path.join(frame_dir, frame)}'\n")

def combine_frames_to_video(output_path, frame_dir):
    # 创建临时文件用于存储帧列表
    file_list_path = os.path.join(frame_dir, 'frames_list.txt')
    generate_file_list(frame_dir, file_list_path)
    
    # 使用 ffmpeg 读取帧列表并合成视频
    command = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', file_list_path,
        '-vsync', 'vfr', '-pix_fmt', 'yuv420p', output_path
    ]
    
    print("Running command:", ' '.join(command))
    subprocess.run(command)

    # 删除临时文件
    os.remove(file_list_path)

# 生成并保存视频
generate_a1_frames()
generate_a2_frames()

# 合成YUV视频文件
combine_frames_to_video(os.path.join(output_dir_a1, 'A1_transformed_1920x1080_120fps_420_8bit_YUV.yuv'), output_dir_a1)
combine_frames_to_video(os.path.join(output_dir_a2, 'A2_transformed_1920x1080_120fps_420_8bit_YUV.yuv'), output_dir_a2)

print("视频生成完毕！")
