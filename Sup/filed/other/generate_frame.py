import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def process_yuv_video(file_path, width, height):
    # 计算每帧大小 (YUV420格式)
    frame_size = width * height * 3 // 2
    # 计算总帧数
    file_size = os.path.getsize(file_path)
    total_frames = file_size // frame_size
    # 存储所有帧的列表
    video_frames = []
    # 打开YUV文件
    with open(file_path, 'rb') as f:
        # 使用tqdm展示进度
        for _ in tqdm(range(total_frames), desc="Processing frames"):
            # 读取一帧YUV数据
            yuv_frame = f.read(frame_size)
            if not yuv_frame:
                break  # 如果读取完毕，则退出循环
            # 将YUV数据转换为numpy数组
            yuv = np.frombuffer(yuv_frame, dtype=np.uint8).reshape((height * 3 // 2, width))
            # 将YUV420转换为BGR格式
            rgb_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            # 将BGR帧存入列表
            video_frames.append(rgb_frame)
    return video_frames

if __name__ == "__main__":
    # 文件路径
    file_path = '/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv'
    # 视频帧的分辨率
    width = 1920
    height = 1080
    video_frames = process_yuv_video(file_path, width, height)
    # 展示并保存第一帧
    if video_frames:
        first_frame = video_frames[0]
        # 使用matplotlib展示第一帧
        plt.imshow(first_frame)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
        # 保存第一帧为图片文件
        cv2.imwrite('first_frame.png', first_frame)
        print(np.shape(first_frame))
    else:
        print("No frames were read from the file.")