import numpy as np
from tqdm import tqdm  # 进度条库

# 视频分辨率
width = 1920
height = 1080

# 每帧的大小：Y 组件大小 + U 组件大小 + V 组件大小
frame_size = width * height * 3 // 2

# 初始化帧存储列表
video_frames = []

# 计算文件总大小以估算帧的数量
file_path = '/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv'
file_size = os.path.getsize(file_path)
total_frames = file_size // frame_size

# 打开YUV文件
with open(file_path, 'rb') as f:
    for _ in tqdm(range(total_frames), desc="Reading YUV frames"):
        # 读取单帧
        frame_data = f.read(frame_size)
        if not frame_data:
            break  # 读取结束

        # 将读取的数据存储为numpy数组以便进一步处理
        yuv_frame = np.frombuffer(frame_data, dtype=np.uint8)

        # 将当前帧添加到列表中
        video_frames.append(yuv_frame)

# 现在video_frames中包含所有的YUV帧数据
print(f"Total frames: {len(video_frames)}")
print(np.shape(video_frames[0]))
