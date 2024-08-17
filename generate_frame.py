import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import pdb
# 视频分辨率
width = 1920
height = 1080

# 每帧的大小：Y 组件大小 + U 组件大小 + V 组件大小
frame_size = width * height * 3 // 2

# 初始化帧存储列表
video_frames_rgb = []

# 计算文件总大小以估算帧的数量
file_path = '/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv'
file_size = os.path.getsize(file_path)
total_frames = file_size // frame_size

# YUV to RGB conversion function
# def yuv420_to_rgb(yuv_frame, width, height):
#     y_size = width * height
#     uv_size = y_size // 4
#     #pdb.set_trace()
#     # Separate Y, U, and V planes
#     y = yuv_frame[:y_size].reshape((height, width))
#     u = yuv_frame[y_size:y_size + uv_size].reshape((height // 2, width // 2))
#     v = yuv_frame[y_size + uv_size:].reshape((height // 2, width // 2))

#     # Upsample U and V to the same size as Y
#     u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
#     v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
    
#     # Convert YUV to RGB
#     c = y - 16
#     d = u - 128
#     e = v - 128
#     r = (298 *c + 409 * e + 128) >> 8
#     g = (298 *c - 100 * d - 208 * e + 128) >> 8
#     b = (298 *c + 516 * d + 128) >> 8
    
#     # Clip values to be between 0 and 255
#     r = np.clip(r, 0, 255).astype(np.uint8)
#     g = np.clip(g, 0, 255).astype(np.uint8)
#     b = np.clip(b, 0, 255).astype(np.uint8)
    
#     # Stack the channels to form an RGB image
#     rgb_frame = np.stack([r, g, b], axis=-1)
#     return rgb_frame
def yuv420_to_rgb(yuv_frame, width, height):
    y_size = width * height
    uv_size = y_size // 4

    # 分离 Y、U、V 分量
    y = yuv_frame[:y_size].reshape((height, width))
    u = yuv_frame[y_size:y_size + uv_size].reshape((height // 2, width // 2))
    v = yuv_frame[y_size + uv_size:].reshape((height // 2, width // 2))

    # 将 U 和 V 上采样至与 Y 相同的分辨率
    u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

    # 使用不同的缩放因子进行 YUV 到 RGB 转换
    c = y - 16
    d = u - 128
    e = v - 128
    r = (1.164 * c + 1.596 * e).clip(0, 255).astype(np.uint8)
    g = (1.164 * c - 0.392 * d - 0.813 * e).clip(0, 255).astype(np.uint8)
    b = (1.164 * c + 2.017 * d).clip(0, 255).astype(np.uint8)

    # 叠加 R、G、B 三个通道形成 RGB 图像
    rgb_frame = np.stack([r, g, b], axis=-1)
    return rgb_frame

# 打开YUV文件
with open(file_path, 'rb') as f:
    for _ in tqdm(range(total_frames), desc="Reading YUV frames"):
        # 读取单帧
        frame_data = f.read(frame_size)
        if not frame_data:
            break  # 读取结束

        # 将读取的数据存储为numpy数组以便进一步处理
        yuv_frame = np.frombuffer(frame_data, dtype=np.uint8)

        # Convert YUV frame to RGB format
        rgb_frame = yuv420_to_rgb(yuv_frame, width, height)

        # 将当前RGB帧添加到列表中
        video_frames_rgb.append(rgb_frame)

# 现在 video_frames_rgb 中包含所有的RGB帧数据
print(f"Total frames: {len(video_frames_rgb)}")
print(np.shape(video_frames_rgb))
if video_frames_rgb:
    first_frame = video_frames_rgb[0]
    img = Image.fromarray(first_frame)
    img.save("first_frame.png")  # Save to your preferred location
    print("First frame saved successfully.")
else:
    print("No frames available to save.")
