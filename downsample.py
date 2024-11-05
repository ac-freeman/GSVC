import cv2
import numpy as np
import os

def read_yuv_frame(file, width, height):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    # 读取Y、U、V平面
    y = np.frombuffer(file.read(y_size), dtype=np.uint8).reshape((height, width))
    u = np.frombuffer(file.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))
    v = np.frombuffer(file.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2))

    return y, u, v

def write_yuv_frame(file, y, u, v):
    file.write(y.tobytes())
    file.write(u.tobytes())
    file.write(v.tobytes())

def downscale_yuv(input_path, output_path, input_width, input_height, scale_factor):
    output_width = int(input_width * scale_factor)
    output_height = int(input_height * scale_factor)

    with open(input_path, 'rb') as input_file, open(output_path, 'wb') as output_file:
        while True:
            try:
                y, u, v = read_yuv_frame(input_file, input_width, input_height)
            except ValueError:
                break  # 文件结束

            # 缩放Y、U和V平面
            y_resized = cv2.resize(y, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            u_resized = cv2.resize(u, (output_width // 2, output_height // 2), interpolation=cv2.INTER_LINEAR)
            v_resized = cv2.resize(v, (output_width // 2, output_height // 2), interpolation=cv2.INTER_LINEAR)

            # 写入缩放后的帧
            write_yuv_frame(output_file, y_resized, u_resized, v_resized)

# 文件列表和参数设置
files = [
    "/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv",
    "/home/e/e1344641/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv",
    "/home/e/e1344641/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv"
]
# files = [
#     "./Loadmodel/Beauty_1920x1080_120fps_420_8bit_YUV.yuv"
# ]
input_width, input_height = 1920, 1080
scale_factor = 0.5  # 将分辨率减半

# 批量处理每个文件
for input_path in files:
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_960x540{ext}"
    
    downscale_yuv(input_path, output_path, input_width, input_height, scale_factor)
    print(f"Processed {input_path} -> {output_path}")

