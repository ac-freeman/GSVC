import numpy as np
import cv2
from tqdm import tqdm
import os
# 视频的宽度和高度
width = 1920
height = 1080

# 每一帧 YUV 数据的大小
frame_size = width * height * 3 // 2

# 输入 YUV 视频路径
input_file = "/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv"
output_file = "/home/e/e1344641/data/UVG/Beauty/Beauty_transformed_1920x1080_120fps_420_8bit_YUV.yuv"

# 计算总帧数 (文件大小 / 每帧大小)
file_size = os.path.getsize(input_file)
total_frames = file_size // frame_size
print(total_frames)
# 打开输入和输出文件
with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
    # 使用 tqdm 显示进度条
    for frame_index in tqdm(range(total_frames), desc="Processing frames"):
        # 读取一帧
        yuv_frame = f_in.read(frame_size)
        if not yuv_frame:
            break

        # 将帧分成 Y, U, V 平面
        y_plane = np.frombuffer(yuv_frame[:width * height], dtype=np.uint8).reshape((height, width))
        u_plane = np.frombuffer(yuv_frame[width * height:width * height + (width // 2) * (height // 2)], dtype=np.uint8).reshape((height // 2, width // 2))
        v_plane = np.frombuffer(yuv_frame[width * height + (width // 2) * (height // 2):], dtype=np.uint8).reshape((height // 2, width // 2))

        # 对 Y 平面进行切割和翻转操作
        top_left = y_plane[:height//2, :width//2]
        top_right = y_plane[:height//2, width//2:]
        bottom_left = y_plane[height//2:, :width//2]
        bottom_right = y_plane[height//2:, width//2:]

        # 左右、上下翻转每个部分
        top_left_flipped = cv2.flip(top_left, -1)  # 左右上下翻转
        top_right_flipped = cv2.flip(top_right, -1)
        bottom_left_flipped = cv2.flip(bottom_left, -1)
        bottom_right_flipped = cv2.flip(bottom_right, -1)

        # 重新组合 Y 平面
        top_combined = np.hstack((top_left_flipped, top_right_flipped))
        bottom_combined = np.hstack((bottom_left_flipped, bottom_right_flipped))
        y_plane_transformed = np.vstack((top_combined, bottom_combined))

        # 对 U, V 平面进行相同操作
        top_left_u = u_plane[:height//4, :width//4]
        top_right_u = u_plane[:height//4, width//4:]
        bottom_left_u = u_plane[height//4:, :width//4]
        bottom_right_u = u_plane[height//4:, width//4:]

        top_left_v = v_plane[:height//4, :width//4]
        top_right_v = v_plane[:height//4, width//4:]
        bottom_left_v = v_plane[height//4:, :width//4]
        bottom_right_v = v_plane[height//4:, width//4:]

        top_left_u_flipped = cv2.flip(top_left_u, -1)
        top_right_u_flipped = cv2.flip(top_right_u, -1)
        bottom_left_u_flipped = cv2.flip(bottom_left_u, -1)
        bottom_right_u_flipped = cv2.flip(bottom_right_u, -1)

        top_left_v_flipped = cv2.flip(top_left_v, -1)
        top_right_v_flipped = cv2.flip(top_right_v, -1)
        bottom_left_v_flipped = cv2.flip(bottom_left_v, -1)
        bottom_right_v_flipped = cv2.flip(bottom_right_v, -1)

        top_combined_u = np.hstack((top_left_u_flipped, top_right_u_flipped))
        bottom_combined_u = np.hstack((bottom_left_u_flipped, bottom_right_u_flipped))
        u_plane_transformed = np.vstack((top_combined_u, bottom_combined_u))

        top_combined_v = np.hstack((top_left_v_flipped, top_right_v_flipped))
        bottom_combined_v = np.hstack((bottom_left_v_flipped, bottom_right_v_flipped))
        v_plane_transformed = np.vstack((top_combined_v, bottom_combined_v))

        # 将 Y, U, V 平面写入输出文件
        f_out.write(y_plane_transformed.tobytes())
        f_out.write(u_plane_transformed.tobytes())
        f_out.write(v_plane_transformed.tobytes())

with open(output_file, 'rb') as f_in:
    # 读取第一帧
    yuv_frame = f_in.read(frame_size)
    if yuv_frame:
        # 将 YUV 数据分解成 Y, U, V 三个平面
        y_plane = np.frombuffer(yuv_frame[:width * height], dtype=np.uint8).reshape((height, width))
        u_plane = np.frombuffer(yuv_frame[width * height:width * height + (width // 2) * (height // 2)], dtype=np.uint8).reshape((height // 2, width // 2))
        v_plane = np.frombuffer(yuv_frame[width * height + (width // 2) * (height // 2):], dtype=np.uint8).reshape((height // 2, width // 2))

        # 扩展 U 和 V 平面的大小，使其与 Y 平面匹配
        u_plane_upscaled = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_LINEAR)
        v_plane_upscaled = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_LINEAR)

        # 合并 YUV 平面到 YUV 格式图像
        yuv_img = cv2.merge((y_plane, u_plane_upscaled, v_plane_upscaled))

        # 将 YUV 图像转换为 BGR 格式 (适合 OpenCV 显示或保存)
        bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

        # 保存处理后的第一帧为 JPG 图片
        cv2.imwrite('/home/e/e1344641/data/UVG/Beauty/first_frame_transformed_beauty.jpg', bgr_img)