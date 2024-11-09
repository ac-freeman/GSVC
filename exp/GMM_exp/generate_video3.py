import numpy as np
import cv2
import os


# 定义视频参数
width, height = 1920, 1080
num_frames = 12
fps = 5
output_video_path = 'output_video2.yuv'
output_image_folder = 'frames2'

# 创建图片文件夹
os.makedirs(output_image_folder, exist_ok=True)

# 创建YUV文件
with open(output_video_path, 'wb') as yuv_file:
    for i in range(1, num_frames + 1, 2):  # 每两帧内容相同
        # 初始化黑色背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        position = (width//2, height // 2)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        position = (width // 2, height // 2)
        
        # 决定长方形的位置和尺寸
        if (i // 2) % 2 == 0:  # 第1、2，5、6，9、10帧在左边
            rect_height = 500  # 长方形的高度
            rect_width = 100    # 长方形的宽度
        else:                  # 第3、4，7、8，11、12帧在右边
            rect_height = 100
            rect_width = 500

        # 计算长方形的左上角和右下角坐标（以长方形中心为position）
        top_left = (position[0] - rect_width // 2, position[1] - rect_height // 2)
        bottom_right = (position[0] + rect_width // 2, position[1] + rect_height // 2)
        
        # 在frame中绘制长方形
        frame = cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)  # 使用白色填充长方形

        # 保存两帧相同的PNG图片和YUV数据
        for j in range(2):
            # 保存帧为PNG图片
            frame_num = i + j
            image_path = os.path.join(output_image_folder, f'frame_{frame_num:02d}.png')
            cv2.imwrite(image_path, frame)

            # 转换到YUV格式并写入YUV文件
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            yuv_file.write(yuv_frame.tobytes())
