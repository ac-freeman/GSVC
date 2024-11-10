import numpy as np
import cv2
import os


# 定义视频参数
width, height = 1920, 1080
num_frames = 12
fps = 5
output_video_path = 'output_video1.yuv'
output_image_folder = 'frames1'

# 创建图片文件夹
os.makedirs(output_image_folder, exist_ok=True)

# 创建YUV文件
with open(output_video_path, 'wb') as yuv_file:        
    for i in range(1, num_frames + 1, 2):  # 每两帧内容相同
        if i<=6:
            # 生成10像素大小的彩色网格背景
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            grid_size = 50
            for y in range(0, height, grid_size):
                for x in range(0, width, grid_size):
                    color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
                    frame[y:y+grid_size, x:x+grid_size] = color
        else:
            # 初始化黑色背景
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 决定小球的位置
            if (i // 2) % 2 == 0:  # 第1、2，5、6，9、10帧在左边
                position = (300, height // 2)
            else:                  # 第3、4，7、8，11、12帧在右边
                position = (width - 300, height // 2)

            # 创建一个带有对称渐变纹理的小球图案
            radius = 300
            ball_texture = np.zeros((radius * 2, radius * 2, 3), dtype=np.uint8)
            
            # 使用径向对称渐变纹理
            for y in range(radius * 2):
                for x in range(radius * 2):
                    # 计算到中心的距离
                    distance = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
                    if distance < radius:
                        # 对称渐变纹理：根据距离生成颜色
                        intensity = int((1 - distance / radius) * 255)
                        r = intensity  # 红色分量根据距离衰减
                        g = intensity  # 绿色分量也根据距离衰减
                        b = 255 - intensity  # 蓝色为反向渐变
                        ball_texture[y, x] = (b, g, r)
            
            # 将带纹理的小球叠加到背景帧上
            for y in range(-radius, radius):
                for x in range(-radius, radius):
                    if np.sqrt(x**2 + y**2) < radius:
                        px, py = position[0] + x, position[1] + y
                        if 0 <= px < width and 0 <= py < height:
                            frame[py, px] = ball_texture[y + radius, x + radius]

        # 保存两帧相同的PNG图片和YUV数据
        for j in range(2):
            # 保存帧为PNG图片
            frame_num = i + j
            image_path = os.path.join(output_image_folder, f'frame_{frame_num:02d}.png')
            cv2.imwrite(image_path, frame)

            # 转换到YUV格式并写入YUV文件
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            yuv_file.write(yuv_frame.tobytes())
