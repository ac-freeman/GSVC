import os
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path



class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    def write(self, text):
        # 打印到控制台
        print(text)
        # 追加到文件
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')


def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    return loss

def strip_lowerdiag(L):
    if L.shape[1] == 3:
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]

    elif L.shape[1] == 2:
        uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 1, 1]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_rotation_2d(r):
    '''
    Build rotation matrix in 2D.
    '''
    R = torch.zeros((r.size(0), 2, 2), device='cuda')
    R[:, 0, 0] = torch.cos(r)[:, 0]
    R[:, 0, 1] = -torch.sin(r)[:, 0]
    R[:, 1, 0] = torch.sin(r)[:, 0]
    R[:, 1, 1] = torch.cos(r)[:, 0]
    return R

def build_scaling_rotation_2d(s, r, device):
    L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device='cuda')
    R = build_rotation_2d(r, device)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L = R @ L
    return L
    
def build_covariance_from_scaling_rotation_2d(scaling, scaling_modifier, rotation, device):
    '''
    Build covariance metrix from rotation and scale matricies.
    '''
    L = build_scaling_rotation_2d(scaling_modifier * scaling, rotation, device)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_triangular(r):
    R = torch.zeros((r.size(0), 2, 2), device=r.device)
    R[:, 0, 0] = r[:, 0]
    R[:, 1, 0] = r[:, 1]
    R[:, 1, 1] = r[:, 2]
    return R



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


def generate_video(savdir,image_list, data_name, model_name,fps,iterations,num_points,origin):
    video_path = Path(f"./checkpoints/{savdir}/{data_name}/{model_name}_{iterations}_{num_points}/video")
    video_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    # Define the output video file name
    if origin:
        filename = "video.mp4"
    else:
        filename = "combined_video.mp4"
    # Get the size of the first image dynamically
    first_image = image_list[0]
    width, height = first_image.size  # Extract the size of the first image
    # Create the video writer with the actual image dimensions
    output_size = (width, height)
    video = cv2.VideoWriter(str(video_path / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    # Add images to the video writer
    for img in tqdm(image_list, desc="Processing images", unit="image"):  # Iterate directly over the image_list      
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(img_cv)
    # Finalize and close the video writer
    video.release()
    print("MP4 video created successfully.")

def generate_video_test(videodir,image_list, data_name, model_name,fps,iterations,num_points,origin):
    video_path = Path(f"./checkpoints/{videodir}/{data_name}/{model_name}_{iterations}_{num_points}/video")
    video_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    # Define the output video file name
    if origin:
        filename = "video.mp4"
    else:
        filename = "combined_video.mp4"
    # Get the size of the first image dynamically
    first_image = image_list[0]
    width, height = first_image.size  # Extract the size of the first image
    # Create the video writer with the actual image dimensions
    output_size = (width, height)
    video = cv2.VideoWriter(str(video_path / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    # Add images to the video writer
    for img in tqdm(image_list, desc="Processing images", unit="image"):  # Iterate directly over the image_list      
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(img_cv)
    # Finalize and close the video writer
    video.release()
    if origin:
        print("video.mp4: MP4 video created successfully.")
    else:
        print("combined_video.mp4: MP4 video created successfully.")


def generate_video_density(videodir,image_list, data_name, model_name,fps,iterations,num_points,origin):
    video_path = Path(f"./checkpoints/{videodir}/{data_name}/{model_name}_{iterations}_{num_points}/video")
    video_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    # Define the output video file name
    if origin:
        filename = "video.mp4"
    else:
        filename = "combined_video.mp4"
    # Get the size of the first image dynamically
    first_image = image_list[0]
    width, height = first_image.size  # Extract the size of the first image
    # Create the video writer with the actual image dimensions
    output_size = (width, height)
    video = cv2.VideoWriter(str(video_path / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    # Add images to the video writer
    for img in tqdm(image_list, desc="Processing images", unit="image"):  # Iterate directly over the image_list      
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(img_cv)
    # Finalize and close the video writer
    video.release()
    if origin:
        print("video.mp4: MP4 video created successfully.")
    else:
        print("combined_video.mp4: MP4 video created successfully.")

def downsample_image(image, scale_factor):
    # 使用模糊降低图像分辨率，保留尺寸不变
    downsampled_image = cv2.GaussianBlur(image, (scale_factor * 2 + 1, scale_factor * 2 + 1), 0)
    return downsampled_image

# def extend_image(image):
#     # 使用cv2.copyMakeBorder扩展图像，添加10像素的黑色边框
#     new_image = cv2.copyMakeBorder(
#         image, 
#         10, 10, 10, 10,  # 分别为上、下、左、右扩展的像素
#         cv2.BORDER_CONSTANT, 
#         value=[0, 0, 0]  # 黑色边框
#     )
#     return new_image
def extend_image(image, border_size=10):
    mean_color = np.mean(image, axis=(0, 1)).astype(np.uint8)
    H, W, C = image.shape
    block_size=2
    extended_image = np.zeros((H + 2 * border_size, W + 2 * border_size, C), dtype=np.uint8)
    extended_image[border_size:H + border_size, border_size:W + border_size] = image
    #上边界
    extended_image[border_size-1,:,:]=extended_image[border_size,:,:]
    for j in range(1,border_size):
        block_size=j*j+1
        for i in range(W):
            if(i-block_size//2>=0 and i+block_size//2<=W-1):
                block_image=np.zeros((block_size, block_size, C), dtype=np.uint8)
                block_image=extended_image[border_size-j:border_size+block_size-1-j,border_size+i-block_size//2:border_size+i+block_size//2,:]
            elif(i-block_size//2<0):
                block_image=np.zeros((block_size, i+block_size//2+1, C), dtype=np.uint8)
                block_image=extended_image[border_size-j:border_size+block_size-1-j,border_size:border_size+i+block_size//2,:]
            elif(i+block_size//2>W-1):
                block_image=np.zeros((block_size, W-1-i+block_size//2+1, C), dtype=np.uint8)
                block_image=extended_image[border_size-j:border_size+block_size-1-j,border_size+i-block_size//2:border_size+W-1,:]
            extended_image[border_size-1-j,border_size+i,:]=np.mean(block_image, axis=(0, 1)).astype(np.uint8)
    #下边界
    extended_image[border_size+H,:,:]=extended_image[border_size+H-1,:,:]
    for j in range(1,border_size):
        block_size=j*j+1
        for i in range(W):
            if(i-block_size//2>=0 and i+block_size//2<=W-1):
                block_image=np.zeros((block_size, block_size, C), dtype=np.uint8)
                block_image=extended_image[border_size+H-block_size+j:border_size+H-1+j,border_size+i-block_size//2:border_size+i+block_size//2,:]
            elif(i-block_size//2<0):
                block_image=np.zeros((block_size, i+block_size//2+1, C), dtype=np.uint8)
                block_image=extended_image[border_size+H-block_size+j:border_size+H-1+j,border_size:border_size+i+block_size//2,:]
            elif(i+block_size//2>W-1):
                block_image=np.zeros((block_size, W-1-i+block_size//2+1, C), dtype=np.uint8)
                block_image=extended_image[border_size+H-block_size+j:border_size+H-1+j,border_size+i-block_size//2:border_size+W-1,:]
            extended_image[H+border_size+j,border_size+i,:]=np.mean(block_image, axis=(0, 1)).astype(np.uint8)
    #左边界
    extended_image[:,border_size-1,:]=extended_image[:,border_size,:]
    for j in range(1,border_size):
        block_size=j*j+1
        for i in range(H):
            if(i-block_size//2>=0 and i+block_size//2<=H-1):
                block_image=np.zeros((block_size, block_size, C), dtype=np.uint8)
                block_image=extended_image[border_size+i-block_size//2:border_size+i+block_size//2,border_size-j:border_size+block_size-1-j,:]
            elif(i-block_size//2<0):
                block_image=np.zeros(( i+block_size//2+1,block_size, C), dtype=np.uint8)
                block_image=extended_image[border_size:border_size+i+block_size//2,border_size-j:border_size+block_size-1-j,:]
            elif(i+block_size//2>H-1):
                block_image=np.zeros(( H-1-i+block_size//2+1,block_size, C), dtype=np.uint8)
                block_image=extended_image[border_size+i-block_size//2:border_size+H-1,border_size-j:border_size+block_size-1-j,:]
            extended_image[border_size+i,border_size-1-j,:]=np.mean(block_image, axis=(0, 1)).astype(np.uint8)
    #右边界
    extended_image[:,border_size+W,:]=extended_image[:,border_size+W-1,:]
    for j in range(1,border_size):
        block_size=j*j+1
        for i in range(H):
            if(i-block_size//2>=0 and i+block_size//2<=H-1):
                block_image=np.zeros((block_size, block_size, C), dtype=np.uint8)
                block_image=extended_image[border_size+i-block_size//2:border_size+i+block_size//2,border_size+W-block_size+j:border_size+W-1+j,:]
            elif(i-block_size//2<0):
                block_image=np.zeros((block_size, i+block_size//2+1, C), dtype=np.uint8)
                block_image=extended_image[border_size:border_size+i+block_size//2,border_size+W-block_size+j:border_size+W-1+j,:]
            elif(i+block_size//2>W-1):
                block_image=np.zeros((block_size, W-1-i+block_size//2+1, C), dtype=np.uint8)
                block_image=extended_image[border_size+i-block_size//2:border_size+H-1,border_size+H-block_size+j:border_size+W-1+j,:]
            extended_image[border_size+i,W+border_size+j,:]=np.mean(block_image, axis=(0, 1)).astype(np.uint8)


    for i in range(border_size):
        for j in range(border_size):
            x = border_size-j-1
            y = border_size-i-1
            if x==0 and y==0:
                alpha_x=0.5
                alpha_y=0.5
            else:
                alpha_x=x/(x+y)
                alpha_y=y/(x+y)
            extended_image[i, j] = (1-alpha_x) * extended_image[i, border_size] + (1-alpha_y) * extended_image[border_size, j]
            extended_image[i,  W + 2*border_size -1-j] = (1-alpha_x) * extended_image[i,  W +border_size-1] + (1-alpha_y) * extended_image[border_size, W + 2*border_size -1-j]
            extended_image[H + 2*border_size-1-i,  j] = (1-alpha_x) * extended_image[H + 2*border_size -1-i,  border_size] + (1-alpha_y) * extended_image[H+border_size-1, j]
            extended_image[H + 2*border_size-1-i,  W + 2*border_size -1-j] = (1-alpha_x) * extended_image[H + 2*border_size -1-i,  W+border_size-1] + (1-alpha_y) * extended_image[H+border_size-1, W + 2*border_size -1-j]
    return extended_image
# def extend_image(image):
#     border_size=3
#     mean_color = np.mean(image, axis=(0, 1)).astype(np.uint8)
#     H, W, C = image.shape
#     extended_image = np.zeros((H + 2 * border_size, W + 2 * border_size, C), dtype=np.uint8)
#     extended_image[border_size:H + border_size, border_size:W + border_size] = image
#     for i in range(border_size):
#         alpha = i / (border_size-1)
#         # 插值上、下边框的颜色
#         extended_image[border_size-i-1, border_size:W + border_size] = (1-alpha) * image[0, :] + alpha * mean_color
#         extended_image[H + border_size + i, border_size:W + border_size] = (1 - alpha) * image[-1, :] + alpha * mean_color
#         # 插值左、右边框的颜色
#         extended_image[border_size:H + border_size, border_size-i-1] = (1 - alpha) * image[:, 0] + alpha * mean_color
#         extended_image[border_size:H + border_size, W + border_size + i] = (1 - alpha) * image[:, -1] + alpha * mean_color
#     for i in range(border_size):
#         for j in range(border_size):
#             x = border_size-j-1
#             y = border_size-i-1
#             if x==0 and y==0:
#                 alpha_x=0.5
#                 alpha_y=0.5
#             else:
#                 alpha_x=x/(x+y)
#                 alpha_y=y/(x+y)
#             extended_image[i, j] = (1-alpha_x) * extended_image[i, border_size] + (1-alpha_y) * extended_image[border_size, j]
#             extended_image[i,  W + 2*border_size -1-j] = (1-alpha_x) * extended_image[i,  W +border_size-1] + (1-alpha_y) * extended_image[border_size, W + 2*border_size -1-j]
#             extended_image[H + 2*border_size-1-i,  j] = (1-alpha_x) * extended_image[H + 2*border_size -1-i,  border_size] + (1-alpha_y) * extended_image[H+border_size-1, j]
#             extended_image[H + 2*border_size-1-i,  W + 2*border_size -1-j] = (1-alpha_x) * extended_image[H + 2*border_size -1-i,  W+border_size-1] + (1-alpha_y) * extended_image[H+border_size-1, W + 2*border_size -1-j]
#     return extended_image


def restor_image(new_image, H, W):
    # 假设 new_image 是 torch tensor 格式，形状为 [1, C, H_new, W_new]，需要裁剪恢复到 H 和 W
    # new_image 的形状为 [1, 3, H_new, W_new]
    
    # 确定我们需要裁剪的起始和结束位置
    top = (new_image.shape[2] - H) // 2
    left = (new_image.shape[3] - W) // 2
    
    # 裁剪出原始图像
    original_image = new_image[:, :, top:top+H, left:left+W]  # 保持 batch 和通道数不变，裁剪高度和宽度
    return original_image


class EarlyStopping:
    def __init__(self, patience=100, min_delta=0):
        self.patience = patience  # 容忍的迭代次数
        self.min_delta = min_delta  # 最小的改善幅度
        self.best_loss = None  # 用于存储最好的损失
        self.counter = 0  # 记录没有改善的次数

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False  # 不停止训练

        # 如果当前loss和之前最好的loss相比改善小于 min_delta，认为没有改善
        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0  # 重置计数器
        else:
            self.counter += 1

        # 如果计数器超过 patience，就停止训练
        if self.counter >= self.patience:
            return True  # 停止训练

        return False  # 继续训练