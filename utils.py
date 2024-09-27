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

def extend_image(image):
    # Get image dimensions
    H, W, C = image.shape
    add = 10
    n = 1
    neighbor_size = int(min(H / 4, W / 4))  # Reference more neighboring pixels
    
    kernel_size_value = 65  # Must be a positive odd integer
    ksize = (kernel_size_value, kernel_size_value)
    sigma = 0  # Sigma value; 0 means it's calculated by OpenCV

    # Create a blurred version of the image
    blurred_image = cv2.GaussianBlur(image, ksize, sigma)

    # Create a new image with added borders
    new_image = np.zeros((H + add * 2, W + add * 2, C), dtype=np.uint8)

    # Place the original image in the center
    new_image[add:H + add, add:W + add, :] = image

    # Top border
    mean_row_top = np.mean(blurred_image[0:neighbor_size, :, :], axis=0)
    for i in range(add):
        w = ((add - i) / add) ** n
        edge_row_top = np.mean(image[0:i + 1, :, :], axis=0)
        new_image[add - i - 1, add:W + add, :] = (w * edge_row_top + (1 - w) * mean_row_top).astype(np.uint8)

    # Bottom border
    mean_row_bottom = np.mean(blurred_image[max(0, H - neighbor_size):H, :, :], axis=0)
    for i in range(add):
        w = ((add - i) / add) ** n
        edge_row_bottom = np.mean(image[H - i - 1:H, :, :], axis=0)
        new_image[H + add + i, add:W + add, :] = (w * edge_row_bottom + (1 - w) * mean_row_bottom).astype(np.uint8)

    # Left border
    mean_col_left = np.mean(blurred_image[:, 0:neighbor_size, :], axis=1)
    for i in range(add):
        w = ((add - i) / add) ** n
        edge_col_left = np.mean(image[:, 0:i + 1, :], axis=1)
        new_image[add:H + add, add - i - 1, :] = (w * edge_col_left + (1 - w) * mean_col_left).astype(np.uint8)

    # Right border
    mean_col_right = np.mean(blurred_image[:, max(0, W - neighbor_size):W, :], axis=1)
    for i in range(add):
        w = ((add - i) / add) ** n
        edge_col_right = np.mean(image[:, W - i - 1:W, :], axis=1)
        new_image[add:H + add, W + add + i, :] = (w * edge_col_right + (1 - w) * mean_col_right).astype(np.uint8)

    # Corners
    # Mean values for corners
    mean_pixel_top_left = np.mean(blurred_image[0:neighbor_size, 0:neighbor_size, :], axis=(0, 1))
    mean_pixel_top_right = np.mean(blurred_image[0:neighbor_size, W - neighbor_size:W, :], axis=(0, 1))
    mean_pixel_bottom_left = np.mean(blurred_image[H - neighbor_size:H, 0:neighbor_size, :], axis=(0, 1))
    mean_pixel_bottom_right = np.mean(blurred_image[H - neighbor_size:H, W - neighbor_size:W, :], axis=(0, 1))

    # Top-left corner
    for i in range(add):
        for j in range(add):
            w_row = ((add - i) / add) ** n
            w_col = ((add - j) / add) ** n
            w = w_row * w_col
            edge_pixel = np.mean(image[0:i + 1, 0:j + 1, :], axis=(0, 1))
            new_image[add - i - 1, add - j - 1, :] = (w * edge_pixel + (1 - w) * mean_pixel_top_left).astype(np.uint8)

    # Top-right corner
    for i in range(add):
        for j in range(add):
            w_row = ((add - i) / add) ** n
            w_col = (j / add) ** n
            w = w_row * (1 - w_col)
            edge_pixel = np.mean(image[0:i + 1, W - j - 1:W, :], axis=(0, 1))
            new_image[add - i - 1, W + add + j, :] = (w * edge_pixel + (1 - w) * mean_pixel_top_right).astype(np.uint8)

    # Bottom-left corner
    for i in range(add):
        for j in range(add):
            w_row = (i / add) ** n
            w_col = ((add - j) / add) ** n
            w = (1 - w_row) * w_col
            edge_pixel = np.mean(image[H - i - 1:H, 0:j + 1, :], axis=(0, 1))
            new_image[H + add + i, add - j - 1, :] = (w * edge_pixel + (1 - w) * mean_pixel_bottom_left).astype(np.uint8)

    # Bottom-right corner
    for i in range(add):
        for j in range(add):
            w_row = (i / add) ** n
            w_col = (j / add) ** n
            w = (1 - w_row) * (1 - w_col)
            edge_pixel = np.mean(image[H - i - 1:H, W - j - 1:W, :], axis=(0, 1))
            new_image[H + add + i, W + add + j, :] = (w * edge_pixel + (1 - w) * mean_pixel_bottom_right).astype(np.uint8)

    return new_image


def restor_image(new_image, H, W):
    # 假设 new_image 是 torch tensor 格式，形状为 [1, C, H_new, W_new]，需要裁剪恢复到 H 和 W
    # new_image 的形状为 [1, 3, H_new, W_new]
    
    # 确定我们需要裁剪的起始和结束位置
    top = (new_image.shape[2] - H) // 2
    left = (new_image.shape[3] - W) // 2
    
    # 裁剪出原始图像
    original_image = new_image[:, :, top:top+H, left:left+W]  # 保持 batch 和通道数不变，裁剪高度和宽度
    return original_image