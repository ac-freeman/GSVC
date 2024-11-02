import math
import time
from pathlib import Path
import argparse
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import cv2
from pathlib import Path
import os
from scipy.interpolate import CubicSpline

class LoadGaussians:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        num_points,
        image,
        device,
        Model = None,
        args = None,
    ):
        self.device = device
        self.gt_image = image_to_tensor(image).to(self.device)
        self.num_points=num_points
        self.data_name=args.data_name
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        from Gaussian2D import GaussianImage_Cholesky
        self.gaussian_model = GaussianImage_Cholesky(num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
        device=self.device).to(self.device)
        if Model is not None:
            checkpoint = Model
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def render(self):     
        psnr_value, ms_ssim_value,img = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100       
        return psnr_value, ms_ssim_value, 1/test_end_time, img
    
    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
            out_image = out["render"]
        mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_image.float(), self.gt_image.float(), data_range=1, size_average=True).item()
        transform = transforms.ToPILImage()
        img = transform(out_image.float().squeeze(0))
        return psnr, ms_ssim_value,img


def image_to_tensor(img: Image.Image):
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv', help="Dataset Path"
    )
    parser.add_argument(
         "--model_path", type=str, default='/home/e/e1344641/GaussianVideo/models/Models/Beauty/GaussianImage_Cholesky_100000_30000/gmodels_state_dict.pth', help="Model Path"
    )
    parser.add_argument(
        "--data_name", type=str, default='Beauty', help="Training dataset"
    )
    parser.add_argument(
        "--fps", type=int, default=5, help="number of frames per second (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=4000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="width (default: %(default)s)"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="height (default: %(default)s)"
    )
    parser.add_argument("--savdir", type=str, default="result", help="Path to results")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    args = parser.parse_args(argv)
    return args

def main(argv):
    step=2
    args = parse_args(argv)
    savdir=args.savdir
    fps=args.fps
    width = args.width
    height = args.height
    model_path=args.model_path
    num_points=args.num_points
    device=torch.device("cuda:0")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
    logwriter = LogWriter(Path(f"./Loadmodel/{savdir}/{args.data_name}/{args.num_points}"))
    psnrs, ms_ssims, eval_fpses = [], [], []
    video_frames = process_yuv_video(args.dataset, width, height)
    start=0
    img_list=[]
    gmodels_state_dict = torch.load(model_path,map_location=device)
    destroied_gmodels_state_dict = {}
    restored_gmodels_state_dict = {}
    # num_frames = len(gmodels_state_dict)
    num_frames=50
    destroied_gmodels_frame=0
    for i in tqdm(range(start, start + num_frames), desc="Generate destroied video"):
         if i==0 or i%step==0:
            destroied_gmodels_frame+=1
            modelid=i + 1
            destroied_gmodels_state_dict[f"frame_{destroied_gmodels_frame}"] = gmodels_state_dict[f"frame_{modelid}"] 

    
    # #Interpolate
    # num_destroied_frames = len(destroied_gmodels_state_dict)
    # for i in range(num_destroied_frames - 1):
    #     # 获取两个相邻的损坏帧
    #     frame_id_start = f"frame_{i + 1}"
    #     frame_id_end = f"frame_{i + 2}"
        
    #     start_frame = destroied_gmodels_state_dict[frame_id_start]
    #     end_frame = destroied_gmodels_state_dict[frame_id_end]
        
    #     # 插入恢复帧数
    #     for j in range(step):
    #         # 计算当前插值帧的权重
    #         alpha = j / step
    #         # 插值恢复 _xyz
    #         interpolated_xyz = (1 - alpha) * start_frame['_xyz'] + alpha * end_frame['_xyz']
            
    #         # 插值恢复 _cholesky
    #         interpolated_cholesky = (1 - alpha) * start_frame['_cholesky'] + alpha * end_frame['_cholesky']
            
    #         # 插值恢复 _features_dc
    #         interpolated_features_dc = (1 - alpha) * start_frame['_features_dc'] + alpha * end_frame['_features_dc']
            
    #         # # 插值恢复 _xyz
    #         # interpolated_xyz = start_frame['_xyz']
            
    #         # # 插值恢复 _cholesky
    #         # interpolated_cholesky = start_frame['_cholesky']
            
    #         # # 插值恢复 _features_dc
    #         # interpolated_features_dc =start_frame['_features_dc']
            
            
    #         # 保存插值后的帧
    #         frame_index = i * step + j + 1
    #         restored_gmodels_state_dict[f"frame_{frame_index}"] = {
    #             '_xyz': interpolated_xyz,
    #             '_cholesky': interpolated_cholesky,
    #             '_features_dc': interpolated_features_dc
    #         }
    # # 将最后一个损坏帧直接添加到恢复字典中
    # restored_gmodels_state_dict[f"frame_{frame_index+1}"] = destroied_gmodels_state_dict[f"frame_{num_destroied_frames}"]
    
    num_destroied_frames = len(destroied_gmodels_state_dict)
    # 遍历每个损坏的帧段
    for i in range(num_destroied_frames - 1):
        # 获取相邻的两个损坏帧
        frame_id_start = f"frame_{i + 1}"
        frame_id_end = f"frame_{i + 2}"
        
        start_frame = destroied_gmodels_state_dict[frame_id_start]
        end_frame = destroied_gmodels_state_dict[frame_id_end]
        
        # 准备插值数据
        x = [0, step]  # 定义插值位置
        xyz_y = torch.stack([start_frame['_xyz'].detach(), end_frame['_xyz'].detach()]).cpu().numpy()
        cholesky_y = torch.stack([start_frame['_cholesky'].detach(), end_frame['_cholesky'].detach()]).cpu().numpy()
        features_dc_y = torch.stack([start_frame['_features_dc'].detach(), end_frame['_features_dc'].detach()]).cpu().numpy()
        
        # 创建三次样条插值函数
        spline_xyz = CubicSpline(x, xyz_y, axis=0)
        spline_cholesky = CubicSpline(x, cholesky_y, axis=0)
        spline_features_dc = CubicSpline(x, features_dc_y, axis=0)
        
        # 插入中间帧
        for j in range(step):
            alpha = j  # 对应位置
            interpolated_xyz = torch.tensor(spline_xyz(alpha), device=start_frame['_xyz'].device)
            interpolated_cholesky = torch.tensor(spline_cholesky(alpha), device=start_frame['_cholesky'].device)
            interpolated_features_dc = torch.tensor(spline_features_dc(alpha), device=start_frame['_features_dc'].device)
            
            # 对 _features_dc 的插值结果进行裁剪到 [0, 1] 范围
            interpolated_features_dc = torch.clamp(interpolated_features_dc, 0, 1)
            
            # 保存插值帧
            frame_index = i * step + j + 1
            restored_gmodels_state_dict[f"frame_{frame_index}"] = {
                '_xyz': interpolated_xyz,
                '_cholesky': interpolated_cholesky,
                '_features_dc': interpolated_features_dc
            }

    # 将最后一个损坏帧直接添加到恢复字典中
    restored_gmodels_state_dict[f"frame_{frame_index + 1}"] = destroied_gmodels_state_dict[f"frame_{num_destroied_frames}"]

    num_frames = len(restored_gmodels_state_dict)
    for i in tqdm(range(start, start + num_frames), desc="Processing Frames"):
        frame_num=i+1
        modelid=f"frame_{i + 1}"
        Model = restored_gmodels_state_dict[modelid]
        Gaussianframe = LoadGaussians(num_points=num_points,image=video_frames[i], Model=Model,device=device,args=args)
        psnr, ms_ssim,eval_fps, img = Gaussianframe.render()
        img_list.append(img)
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        eval_fpses.append(eval_fps)
        torch.cuda.empty_cache()
        # logwriter.write(
        #         "Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, FPS:{:.4f}\n".format(
        #             frame_num, Gaussianframe.H, Gaussianframe.W, psnr, ms_ssim, eval_fps
        #         )
        #     )
    file_size = os.path.getsize(model_path)
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f},FPS:{:.4f}, Size:{:.4f}".format(
        height, width, avg_psnr, avg_ms_ssim, avg_eval_fps, file_size/ (1024 * 1024)))
    

    video_path = Path(f"./Loadmodel/{savdir}/{args.data_name}/{args.num_points}/video")
    video_path.mkdir(parents=True, exist_ok=True)
    filename = "recovered_video.mp4"
    output_size = (width, height)
    video = cv2.VideoWriter(str(video_path / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    for img in tqdm(img_list, desc="Processing images", unit="image"):    
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(img_cv)
    video.release()
    print("video.mp4: MP4 video created successfully.")



if __name__ == "__main__":
    
    main(sys.argv[1:])


