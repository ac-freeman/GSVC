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
from scipy.interpolate import CubicSpline, interp1d


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
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
            out_image = out["render"]
        transform = transforms.ToPILImage()
        img = transform(out_image.float().squeeze(0))
        return img
    
    def render_pos(self):  
        self.gaussian_model.eval()
        with torch.no_grad():
            out_pos =self.gaussian_model.forward_pos(self.num_points)
            out_pos_img = out_pos["render_pos"]
            transform = transforms.ToPILImage()
            img_pos = transform(out_pos_img.float().squeeze(0))
        return img_pos

    # def render_pos(self):  
    #     self.gaussian_model.eval()
    #     with torch.no_grad():
    #         out = self.gaussian_model()
    #         out_image = out["render"]
    #         out_pos =self.gaussian_model.forward_pos(self.num_points)
    #         out_pos_img = out_pos["render_pos"]

    #         transform = transforms.ToPILImage()
    #         img_pos = transform(out_pos_img.float().squeeze(0))
    #         img = transform(out_image.float().squeeze(0))
    #         combined_width =img.width+img_pos.width
    #         combined_height = max(img.height, img_pos.height)
    #         combined_img = Image.new("RGB", (combined_width, combined_height))
    #         combined_img.paste(img_pos, (0, 0))
    #         combined_img.paste(img, (img_pos.width, 0))
    #     return combined_img



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
    step=3
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
    video_frames = process_yuv_video(args.dataset, width, height)
    start=0
    img_list=[]
    gmodels_state_dict = torch.load(model_path,map_location=device)
    restored_gmodels_state_dict = {}
    
    num_frames=49

    all_xyz, all_cholesky, all_features_dc = [], [], []
    for i in range(1, num_frames-1):
        frame = gmodels_state_dict[f"frame_{i}"]
        all_xyz.append(frame['_xyz'].detach().cpu().numpy())
        all_cholesky.append(frame['_cholesky'].detach().cpu().numpy())
        all_features_dc.append(frame['_features_dc'].detach().cpu().numpy())
    
    # 将所有帧的数据堆叠为 numpy 数组
    all_xyz = np.stack(all_xyz)
    all_cholesky = np.stack(all_cholesky)
    all_features_dc = np.stack(all_features_dc)
    x = np.arange(len(gmodels_state_dict))

    # 使用CubicSpline或其他方法在所有帧上拟合轨迹
    spline_xyz = CubicSpline(x, all_xyz, axis=0)
    spline_cholesky = CubicSpline(x, all_cholesky, axis=0)
    spline_features_dc = CubicSpline(x, all_features_dc, axis=0)

    # 生成插值帧
    for i in range(num_frames - 1):
        for j in range(step):
            alpha = i + j / step  # 使用帧索引和步进位置插值
            interpolated_xyz = torch.tensor(spline_xyz(alpha), device=device)
            interpolated_cholesky = torch.tensor(spline_cholesky(alpha), device=device)
            interpolated_features_dc = torch.tensor(spline_features_dc(alpha), device=device)
            frame_index = i * step + j + 1
            restored_gmodels_state_dict[f"frame_{frame_index}"] = {
                '_xyz': interpolated_xyz,
                '_cholesky': interpolated_cholesky,
                '_features_dc': interpolated_features_dc
            }
    restored_gmodels_state_dict[f"frame_{frame_index}"] = gmodels_state_dict[f"frame_{len(gmodels_state_dict)}"]

    # for i in range(num_frames - 1):
    #     frame_id_start = f"frame_{i + 1}"
    #     frame_id_end = f"frame_{i + 2}"
    #     start_frame = gmodels_state_dict[frame_id_start]
    #     end_frame = gmodels_state_dict[frame_id_end]
    #     x = [0, step]
    #     xyz_y = torch.stack([start_frame['_xyz'].detach(), end_frame['_xyz'].detach()]).cpu().numpy()
    #     cholesky_y = torch.stack([start_frame['_cholesky'].detach(), end_frame['_cholesky'].detach()]).cpu().numpy()
    #     features_dc_y = torch.stack([start_frame['_features_dc'].detach(), end_frame['_features_dc'].detach()]).cpu().numpy()
    #     spline_xyz = CubicSpline(x, xyz_y, axis=0)
    #     spline_cholesky = CubicSpline(x, cholesky_y, axis=0)
    #     spline_features_dc = CubicSpline(x, features_dc_y, axis=0)
    #     for j in range(step):
    #         alpha = j 
    #         interpolated_xyz = torch.tensor(spline_xyz(alpha), device=start_frame['_xyz'].device)
    #         interpolated_cholesky = torch.tensor(spline_cholesky(alpha), device=start_frame['_cholesky'].device)
    #         interpolated_features_dc = torch.tensor(spline_features_dc(alpha), device=start_frame['_features_dc'].device)
    #         frame_index = i * step + j + 1
    #         restored_gmodels_state_dict[f"frame_{frame_index}"] = {
    #             '_xyz': interpolated_xyz,
    #             '_cholesky': interpolated_cholesky,
    #             '_features_dc': interpolated_features_dc
    #         }
    # restored_gmodels_state_dict[f"frame_{frame_index + 1}"] = gmodels_state_dict[f"frame_{num_frames}"]


    num_frames = len(restored_gmodels_state_dict)
    for i in tqdm(range(start, start + num_frames), desc="Processing Frames"):
        modelid=f"frame_{i + 1}"
        Model = restored_gmodels_state_dict[modelid]
        Gaussianframe = LoadGaussians(num_points=num_points,image=video_frames[i], Model=Model,device=device,args=args)
        img_pos = Gaussianframe.render_pos()
        img = Gaussianframe.render()
        combined_img = Image.new('RGB', (img.width + img_pos.width, max(img.height, img_pos.height)))
        combined_img.paste(img, (0, 0))
        combined_img.paste(img_pos, (img.width, 0))
        
        img_list.append(combined_img)
        torch.cuda.empty_cache()
    

    video_path = Path(f"./Loadmodel/{savdir}/{args.data_name}/{args.num_points}/video")
    video_path.mkdir(parents=True, exist_ok=True)
    filename = "recovered_video.mp4"
    # output_size = (width, height)
    output_size = (combined_img.width, combined_img.height)
    video = cv2.VideoWriter(str(video_path / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps*step, output_size)
    for img in tqdm(img_list, desc="Processing images", unit="image"):    
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(img_cv)
    video.release()
    print("video.mp4: MP4 video created successfully.")



if __name__ == "__main__":
    
    main(sys.argv[1:])


