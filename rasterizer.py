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
        from gaussian2D import GaussianImage_Cholesky
        self.gaussian_model = GaussianImage_Cholesky(num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
        device=self.device).to(self.device)
        if Model is not None:
            checkpoint = Model
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    # def render(self):     
    #     psnr_value, ms_ssim_value,img = self.test()
    #     with torch.no_grad():
    #         self.gaussian_model.eval()
    #         test_start_time = time.time()
    #         for i in range(100):
    #             _ = self.gaussian_model()
    #         test_end_time = (time.time() - test_start_time)/100       
    #     return psnr_value, ms_ssim_value, 1/test_end_time, img
    
    # def test(self):
    #     self.gaussian_model.eval()
    #     with torch.no_grad():
    #         out = self.gaussian_model()
    #         out_image = out["render"]
    #     mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
    #     psnr = 10 * math.log10(1.0 / mse_loss.item())
    #     ms_ssim_value = ms_ssim(out_image.float(), self.gt_image.float(), data_range=1, size_average=True).item()
    #     transform = transforms.ToPILImage()
    #     img = transform(out_image.float().squeeze(0))
    #     return psnr, ms_ssim_value,img


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
        "--fps", type=int, default=120, help="number of frames per second (default: %(default)s)"
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

# def main(argv):
#     step=10
#     args = parse_args(argv)
#     savdir=args.savdir
#     fps=args.fps/step
#     width = args.width
#     height = args.height
#     model_path=args.model_path
#     num_points=args.num_points
#     device=torch.device("cuda:0")
#     if args.seed is not None:
#         torch.manual_seed(args.seed)
#         random.seed(args.seed)
#         torch.cuda.manual_seed(args.seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         np.random.seed(args.seed)
#     logwriter = LogWriter(Path(f"./Loadmodel/{savdir}/{args.data_name}/{args.num_points}"))
#     psnrs, ms_ssims, eval_fpses = [], [], []
#     image_h, image_w = 0, 0
#     video_frames = process_yuv_video(args.dataset, width, height)
#     image_length,start=len(video_frames),0
#     # image_length=5
#     img_list=[]
#     print(f"loading model path:{model_path}")
#     gmodels_state_dict = torch.load(model_path,map_location=device)
#     for i in tqdm(range(start, start + image_length), desc="Processing Frames"):
#         if i==0 or i%step==0:
#             frame_num=i+1
#             modelid=f"frame_{i + 1}"
#             Model = gmodels_state_dict[modelid]
#             Gaussianframe = LoadGaussians(num_points=num_points,image=video_frames[i], Model=Model,device=device,args=args)
#             psnr, ms_ssim,eval_fps, img = Gaussianframe.render()
#             img_list.append(img)
#             psnrs.append(psnr)
#             ms_ssims.append(ms_ssim)
#             eval_fpses.append(eval_fps)
#             torch.cuda.empty_cache()
#             # logwriter.write(
#             #         "Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, FPS:{:.4f}\n".format(
#             #             frame_num, Gaussianframe.H, Gaussianframe.W, psnr, ms_ssim, eval_fps
#             #         )
#             #     )
#     file_size = os.path.getsize(model_path)
#     avg_psnr = torch.tensor(psnrs).mean().item()
#     avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
#     avg_eval_fps = torch.tensor(eval_fpses).mean().item()
#     logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f},FPS:{:.4f}, Size:{:.4f}".format(
#         height, width, avg_psnr, avg_ms_ssim, avg_eval_fps, file_size/ (1024 * 1024)))
    

#     video_path = Path(f"./Loadmodel/{savdir}/{args.data_name}/{args.num_points}/video")
#     video_path.mkdir(parents=True, exist_ok=True)
#     filename = "video.mp4"
#     output_size = (width, height)
#     video = cv2.VideoWriter(str(video_path / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
#     for img in tqdm(img_list, desc="Processing images", unit="image"):    
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
#         img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#         video.write(img_cv)
#     video.release()
#     print("video.mp4: MP4 video created successfully.")

def main(argv):
    step=10
    args = parse_args(argv)
    savdir=args.savdir
    fps=5
    width = args.width
    height = args.height
    model_path=args.model_path
    num_points=args.num_points
    device=torch.device("cuda:0")
    video_frames = process_yuv_video(args.dataset, width, height)
    image_length,start=len(video_frames),0
    image_length=50
    img_list=[]
    print(f"loading model path:{model_path}")
    gmodels_state_dict = torch.load(model_path,map_location=device)
    for i in tqdm(range(start, start + image_length), desc="Processing Frames"):
        modelid=f"frame_{i + 1}"
        Model = gmodels_state_dict[modelid]
        Gaussianframe = LoadGaussians(num_points=num_points,image=video_frames[i], Model=Model,device=device,args=args)
        # img = Gaussianframe.render()
        # img_list.append(img)
        # torch.cuda.empty_cache()
        img_pos = Gaussianframe.render_pos()
        img = Gaussianframe.render()
        combined_img = Image.new('RGB', (img.width + img_pos.width, max(img.height, img_pos.height)))
        combined_img.paste(img, (0, 0))
        combined_img.paste(img_pos, (img.width, 0))
        img_list.append(combined_img)
        torch.cuda.empty_cache()

    video_path = Path(f"./Loadmodel/{savdir}/{args.data_name}/{args.num_points}/video")
    video_path.mkdir(parents=True, exist_ok=True)
    filename = "video.mp4"
    # output_size = (width, height)
    output_size = (combined_img.width, combined_img.height)
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


