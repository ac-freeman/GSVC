import math
import time
from pathlib import Path
import argparse
import re
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
savdir="fps"
savdir_m="fps"
mpath="/home/e/e1344641/GaussianVideo/checkpoints/models/Beauty/GaussianImage_Cholesky_30000_50000/gmodels_state_dict.pth"
class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image,
        frame_num,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        args = None,
        Trained_Model=None,
        isdensity=True
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_to_tensor(image).to(self.device)
        self.frame_num=frame_num
        self.num_points = num_points
        self.max_num_points=num_points*2
        self.model_name=model_name
        self.data_name=args.data_name
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.densification_interval=args.densification_interval
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
        self.isdensity=isdensity
        if model_name == "GaussianImage_Cholesky":
            from gaussianimage_cholesky import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points,max_num_points=self.max_num_points,densification_interval=self.densification_interval, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)
        if Trained_Model is not None:
            checkpoint = Trained_Model
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)
    def train(self,frame,ispos):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, int(self.iterations)+1), desc="Training progress")
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, int(self.iterations)+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_image,iter,self.isdensity)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        num_gaussian_points =self.gaussian_model._xyz.size(0)
        psnr_value, ms_ssim_value,img = self.test(frame,num_gaussian_points,ispos)
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100       
        Gmodel =self.gaussian_model.state_dict()
        filtered_Gmodel = {
            k: v for k, v in Gmodel.items()
            if k in ['_xyz', '_cholesky', '_features_dc']
        }
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time, filtered_Gmodel,img,num_gaussian_points
    def test(self,frame,num_gaussian_points,ispos):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
            #out_pos =self.gaussian_model.forward_pos(num_gaussian_points)
            if ispos:
                out_pos_sca =self.gaussian_model.forward_pos_sca(num_gaussian_points)
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        if ispos:
            if (frame==0 or (frame+1)%100==0 ) and self.save_imgs:
                # save_path_img = self.log_dir / "img"
                # save_path_img.mkdir(parents=True, exist_ok=True)
                # transform = transforms.ToPILImage()
                # img_pos = transform(out["render_pos"].float().squeeze(0))
                # img = transform(out["render"].float().squeeze(0))
                # name_pos =str(self.frame_num) + "_fitting_pos.png"
                # name =str(self.frame_num) + "_fitting.png"  
                # img_pos.save(str(save_path_img / name_pos))
                # img.save(str(save_path_img / name))

                save_path_img = self.log_dir / "img"
                save_path_img.mkdir(parents=True, exist_ok=True)
                # 转换为PIL图像
                transform = transforms.ToPILImage()
                img = transform(out["render"].float().squeeze(0))
                img_pos_sca = transform(out_pos_sca["render_pos_sca"].float().squeeze(0))
                #img_pos = transform(out_pos["render_pos"].float().squeeze(0))
                
                # 拼接图片
                # combined_width = img_pos.width + img.width+img_pos_sca.width
                # combined_height = max(img_pos.height, img.height, img_pos_sca.height)
                # combined_img = Image.new("RGB", (combined_width, combined_height))
                # combined_img.paste(img_pos_sca, (0, 0))
                # combined_img.paste(img_pos, (img_pos_sca.width, 0))
                # combined_img.paste(img, (img_pos.width + img_pos_sca.width, 0))


                combined_width =img.width+img_pos_sca.width
                combined_height = max(img.height, img_pos_sca.height)
                combined_img = Image.new("RGB", (combined_width, combined_height))
                combined_img.paste(img_pos_sca, (0, 0))
                combined_img.paste(img, (img_pos_sca.width, 0))

                # 保存拼接后的图片
                combined_name = str(self.frame_num) + "_fitting_combined_pos.png"
                combined_img.save(str(save_path_img / combined_name))
            else:
                transform = transforms.ToPILImage()
                img_pos_sca = transform(out_pos_sca["render_pos_sca"].float().squeeze(0))
                #img_pos = transform(out_pos["render_pos"].float().squeeze(0))
                img = transform(out["render"].float().squeeze(0))
                # combined_width = img_pos.width + img.width+img_pos_sca.width
                # combined_height = max(img_pos.height, img.height, img_pos_sca.height)
                # combined_img = Image.new("RGB", (combined_width, combined_height))
                # combined_img.paste(img_pos_sca, (0, 0))
                # combined_img.paste(img_pos, (img_pos_sca.width, 0))
                # combined_img.paste(img, (img_pos.width + img_pos_sca.width, 0))
                combined_width =img.width+img_pos_sca.width
                combined_height = max(img.height, img_pos_sca.height)
                combined_img = Image.new("RGB", (combined_width, combined_height))
                combined_img.paste(img_pos_sca, (0, 0))
                combined_img.paste(img, (img_pos_sca.width, 0))
            return psnr, ms_ssim_value,combined_img
        if (frame==0 or (frame+1)%100==0 ) and self.save_imgs:
            save_path_img = self.log_dir / "img"
            save_path_img.mkdir(parents=True, exist_ok=True)
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name =str(self.frame_num) + "_fitting.png" 
            img.save(str(save_path_img / name))
        else:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
        return psnr, ms_ssim_value,img

def image_to_tensor(img: Image.Image):
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='Beauty', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=5000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--densification_interval",type=int,default=2500,help="densification_interval (default: %(default)s)"
    )
    parser.add_argument(
        "--fps", type=int, default=120, help="number of frames per second (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    width = 1920
    height = 1080
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
    
    fps_list = []
    
    # 创建保存路径
    save_path = Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 读取视频帧
    video_frames = process_yuv_video(args.dataset, width, height)
    
    # 加载模型
    gmodels_state_dict = torch.load(mpath)    
    
    # 逐帧处理
    for frame_num_str, Gmodel in tqdm(gmodels_state_dict.items(), desc="Processing frames"):
        frame_num = int(re.search(r'\d+', frame_num_str).group())
        
        model = SimpleTrainer2d(image=video_frames[frame_num-1], frame_num=frame_num, num_points=args.num_points, 
                                iterations=args.iterations, model_name=args.model_name, args=args, 
                                model_path=None, Trained_Model=Gmodel, isdensity=False)
        
        with torch.no_grad():
            model.gaussian_model.eval()
            test_start_time = time.time()
            
            # 模拟执行100次，计算FPS
            for i in range(100):
                _ = model.gaussian_model()
            
            fps = 1 / ((time.time() - test_start_time) / 100)
            fps_list.append((frame_num, fps))
    
    # 将FPS结果保存到txt文件中
    fps_file_path = save_path / "fps_results.txt"
    with open(fps_file_path, 'w') as f:
        for frame_num, fps in fps_list:
            f.write(f"Frame_{frame_num}: FPS: {fps:.4f}\n")
    
    print(f"FPS results saved to {fps_file_path}")
    
if __name__ == "__main__":
    
    main(sys.argv[1:])


