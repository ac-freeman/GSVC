import math
import time
from pathlib import Path
import argparse
import yaml
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
class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image,
        frame_num,
        save_dir,
        loss_type,
        isclip=True,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        args = None,
        Trained_Model=None,
        isdensity=True,
        removal_rate=0.25
    ):
        self.device = torch.device("cuda:0")
        self.eimage = extend_image(image)
        self.gt_image = image_to_tensor(image).to(self.device)
        self.isclip = isclip
        self.gt_eimage = image_to_tensor(self.eimage).to(self.device)
        self.frame_num=frame_num
        self.num_points = num_points
        self.max_num_points=num_points
        self.model_name=model_name
        self.data_name=args.data_name
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        
        self.eH, self.eW = self.gt_eimage.shape[2], self.gt_eimage.shape[3]

        self.iterations = iterations
        self.densification_interval=args.densification_interval
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./checkpoints/{save_dir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
        self.isdensity=isdensity
        self.loss_type = loss_type
        if model_name == "GaussianImage_Cholesky":
            from gaussianimage_cholesky_Opacity import GaussianImage_Cholesky
            if self.isclip:
                self.gaussian_model = GaussianImage_Cholesky(loss_type=self.loss_type, opt_type="adan", num_points=self.num_points,max_num_points=self.max_num_points,densification_interval=self.densification_interval,iterations=self.iterations, H=self.eH, W=self.eW, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False,removal_rate=removal_rate).to(self.device)
            else:
                self.gaussian_model = GaussianImage_Cholesky(loss_type=self.loss_type, opt_type="adan", num_points=self.num_points,max_num_points=self.max_num_points,densification_interval=self.densification_interval,iterations=self.iterations, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False,removal_rate=removal_rate).to(self.device)

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
        psnr_list, iter_list, img_list=[], [], []
        progress_bar = tqdm(range(1, int(self.iterations)+1), desc="Training progress")
        self.gaussian_model.train()
        start_time = time.time()
        save_path_img = self.log_dir / "img"
        save_path_img.mkdir(parents=True, exist_ok=True)
        early_stopping = EarlyStopping(patience=100, min_delta=1e-7)
        density_control=6000
        for iter in range(1, int(self.iterations)+1):
            start_adaptivecontrol=False
            if self.isclip:
                loss, psnr,img = self.gaussian_model.train_iter_img_Opacity(self.gt_eimage,iter,start_adaptivecontrol)
            else:
                loss, psnr,img = self.gaussian_model.train_iter_img_Opacity(self.gt_image,iter,start_adaptivecontrol)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
                if iter==1 or iter % 50 == 0:
                    num_gaussian_points =self.gaussian_model._xyz.size(0)
                    out_pos_sca =self.gaussian_model.forward_pos_sca(num_gaussian_points)
                    transform = transforms.ToPILImage()
                    img = transform(img.float().squeeze(0))
                    img_pos_sca = transform(out_pos_sca["render_pos_sca"].float().squeeze(0))
                    combined_width =img.width+img_pos_sca.width
                    combined_height = max(img.height, img_pos_sca.height)
                    combined_img = Image.new("RGB", (combined_width, combined_height))
                    combined_img.paste(img_pos_sca, (0, 0))
                    combined_img.paste(img, (img_pos_sca.width, 0))
                    img_list.append(combined_img)
            if self.isdensity:
                if early_stopping(loss.item()):
                    start_adaptivecontrol=True
            elif early_stopping(loss.item()):
                print(f"Early stopping at iteration {iter}")
                break
            if start_adaptivecontrol:
                density_control=density_control-1
                if density_control<0 and early_stopping(loss.item()):
                    print(f"After adaptive control: Early stopping at iteration {iter}")
                    break

        end_time = time.time() - start_time
        progress_bar.close()
        num_gaussian_points =self.gaussian_model._xyz.size(0)
        self.test(num_gaussian_points)
        Gmodel =self.gaussian_model.state_dict()
        filtered_Gmodel = {
            k: v for k, v in Gmodel.items()
            # if k in ['_xyz', '_cholesky', '_features_dc','_opacity']
            if k in ['_xyz', '_cholesky', '_features_dc','_opacity']
        }
        return filtered_Gmodel,img_list,num_gaussian_points
    
    def test(self,num_gaussian_points):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
            out_pos_sca =self.gaussian_model.forward_pos_sca(num_gaussian_points)
            if self.isclip:
                out_image = restor_image(out["render"],self.H,self.W)
                out_pos_sca_img = restor_image(out_pos_sca["render_pos_sca"],self.H,self.W)
            else:
                out_image = out["render"]
                out_pos_sca_img = out_pos_sca["render_pos_sca"]            
        save_path_img = self.log_dir / "img"
        save_path_img.mkdir(parents=True, exist_ok=True)
        # 转换为PIL图像
        transform = transforms.ToPILImage()
        img = transform(out_image.float().squeeze(0))
        img_pos_sca = transform(out_pos_sca_img.float().squeeze(0))
        combined_width =img.width+img_pos_sca.width
        combined_height = max(img.height, img_pos_sca.height)
        combined_img = Image.new("RGB", (combined_width, combined_height))
        combined_img.paste(img_pos_sca, (0, 0))
        combined_img.paste(img, (img_pos_sca.width, 0))
        # 保存拼接后的图片
        combined_name = str(self.frame_num) + "_fitting_combined_pos.png"
        combined_img.save(str(save_path_img / combined_name))

        
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
        "--iterations", type=int, default=30000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--densification_interval",type=int,default=100,help="densification_interval (default: %(default)s)"
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
        default=4000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="width (default: %(default)s)"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="height (default: %(default)s)"
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--loss_type", type=str, default=None, help="Type of Loss")
    parser.add_argument("--savdir", type=str, default="result", help="Path to results")
    parser.add_argument("--savdir_m", type=str, default="models", help="Path to models")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--removal_rate", type=float, default=0.1, help="Removal rate")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument("--is_pos", action="store_true", help="Show the position of gaussians")
    parser.add_argument("--is_warmup",action="store_true", help="Warmup setup")
    parser.add_argument("--is_ad", action="store_true", help="Adaptive control of gaussians setup")
    parser.add_argument("--is_clip", action="store_true", help="Adaptive control of Clip")
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
    args.save_imgs=True
    loss_type=args.loss_type
    savdir=args.savdir
    savdir_m=args.savdir_m
    ispos = args.is_pos
    iswarmup=args.is_warmup
    is_ad=args.is_ad
    isclip = args.is_clip
    args.fps=120
    width = args.width
    height = args.height
    removal_rate=args.removal_rate
    gmodel_save_path = Path(f"./checkpoints/{savdir_m}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
    gmodel_save_path.mkdir(parents=True, exist_ok=True)  # 确保保存目录存在
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))

    image_h, image_w = 0, 0
    video_frames = process_yuv_video(args.dataset, width, height)
    image_length,start=len(video_frames),0
    image_length=1
    Gmodel=None
    gmodels_state_dict = {}
    num_gaussian_points_dict={}
    for i in range(start, start+image_length):
        frame_num=i+1
        if frame_num ==1 or frame_num%50==0:
            if iswarmup:
                trainer = SimpleTrainer2d(image=downsample_image(video_frames[i],4),frame_num=frame_num,save_dir=savdir,loss_type=loss_type, num_points=args.num_points, 
                    iterations=1000, model_name=args.model_name, args=args, model_path=None,Trained_Model=None,isdensity=False,removal_rate=removal_rate,isclip=isclip)
                _, _, _, _, _, Gmodel,_,num_gaussian_points = trainer.train(i,ispos)
                trainer = SimpleTrainer2d(image=downsample_image(video_frames[i],2),frame_num=frame_num,save_dir=savdir,loss_type=loss_type, num_points=num_gaussian_points, 
                    iterations=1000, model_name=args.model_name, args=args, model_path=None,Trained_Model=Gmodel,isdensity=False,removal_rate=removal_rate,isclip=isclip)
                _, _, _, _, _, Gmodel, _, num_gaussian_points= trainer.train(i,ispos)
                trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,save_dir=savdir,loss_type=loss_type, num_points=num_gaussian_points, 
                    iterations=args.iterations, model_name=args.model_name, args=args, model_path=None,Trained_Model=Gmodel,isdensity=is_ad,removal_rate=removal_rate,isclip=isclip)
            else:
                trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,save_dir=savdir,loss_type=loss_type, num_points=args.num_points,
                    iterations=args.iterations, model_name=args.model_name, args=args, model_path=None,Trained_Model=None,isdensity=is_ad,removal_rate=removal_rate,isclip=isclip)
        Gmodel,img_list,num_gaussian_points = trainer.train(i,ispos)

        image_h += trainer.H
        image_w += trainer.W
        gmodels_state_dict[f"frame_{frame_num}"] = Gmodel
        num_gaussian_points_dict[f"frame_{frame_num}"]=num_gaussian_points
        torch.cuda.empty_cache()
        # with open(Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}") / "loss_list.txt", 'w') as f:
        #     for loss in loss_list:
        #         f.write(f"{loss}\n")
        
    torch.save(gmodels_state_dict, gmodel_save_path / "gmodels_state_dict.pth")

    with open(Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}") / "num_gaussian_points.txt", 'w') as f:
        for key, value in num_gaussian_points_dict.items():
            f.write(f'{key}: {value}\n')

    if ispos:
        generate_video_test(savdir,img_list, args.data_name, args.model_name,args.fps,args.iterations,args.num_points,origin=False)    
    else:
        generate_video_test(savdir,img_list, args.data_name, args.model_name,args.fps,args.iterations,args.num_points,origin=True)  
    
if __name__ == "__main__":
    
    main(sys.argv[1:])


