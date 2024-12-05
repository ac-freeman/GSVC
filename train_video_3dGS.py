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
from sklearn.mixture import GaussianMixture
class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image,
        frame_num,
        savdir,
        loss_type,
        num_points: int = 2000,
        max_num_points: int = 2000,
        model_name:str = "GaussianSplatting3d",
        iterations:int = 30000,
        model_path = None,
        args = None,
        Trained_Model=None,
        isdensity=False,
        isremoval=True,
        removal_rate=0.25
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_to_tensor(image).to(self.device)
        self.frame_num=frame_num
        self.num_points = num_points
        self.max_num_points=max_num_points
        self.model_name=model_name
        self.data_name=args.data_name
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.densification_interval=args.densification_interval
        self.save_imgs = args.save_imgs
        self.save_everyimgs = args.save_everyimgs
        self.log_dir = Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
        self.isdensity=isdensity
        self.isremoval=isremoval
        self.loss_type = loss_type
        if model_name == "GaussianSplatting3d":
            from gaussiansplatting_3d import Gaussian3D
            self.gaussian_model = Gaussian3D(loss_type="Fusion2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, sh_degree=args.sh_degree, lr=args.lr).to(self.device)
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
    
    def print_stats(self,name, tensor):
        max_val = torch.max(tensor)  # Use PyTorch max
        mean_val = torch.mean(tensor)  # Use PyTorch mean
        median_val = torch.median(tensor)  # Use PyTorch median
        min_val = torch.min(tensor)  # Use PyTorch min
        
        print(f"{name} - Max: {max_val}, Mean: {mean_val}, Median: {median_val}, Min: {min_val}")

    def train(self,frame,ispos):     
        progress_bar = tqdm(range(1, int(self.iterations)+1), desc="Training progress")
        self.gaussian_model.train()
        start_time = time.time()
        early_stopping = EarlyStopping(patience=100, min_delta=1e-7)
        early_stopping_PSNR = EarlyStopping(patience=100, min_delta=1e-4)
        stabel_control=5000
        for iter in range(1, int(self.iterations)+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_image)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        num_gaussian_points =self.gaussian_model._xyz.size(0)
        psnr_value, ms_ssim_value,img = self.test(frame)
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100       
        Gmodel =self.gaussian_model.state_dict()
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time, Gmodel, img, num_gaussian_points, loss
    

    def test(self,frame):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
            out_image = out["render"]
        mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_image.float(), self.gt_image.float(), data_range=1, size_average=True).item()
        if (frame==0 or (frame+1)%100==0 ) and self.save_imgs:
            save_path_img = self.log_dir / "img"
            save_path_img.mkdir(parents=True, exist_ok=True)
            transform = transforms.ToPILImage()
            img = transform(out_image.float().squeeze(0))
            name =str(self.frame_num) + "_fitting.png" 
            img.save(str(save_path_img / name))
        else:
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
        "--model_name", type=str, default="GaussianSplatting3d", help="model selection: GaussianVideo, GaussianImage, 3DGS"
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
    parser.add_argument("--save_everyimgs", action="store_true", help="Save Every Images")
    parser.add_argument("--is_pos", action="store_true", help="Show the position of gaussians")
    parser.add_argument("--is_warmup",action="store_true", help="Warmup setup")
    parser.add_argument("--is_ad", action="store_true", help="Adaptive control of gaussians setup")
    parser.add_argument("--is_rm", action="store_true", help="Removal control of gaussians setup")
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
    args.fps=120
    width = args.width
    height = args.height
    removal_rate=args.removal_rate
    gmodel_save_path = Path(f"./checkpoints/{savdir_m}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
    gmodel_save_path.mkdir(parents=True, exist_ok=True)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    is_ad=args.is_ad
    is_rm=args.is_rm
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    psnrs, ms_ssims, training_times, eval_times, eval_fpses, gaussian_number = [], [], [], [], [],[]
    image_h, image_w = 0, 0
    video_frames = process_yuv_video(args.dataset, width, height)
    image_length,start=len(video_frames),0
    # image_length=5
    Gmodel=None
    img_list=[]
    gmodels_state_dict = {}
    num_gaussian_points_dict={}

    Gmodel=None
    for i in range(start, start+image_length):
        frame_num=i+1
        trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=args.num_points,max_num_points=args.num_points,
                iterations=args.iterations, model_name=args.model_name, args=args, model_path=None,Trained_Model=None,isdensity=False,isremoval=False,removal_rate=removal_rate)
        psnr, ms_ssim, training_time, eval_time, eval_fps, Gmodel, img, num_gaussian_points, loss = trainer.train(i,ispos)
        img_list.append(img)
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        gaussian_number.append(num_gaussian_points)
        image_h += trainer.H
        image_w += trainer.W
        gmodels_state_dict[f"frame_{frame_num}"] = Gmodel
        num_gaussian_points_dict[f"frame_{frame_num}"]=num_gaussian_points
        torch.cuda.empty_cache()
        if i==0 or (i+1)%1==0:
            logwriter.write("Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}, Loss:{:.4f}".format(frame_num, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps, loss))
    torch.save(gmodels_state_dict, gmodel_save_path / "gmodels_state_dict.pth")
    file_size = os.path.getsize(os.path.join(gmodel_save_path, 'gmodels_state_dict.pth'))
    with open(Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}") / "num_gaussian_points.txt", 'w') as f:
        for key, value in num_gaussian_points_dict.items():
            f.write(f'{key}: {value}\n')
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length
    gaussians = sum(gaussian_number) / len(gaussian_number)
    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}, Size:{:.4f}, Gaussian_number:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps, file_size/ (1024 * 1024),gaussians))
    if ispos:
        generate_video_density(savdir,img_list, args.data_name, args.model_name,args.fps,args.iterations,args.num_points,origin=False)    
    else:
        generate_video_density(savdir,img_list, args.data_name, args.model_name,args.fps,args.iterations,args.num_points,origin=True)  
    
if __name__ == "__main__":
    
    main(sys.argv[1:])


