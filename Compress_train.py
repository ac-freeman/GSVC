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
import copy
class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image,
        frame_num,
        savdir,
        loss_type,
        num_points: int = 2000,
        model_name:str = "GaussianVideo",
        iterations:int = 30000,
        model_path = None,
        args = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_to_tensor(image).to(self.device)
        self.frame_num=frame_num
        self.num_points = num_points
        self.model_name=model_name
        self.data_name=args.data_name
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.save_everyimgs = args.save_everyimgs
        self.log_dir = Path(f"./checkpoints_quant/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
        self.loss_type = loss_type
        if model_name == "GaussianVideo":
            from GaussianSplats_Compress_train import GaussianVideo_frame
            self.gaussian_model = GaussianVideo_frame(loss_type=self.loss_type, opt_type="adan", num_points=self.num_points,iterations=self.iterations, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
            device=self.device, lr=args.lr, quantize=True).to(self.device)
        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def train(self): 
        progress_bar = tqdm(range(1, int(self.iterations)+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        early_stopping = EarlyStopping(patience=100, min_delta=1e-7)
        for iter in range(1, int(self.iterations)+1):
            loss, psnr = self.gaussian_model.train_iter_quantize(self.gt_image,iter)
            if best_psnr < psnr:
                best_psnr = psnr
                best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
            if early_stopping(loss.item()):
                break
        end_time = time.time() - start_time
        progress_bar.close()
        self.gaussian_model.load_state_dict(best_model_dict)
        psnr_value, ms_ssim_value, bpp = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100  
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time, bpp, best_model_dict   
        
    
    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward_quantize()
        out_image = out["render"]
        mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_image.float(), self.gt_image.float(), data_range=1, size_average=True).item()
        m_bit, s_bit, r_bit, c_bit = out["unit_bit"]
        bpp = (m_bit + s_bit + r_bit + c_bit)/self.H/self.W
        return psnr, ms_ssim_value, bpp

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
        "--fps", type=int, default=120, help="number of frames per second (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianVideo", help="model selection: GaussianVideo, GaussianImage, 3DGS"
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
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument("--save_everyimgs", action="store_true", help="Save Every Images")
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
    args.fps=120
    width = args.width
    height = args.height
    gmodel_save_path = Path(f"./checkpoints_quant/{savdir_m}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
    gmodel_save_path.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./checkpoints_quant/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    psnrs, ms_ssims, training_times, eval_times, eval_fpses, bpps = [], [], [], [], [], []
    image_h, image_w = 0, 0
    video_frames = process_yuv_video(args.dataset, width, height)
    image_length,start=len(video_frames),0
    # image_length=5
    Gmodel=None
    gmodels_state_dict={}
    for i in range(start, start+image_length):
        frame_num=i+1
        trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=args.num_points,
                iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path)
        psnr, ms_ssim, training_time, eval_time, eval_fps, bpp, Gmodel = trainer.train()
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        bpps.append(bpp)
        image_h += trainer.H
        image_w += trainer.W
        gmodels_state_dict[f"frame_{frame_num}"] = Gmodel
        torch.cuda.empty_cache()
        if i==0 or (i+1)%1==0:
            logwriter.write("Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            frame_num, trainer.H, trainer.W, psnr, ms_ssim, bpp, training_time, eval_time, eval_fps))
    torch.save(gmodels_state_dict, gmodel_save_path / "gmodels_state_dict.pth")
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_bpp = torch.tensor(bpps).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length
    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Bpp:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_bpp, avg_training_time, avg_eval_time, avg_eval_fps))    

if __name__ == "__main__":
    
    main(sys.argv[1:])


