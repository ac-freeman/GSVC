from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan
from torch.distributions import MultivariateNormal

class GaussianImage_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.max_num_points = kwargs["max_num_points"]
        self.densification_interval=kwargs["densification_interval"]
        self.iterations=kwargs["iterations"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]
        self.removal_rate=kwargs["removal_rate"]
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        # self._opacity = nn.Parameter(torch.ones(self.init_num_points, 1))
        self._opacity = nn.Parameter(0.01 * torch.ones(self.init_num_points, 1))
        #self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.lr = kwargs["lr"]
        self.opt_type =kwargs["opt_type"]
        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    def forward_pos_sca(self,num_points):
        features_dc = torch.ones(num_points, 3).to(self.device)
        cholesky = torch.full((num_points, 3), 1.0).to(self.device)
        _opacity = torch.ones(num_points, 1).to(self.device)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, cholesky+self.cholesky_bound, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                features_dc, _opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render_pos_sca": out_img}
    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def update_optimizer(self):
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            self.optimizer = Adan(self.parameters(), lr=self.lr)
        # 重新初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def density_control(self, iter):
        iter_threshold_remove = 4000  # 根据训练计划调整这个阈值
        if iter > iter_threshold_remove:
            return
        grad_xyz = self._xyz.grad
        if grad_xyz is None:
            raise RuntimeError("grad_xyz is None,请检查 self._xyz 是否参与了计算图。")
        # 计算每个点的梯度幅值
        # grad_magnitude = torch.norm(grad_xyz, dim=1)
        grad_magnitude =torch.norm(grad_xyz, dim=1)

        # 对梯度幅值进行升序排序（最小的梯度在前）
        _, sorted_indices = torch.sort(grad_magnitude)
        removal_rate_per_step = self.removal_rate/int(iter_threshold_remove/(self.densification_interval))
        if iter < iter_threshold_remove:
            # 训练早期：只执行删除操作，减少总的高斯点数量
            remove_count = int(removal_rate_per_step * self.max_num_points)
            
            remove_indices = sorted_indices[:remove_count]

            # 删除选定的点
            keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
            keep_indices[remove_indices] = False

            self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
            self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
            self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
            self._opacity = torch.nn.Parameter(self._opacity[keep_indices])
        elif iter == iter_threshold_remove:
            # 训练早期：只执行删除操作，减少总的高斯点数量
            remove_count = self._xyz.shape[0]-int(self.max_num_points * (1-self.removal_rate))
            if remove_count>0:
                remove_indices = sorted_indices[:remove_count]

                # 删除选定的点
                keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
                keep_indices[remove_indices] = False

                self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
                self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
                self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
                self._opacity = torch.nn.Parameter(self._opacity[keep_indices])
        # 更新优化器中的参数
        self.update_optimizer()

    def density_control_Opacity(self, iter,strat_iter_adaptive_control):
        iter_threshold_remove = 4000  # 根据训练计划调整这个阈值
        opacity = self._opacity
        grad_magnitude =torch.norm(opacity, dim=1)
        _, sorted_indices = torch.sort(grad_magnitude)
        removal_rate_per_step = self.removal_rate/int(iter_threshold_remove/(self.densification_interval))
        if iter < strat_iter_adaptive_control+iter_threshold_remove:
            print("adaptive")
            remove_count = int(removal_rate_per_step * self.max_num_points)
            
            remove_indices = sorted_indices[:remove_count]

            
            keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
            keep_indices[remove_indices] = False

            self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
            self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
            self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
            self._opacity = torch.nn.Parameter(self._opacity[keep_indices])
            # # 更新优化器中的参数
            # if iter%3000==0:
            #     self._opacity = torch.nn.Parameter(0.01 * torch.ones_like(self._opacity))
        elif iter == strat_iter_adaptive_control+iter_threshold_remove:
            # 训练早期：只执行删除操作，减少总的高斯点数量
            remove_count = self._xyz.shape[0]-int(self.max_num_points * (1-self.removal_rate))
            #print(remove_count,self._xyz.shape[0])
            if remove_count>0:
                remove_indices = sorted_indices[:remove_count]

                # 删除选定的点
                keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
                keep_indices[remove_indices] = False

                self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
                self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
                self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
                self._opacity = torch.nn.Parameter(self._opacity[keep_indices])  
                #print(self._xyz.shape[0]) 
        
        self.update_optimizer()

    # def density_control_Opacity_info(self, iter):
    #     begin_iter = 10000
    #     iter_threshold_remove = 20000  # 根据训练计划调整这个阈值
    #     if iter > iter_threshold_remove or iter< begin_iter:
    #         return
    #     opacity = self._opacity
    #     grad_magnitude = torch.norm(opacity, dim=1)
    #     _, sorted_indices = torch.sort(grad_magnitude)
    #     removal_rate_per_step = self.removal_rate / int((iter_threshold_remove-begin_iter) / (self.densification_interval))
        
    #     if iter>= begin_iter and iter < iter_threshold_remove:
    #         remove_count = int(removal_rate_per_step * self.max_num_points)
    #         remove_indices = sorted_indices[:remove_count]
            
    #         # 计算被移除点的 opacity 的统计信息
    #         removed_opacity = self._opacity[remove_indices]
    #         max_opacity = torch.max(removed_opacity).item()
    #         min_opacity = torch.min(removed_opacity).item()
    #         mean_opacity = torch.mean(removed_opacity).item()
    #         median_opacity = torch.median(removed_opacity).item()
            
            
    #         keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
    #         keep_indices[remove_indices] = False

    #         self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
    #         self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
    #         self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
    #         self._opacity = torch.nn.Parameter(self._opacity[keep_indices])
            
    #         # # 更新优化器中的参数
    #         # if iter % 3000 == 0:
    #         #     self._opacity = torch.nn.Parameter(0.01 * torch.ones_like(self._opacity))
    
    #     elif iter == iter_threshold_remove:
    #         # 训练早期：只执行删除操作，减少总的高斯点数量
    #         remove_count = self._xyz.shape[0] - int(self.max_num_points * (1 - self.removal_rate))
    #         if remove_count > 0:
    #             remove_indices = sorted_indices[:remove_count]
                
    #             # 计算被移除点的 opacity 的统计信息
    #             removed_opacity = self._opacity[remove_indices]
    #             max_opacity = torch.max(removed_opacity).item()
    #             min_opacity = torch.min(removed_opacity).item()
    #             mean_opacity = torch.mean(removed_opacity).item()
    #             median_opacity = torch.median(removed_opacity).item()

    #             # 删除选定的点
    #             keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
    #             keep_indices[remove_indices] = False

    #             self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
    #             self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
    #             self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
    #             self._opacity = torch.nn.Parameter(self._opacity[keep_indices])
            
    #     self.update_optimizer()
    #     return max_opacity,min_opacity,mean_opacity,median_opacity



    def train_iter_Opacity(self, gt_image,iter,isdensity):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        if (iter) % (self.densification_interval) == 0 and iter > 0 and isdensity:
            self.density_control_Opacity(iter)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        
        self.scheduler.step()
        return loss, psnr
    
    def train_iter_img_Opacity(self, gt_image,iter,isdensity,strat_iter_adaptive_control):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        if (iter) % (self.densification_interval) == 0 and iter > 0 and isdensity:
            self.density_control_Opacity(iter,strat_iter_adaptive_control)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        
        self.scheduler.step()
        return loss, psnr,image
    
    def train_iter(self, gt_image,iter,isdensity):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        if (iter) % (self.densification_interval) == 0 and iter > 0 and isdensity:
            self.density_control(iter)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        
        self.scheduler.step()
        return loss, psnr
    
    def train_iter_img(self, gt_image,iter,isdensity):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        if (iter) % (self.densification_interval) == 0 and iter > 0 and isdensity:
            self.density_control(iter) 
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        
        self.scheduler.step()
        return loss, psnr,image
    

    
