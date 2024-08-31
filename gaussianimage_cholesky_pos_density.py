from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from filed.quantize import *
from optimizer import Adan

class GaussianImage_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.max_num_points = kwargs["max_num_points"]
        self.densification_interval=kwargs["densification_interval"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]

        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))

        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply 
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.cholesky_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3)

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

    def forward_pos_sca(self):
        self._features_dc = nn.Parameter(torch.ones(self.init_num_points, 3).to(self.device))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3).to(self.device))
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render_pos_sca": out_img}
    def forward_pos(self):
        self._features_dc = nn.Parameter(torch.ones(self.init_num_points, 3).to(self.device))
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render_pos": out_img}
    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def update_optimizer(self):
        # 取得当前的学习率和调度器状态
        current_lr = self.optimizer.param_groups[0]['lr']
        step_size = self.scheduler.step_size
        gamma = self.scheduler.gamma
        last_epoch = self.scheduler.last_epoch

        # 重新初始化优化器
        if isinstance(self.optimizer, Adan):
            self.optimizer = Adan(self.parameters(), lr=current_lr)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=current_lr)

        # 重新初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)



    def density_control(self):
        grad_xyz = self._xyz.grad
        if grad_xyz is None:
            raise RuntimeError("grad_xyz 为空，请检查 self._xyz 是否参与了计算图的构建。")
        
        grad_magnitude = torch.norm(grad_xyz, dim=1)

        # 计算前10%的数量
        percentile_10_count = int(0.1 * len(grad_magnitude))

        # 对grad_magnitude排序并获取前10%的索引
        sorted_grad_magnitude, sorted_indices = torch.sort(grad_magnitude)
        top_10_percent_indices = sorted_indices[:percentile_10_count]

        # 获取这些点的gaussian_values
        gaussian_values = torch.exp(-0.5 * torch.sum(self.get_xyz ** 2 / torch.clamp(self.get_cholesky_elements[:, [0, 2]], min=1e-6), dim=1))
        top_gaussian_values = gaussian_values[top_10_percent_indices]

        # 计算gaussian_values的中位数
        gaussian_threshold = torch.median(gaussian_values)

        # 在前10%的点中进行分类
        split_indices = top_10_percent_indices[top_gaussian_values > gaussian_threshold]
        clone_indices = top_10_percent_indices[top_gaussian_values <= gaussian_threshold]

        # 执行 Split 和 Clone 操作
        original_num_points = self._xyz.shape[0]
        potential_new_points = original_num_points + len(split_indices) + len(clone_indices)
        print(f"percentile_10_count: {percentile_10_count}, split_indices: {len(split_indices)}, clone_indices: {len(clone_indices)}")

        if potential_new_points > self.max_num_points:
            remaining_slots =self.max_num_points - original_num_points
            split_fraction = min(len(split_indices), remaining_slots // 2)
            clone_fraction = min(len(clone_indices), remaining_slots // 2)

            split_indices = split_indices[:split_fraction]
            clone_indices = clone_indices[:clone_fraction]
        
        if len(split_indices) > 0:
            self._xyz.data = torch.cat([self._xyz.data, self._xyz.data[split_indices]], dim=0)
            self._cholesky.data = torch.cat([self._cholesky.data, self._cholesky.data[split_indices] / 2], dim=0)
            self._features_dc.data = torch.cat([self._features_dc.data, self._features_dc.data[split_indices]], dim=0)
            self._opacity = torch.cat([self._opacity, self._opacity[split_indices]], dim=0)

        # 执行 Clone 操作
        if len(clone_indices) > 0:
            self._xyz.data = torch.cat([self._xyz.data, self._xyz.data[clone_indices]], dim=0)
            self._cholesky.data = torch.cat([self._cholesky.data, self._cholesky.data[clone_indices]], dim=0)
            self._features_dc.data = torch.cat([self._features_dc.data, self._features_dc.data[clone_indices]], dim=0)
            self._opacity = torch.cat([self._opacity, self._opacity[clone_indices]], dim=0)
        
        self.update_optimizer()
        print(f"After split/clone: _cholesky size: {self._cholesky.size()}, _features_dc size: {self._features_dc.size()}")

    def train_iter(self, gt_image,iter,isdensity):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        if (iter+1) % (self.densification_interval) == 0 and iter > 0 and isdensity:
            self.density_control()
            
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        
        self.scheduler.step()
        return loss, psnr

    
