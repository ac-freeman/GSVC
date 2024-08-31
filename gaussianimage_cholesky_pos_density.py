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

    def density_control(self):
        grad_xyz = self._xyz.grad
        if grad_xyz is None:
            raise RuntimeError("grad_xyz 为空，请检查 self._xyz 是否参与了计算图的构建。")
        grad_magnitude = torch.norm(grad_xyz, dim=1)
        gaussian_values = torch.exp(-0.5 * torch.sum(self.get_xyz ** 2 / torch.clamp(self.get_cholesky_elements[:, [0, 2]], min=1e-6), dim=1))
        high_gradient_threshold = 0.002
        high_gaussian_threshold = 0.075
        low_gaussian_threshold = 0.0005
        

        # Split large coordinate gradient & large gaussian values
        split_mask = (grad_magnitude > high_gradient_threshold) & (gaussian_values > high_gaussian_threshold)
        split_indices = torch.nonzero(split_mask).squeeze()

        # Clone large coordinate gradient & small gaussian values
        clone_mask = (grad_magnitude > high_gradient_threshold) & (gaussian_values < low_gaussian_threshold)
        clone_indices = torch.nonzero(clone_mask).squeeze()

        if split_indices.dim() > 0:
            num_split_indices = len(split_indices)
        else:
            num_split_indices = 0  # 或者设置为其他默认值

        if clone_indices.dim() > 0:
            num_clone_indices = len(clone_indices)
        else:
            num_clone_indices = 0  # 或者设置为其他默认值

        current_num_points = self._xyz.shape[0]
        potential_new_points = current_num_points + num_split_indices + num_clone_indices

        if potential_new_points > self.max_num_points:
            remaining_slots =self.max_num_points - current_num_points
            split_fraction = min(len(split_indices), remaining_slots // 2)
            clone_fraction = min(len(clone_indices), remaining_slots // 2)

            split_indices = split_indices[:split_fraction]
            clone_indices = clone_indices[:clone_fraction]

        # 执行 Split 操作
        if len(split_indices) > 0:
            self._xyz = torch.nn.Parameter(torch.cat([self._xyz, self._xyz[split_indices]], dim=0))
            self._cholesky = torch.nn.Parameter(torch.cat([self._cholesky, self._cholesky[split_indices] / 2], dim=0))
            self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, self._features_dc[split_indices]], dim=0))
            self._opacity = torch.cat([self._opacity, self._opacity[split_indices]], dim=0)

        # 执行 Clone 操作
        if len(clone_indices) > 0:
            self._xyz = torch.nn.Parameter(torch.cat([self._xyz, self._xyz[clone_indices]], dim=0))
            self._cholesky = torch.nn.Parameter(torch.cat([self._cholesky, self._cholesky[clone_indices]], dim=0))
            self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, self._features_dc[clone_indices]], dim=0))
            self._opacity = torch.cat([self._opacity, self._opacity[clone_indices]], dim=0)

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
            print(f"After split/clone: _cholesky size: {self._cholesky.size()}, _features_dc size: {self._features_dc.size()}")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        self.scheduler.step()
        return loss, psnr

    
