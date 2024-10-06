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
        self._opacity = nn.Parameter(torch.ones(self.init_num_points, 1))
        # self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
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

    # def forward_pos_sca(self,num_points):
    #     self._features_dc = nn.Parameter(torch.ones(num_points, 3).to(self.device))
    #     self._cholesky = nn.Parameter(torch.rand(num_points, 3).to(self.device))
    #     self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
    #     out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
    #             self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
    #     out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
    #     out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
    #     return {"render_pos_sca": out_img}
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
    def forward_pos(self,num_points):
        features_dc = torch.ones(num_points, 3).to(self.device)
        _opacity = torch.ones(num_points, 1).to(self.device)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                features_dc, _opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
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
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            self.optimizer = Adan(self.parameters(), lr=self.lr)
        # 重新初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    # def density_control(self):
    #     grad_xyz = self._xyz.grad
    #     if grad_xyz is None:
    #         raise RuntimeError("grad_xyz 为空，请检查 self._xyz 是否参与了计算图的构建。")
        
    #     grad_magnitude = torch.norm(grad_xyz, dim=1)

        
    #     percentile_count = int(0.05 * len(grad_magnitude))

        
    #     sorted_grad_magnitude, sorted_indices = torch.sort(grad_magnitude, descending=True)
    #     top_percent_indices = sorted_indices[:percentile_count]
        

        
    #     gaussian_values = torch.exp(-0.5 * torch.sum(self.get_xyz ** 2 / torch.clamp(self.get_cholesky_elements[:, [0, 2]], min=1e-6), dim=1))
    #     top_gaussian_values = gaussian_values[top_percent_indices]

        
    #     gaussian_threshold = torch.median(top_gaussian_values)

        
    #     split_indices = top_percent_indices[top_gaussian_values > gaussian_threshold]
    #     clone_indices = top_percent_indices[top_gaussian_values <= gaussian_threshold]

        
    #     current_num_points = self._xyz.shape[0]
    #     potential_new_points = current_num_points + len(split_indices) + len(clone_indices)

    #     if potential_new_points > self.max_num_points:
    #         remaining_slots =self.max_num_points - current_num_points
    #         split_fraction = min(len(split_indices), remaining_slots // 2)
    #         clone_fraction = min(len(clone_indices), remaining_slots // 2)

    #         split_indices = split_indices[:split_fraction]
    #         clone_indices = clone_indices[:clone_fraction]
        
    #     potential_new_points = current_num_points + len(split_indices) + len(clone_indices)
    #     print(f"increase_point: {potential_new_points}, split_indices: {len(split_indices)}, clone_indices: {len(clone_indices)}")

    #     print(f"decrease_point: {potential_new_points}")

    #     if potential_new_points>0:
    #         remove_indices = sorted_indices[-potential_new_points:]
    #         self._xyz = torch.nn.Parameter(torch.cat([self._xyz, self._xyz[~remove_indices]], dim=0))
    #         self._cholesky = torch.nn.Parameter(torch.cat([self._cholesky, self._cholesky[~remove_indices]], dim=0))
    #         self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, self._features_dc[~remove_indices]], dim=0))
    #         self._opacity = torch.cat([self._opacity, self._opacity[~remove_indices]], dim=0)

    #     # 执行 Split 操作
    #     if len(split_indices) > 0:
    #         self._xyz = torch.nn.Parameter(torch.cat([self._xyz, self._xyz[split_indices]], dim=0))
    #         self._cholesky = torch.nn.Parameter(torch.cat([self._cholesky / 2, self._cholesky[split_indices] / 2], dim=0))
    #         self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, self._features_dc[split_indices]], dim=0))
    #         self._opacity = torch.cat([self._opacity, self._opacity[split_indices]], dim=0)

    #     # 执行 Clone 操作
    #     if len(clone_indices) > 0:
    #         self._xyz = torch.nn.Parameter(torch.cat([self._xyz, self._xyz[clone_indices]], dim=0))
    #         self._cholesky = torch.nn.Parameter(torch.cat([self._cholesky, self._cholesky[clone_indices]], dim=0))
    #         self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, self._features_dc[clone_indices]], dim=0))
    #         self._opacity = torch.cat([self._opacity, self._opacity[clone_indices]], dim=0)

    #     self.update_optimizer()
        
        
    #     #print(f"After split/clone: _cholesky size: {self._cholesky.size()}, _features_dc size: {self._features_dc.size()}")    

    # def density_control(self):
    #     grad_xyz = self._xyz.grad
    #     if grad_xyz is None:
    #         raise RuntimeError("grad_xyz is None, please check if self._xyz is involved in the computational graph.")
        
    #     # Compute the gradient magnitude for each point
    #     grad_magnitude = torch.norm(grad_xyz, dim=1)

    #     # Calculate the number of top 1% points
    #     percentile_count = int(0.005 * len(grad_magnitude))

    #     # Sort the gradient magnitudes in descending order to get the indices of largest values
    #     sorted_grad_magnitude, sorted_indices = torch.sort(grad_magnitude, descending=True)  # Sorting in descending order
    #     top_percent_indices = sorted_indices[:percentile_count]  # Select top 5% of points based on gradient

    #     # Compute Gaussian values for all points
    #     gaussian_values = torch.exp(-0.5 * torch.sum(self.get_xyz ** 2 / torch.clamp(self.get_cholesky_elements[:, [0, 2]], min=1e-6), dim=1))
    #     top_gaussian_values = gaussian_values[top_percent_indices]

    #     # Use the median of the top 1% Gaussian values as a threshold
    #     gaussian_threshold = torch.median(top_gaussian_values)

    #     # Select points for split and clone based on the Gaussian threshold
    #     split_indices = top_percent_indices[top_gaussian_values > gaussian_threshold]
    #     clone_indices = top_percent_indices[top_gaussian_values <= gaussian_threshold]

    #     current_num_points = self._xyz.shape[0]
    #     potential_new_points = current_num_points + len(split_indices) + len(clone_indices)

    #     # Ensure that the total number of points does not exceed the maximum allowed points
    #     if potential_new_points > self.max_num_points:
    #         remaining_slots = self.max_num_points - current_num_points
    #         split_fraction = min(len(split_indices), remaining_slots // 2)
    #         clone_fraction = min(len(clone_indices), remaining_slots // 2)

    #         split_indices = split_indices[:split_fraction]
    #         clone_indices = clone_indices[:clone_fraction]

    #     # Remove the points with the smallest gradients to keep the total number of points constant
    #     points_to_remove =  len(split_indices) + len(clone_indices)

    #     # Perform the Split operation
    #     if len(split_indices) > 0:
    #         self._xyz = torch.nn.Parameter(torch.cat([self._xyz, self._xyz[split_indices]], dim=0))
    #         self._cholesky = torch.nn.Parameter(torch.cat([self._cholesky, self._cholesky[split_indices] / 1.6], dim=0))
    #         self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, self._features_dc[split_indices]], dim=0))
    #         self._opacity = torch.cat([self._opacity, self._opacity[split_indices]], dim=0)

    #     # Perform the Clone operation
    #     if len(clone_indices) > 0:
    #         self._xyz = torch.nn.Parameter(torch.cat([self._xyz, self._xyz[clone_indices]], dim=0))
    #         self._cholesky = torch.nn.Parameter(torch.cat([self._cholesky, self._cholesky[clone_indices]], dim=0))
    #         self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, self._features_dc[clone_indices]], dim=0))
    #         self._opacity = torch.cat([self._opacity, self._opacity[clone_indices]], dim=0)
        
    #     if points_to_remove > 0:
    #         # Remove the points with the smallest gradients, i.e., from the tail of the sorted list
    #         remove_indices = sorted_indices[-points_to_remove:]  # Get the indices of the smallest gradients

    #         # Create a mask for the indices to keep
    #         keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
    #         keep_indices[remove_indices] = False  # Set the remove indices to False to exclude them

    #         # Remove points based on the mask
    #         self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
    #         self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
    #         self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
    #         self._opacity = self._opacity[keep_indices]

    #     # Update the optimizer with new parameters
    #     self.update_optimizer()
    #     #print(f"current_number:{self._xyz.shape[0]}, split_indices: {len(split_indices)}, clone_indices: {len(clone_indices)}")

    # def density_control(self, iter):
    #     iter_threshold_remove = self.iterations/3  # 根据您的训练计划调整这个阈值
    #     iter_threshold_add = self.iterations*2/3
    #     if iter > iter_threshold_add:
    #         return
    #     grad_xyz = self._xyz.grad
    #     if grad_xyz is None:
    #         raise RuntimeError("grad_xyz is None,请检查 self._xyz 是否参与了计算图。")

    #     # 计算每个点的梯度幅值
    #     grad_magnitude = torch.norm(grad_xyz, dim=1)

    #     # 对梯度幅值进行升序排序（最小的梯度在前）
    #     sorted_grad_magnitude, sorted_indices = torch.sort(grad_magnitude)

    #     if iter <= iter_threshold_remove:
    #         # 训练早期：只执行删除操作，减少总的高斯点数量
            
    #         remove_count = int(0.001 * self.max_num_points)  # 删除0.1%的点
            
    #         remove_indices = sorted_indices[:remove_count]

    #         # 删除选定的点
    #         keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
    #         keep_indices[remove_indices] = False

    #         self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
    #         self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
    #         self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
    #         self._opacity = self._opacity[keep_indices]
    #     elif iter > iter_threshold_remove:
    #         # 训练后期：只执行增加操作，通过拆分和克隆增加高斯点数量
    #         percentile_count = int(0.001 * self.max_num_points)  # 选择梯度最大的0.25%的点
    #         if percentile_count >= self.max_num_points-len(grad_magnitude):
    #             percentile_count = self.max_num_points-len(grad_magnitude)
    #         if percentile_count<=0:
    #             return
    #         top_indices = sorted_indices[-percentile_count:]  # 梯度最大的点的索引

    #         # 计算选定点的高斯值
    #         gaussian_values = torch.exp(
    #             -0.5 * torch.sum(
    #                 self.get_xyz[top_indices] ** 2 /
    #                 torch.clamp(self.get_cholesky_elements[top_indices][:, [0, 2]], min=1e-6),
    #                 dim=1
    #             )
    #         )
    #         gaussian_threshold = torch.median(gaussian_values)

    #         # 根据高斯阈值选择拆分和克隆的点
    #         split_indices = top_indices[gaussian_values > gaussian_threshold]
    #         clone_indices = top_indices[gaussian_values <= gaussian_threshold]

    #         # 执行拆分操作split
    #         if len(split_indices) > 0:
    #             self._xyz = torch.nn.Parameter(torch.cat([self._xyz, self._xyz[split_indices]], dim=0))
    #             self._cholesky = torch.nn.Parameter(torch.cat([self._cholesky, self._cholesky[split_indices] / 1.6], dim=0))
    #             self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, self._features_dc[split_indices]], dim=0))
    #             self._opacity = torch.cat([self._opacity, self._opacity[split_indices]], dim=0)

    #         # 执行克隆操作clone
    #         if len(clone_indices) > 0:
    #             self._xyz = torch.nn.Parameter(torch.cat([self._xyz, self._xyz[clone_indices]], dim=0))
    #             self._cholesky = torch.nn.Parameter(torch.cat([self._cholesky, self._cholesky[clone_indices]], dim=0))
    #             self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, self._features_dc[clone_indices]], dim=0))
    #             self._opacity = torch.cat([self._opacity, self._opacity[clone_indices]], dim=0)

    #     # 更新优化器中的参数
    #     self.update_optimizer()

    # def density_control(self, iter):
    #     iter_threshold_remove = self.iterations / 4  # 根据训练计划调整这个阈值
    #     iter_threshold_add = self.iterations * 1 / 2
    #     if iter > iter_threshold_add:
    #         return
    #     grad_xyz = self._xyz.grad
    #     if grad_xyz is None:
    #         raise RuntimeError("grad_xyz is None,请检查 self._xyz 是否参与了计算图。")

    #     # 计算每个点的梯度幅值
    #     grad_magnitude = torch.norm(grad_xyz, dim=1)

    #     # 对梯度幅值进行升序排序（最小的梯度在前）
    #     sorted_grad_magnitude, sorted_indices = torch.sort(grad_magnitude)

    #     if iter <= iter_threshold_remove:
    #         # 训练早期：只执行删除操作，减少总的高斯点数量
    #         remove_count = int(0.001 * self.max_num_points)  # 删除0.1%的点
    #         remove_indices = sorted_indices[:remove_count]

    #         # 删除选定的点
    #         keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
    #         keep_indices[remove_indices] = False

    #         self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
    #         self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
    #         self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
    #         self._opacity = self._opacity[keep_indices]
    #     elif iter > iter_threshold_remove:
    #         # 训练后期：通过拆分和克隆增加高斯点数量
    #         percentile_count = int(0.001 * self.max_num_points)  # 选择梯度最大的点
    #         if percentile_count >= self.max_num_points - len(grad_magnitude):
    #             percentile_count = self.max_num_points - len(grad_magnitude)
    #         if percentile_count <= 0:
    #             return
    #         top_indices = sorted_indices[-percentile_count:]  # 梯度最大的点的索引

    #         # 计算选定点的高斯值
    #         # gaussian_values = torch.exp(
    #         #     -0.5 * torch.sum(
    #         #         self.get_xyz[top_indices] ** 2 /
    #         #         torch.clamp(self.get_cholesky_elements[top_indices][:, [0, 2]], min=1e-6),
    #         #         dim=1
    #         #     )
    #         # )
    #         gaussian_values = torch.norm(torch.sigmoid(self.get_cholesky_elements[top_indices][:, 0:2]), dim=1, p=2)

    #         gaussian_threshold = torch.median(gaussian_values)

    #         # 根据高斯阈值选择拆分和克隆的点
    #         split_indices = top_indices[gaussian_values > gaussian_threshold]
    #         clone_indices = top_indices[gaussian_values <= gaussian_threshold]

            
    #         # 执行克隆操作clone
    #         if len(clone_indices) > 0:
    #             # 克隆点沿梯度方向移动
    #             new_positions = self._xyz[clone_indices] + grad_xyz[clone_indices] * 0.001  # 移动距离基于梯度
    #             self._xyz = torch.nn.Parameter(torch.cat([self._xyz, new_positions], dim=0))
    #             self._cholesky = torch.nn.Parameter(torch.cat([self._cholesky, self._cholesky[clone_indices]], dim=0))
    #             self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, self._features_dc[clone_indices]], dim=0))
    #             self._opacity = torch.cat([self._opacity, self._opacity[clone_indices]], dim=0)

    #         # 执行拆分操作split
    #         if len(split_indices) > 0:
                
                
    #             # 生成新的位置，根据高斯分布的PDF进行采样
    #             orig_positions = self._xyz[split_indices]
    #             cholesky_vec  = self._cholesky[split_indices]

    #             # # 生成两个新位置，基于高斯分布随机采样偏移
    #             # new_positions_1 = orig_positions + torch.randn_like(orig_positions) * cov_matrix[:, 0:1]  # 沿着主轴方向偏移
    #             # new_positions_2 = orig_positions - torch.randn_like(orig_positions) * cov_matrix[:, 0:1]  # 沿反向偏移

                
    #             new_positions_1_list = []
    #             new_positions_2_list = []
    #             for i in range(orig_positions.shape[0]):
    #                 l1, l2, l3 = cholesky_vec[i]
    #                 L = torch.tensor([[l1, 0.0], [l2, l3]], device=self.device)
    #                 cov_matrix = L @ L.T*0.001
    #                 distribution = MultivariateNormal(orig_positions[i], cov_matrix)
    #                 new_position_1 = distribution.sample().to(self.device)
    #                 new_position_2 = distribution.sample().to(self.device)
    #                 new_positions_1_list.append(new_position_1)
    #                 new_positions_2_list.append(new_position_2)
    #             new_positions_1 = torch.stack(new_positions_1_list)
    #             new_positions_2 = torch.stack(new_positions_2_list)
    #             # 更新 xyz 和 cholesky（将尺度缩小 1.6 倍），添加新点
    #             self._xyz = torch.nn.Parameter(torch.cat([self._xyz, new_positions_1, new_positions_2], dim=0))
    #             self._cholesky = torch.nn.Parameter(torch.cat([self._cholesky, self._cholesky[split_indices] / 1.6, self._cholesky[split_indices] / 1.6], dim=0))
    #             self._features_dc = torch.nn.Parameter(torch.cat([self._features_dc, self._features_dc[split_indices], self._features_dc[split_indices]], dim=0))
    #             self._opacity = torch.cat([self._opacity, self._opacity[split_indices], self._opacity[split_indices]], dim=0)
                
    #             keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
    #             keep_indices[split_indices] = False
                
    #             # 更新xyz和其他相关的参数，删除
    #             self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
    #             self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
    #             self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
    #             self._opacity = self._opacity[keep_indices]

                
    #     # 更新优化器中的参数
    #     self.update_optimizer()

    # def density_control(self, iter):
    #     iter_threshold_remove = 4000  # 根据训练计划调整这个阈值
    #     if iter > iter_threshold_remove:
    #         return
    #     grad_xyz = self._xyz.grad
    #     grad_eatures_dc = self._features_dc.grad
    #     if grad_xyz is None:
    #         raise RuntimeError("grad_xyz is None,请检查 self._xyz 是否参与了计算图。")
    #     if grad_eatures_dc is None:
    #         raise RuntimeError("grad_xyz is None,请检查 self._xyz 是否参与了计算图。")
    #     # 计算每个点的梯度幅值
    #     # grad_magnitude = torch.norm(grad_xyz, dim=1)
    #     grad_magnitude = torch.norm(grad_eatures_dc, dim=1)+torch.norm(grad_xyz, dim=1)

    #     # 对梯度幅值进行升序排序（最小的梯度在前）
    #     _, sorted_indices = torch.sort(grad_magnitude)
    #     removal_rate_per_step = self.removal_rate/int(iter_threshold_remove/(self.densification_interval))
    #     if iter < iter_threshold_remove:
    #         # 训练早期：只执行删除操作，减少总的高斯点数量
    #         remove_count = int(removal_rate_per_step * self.max_num_points)
            
    #         remove_indices = sorted_indices[:remove_count]

    #         # 删除选定的点
    #         keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
    #         keep_indices[remove_indices] = False

    #         self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
    #         self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
    #         self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
    #         self._opacity = self._opacity[keep_indices]
    #     elif iter == iter_threshold_remove:
    #         # 训练早期：只执行删除操作，减少总的高斯点数量
    #         remove_count = self._xyz.shape[0]-int(self.max_num_points * (1-self.removal_rate))
    #         if remove_count>0:
    #             remove_indices = sorted_indices[:remove_count]

    #             # 删除选定的点
    #             keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
    #             keep_indices[remove_indices] = False

    #             self._xyz = torch.nn.Parameter(self._xyz[keep_indices])
    #             self._cholesky = torch.nn.Parameter(self._cholesky[keep_indices])
    #             self._features_dc = torch.nn.Parameter(self._features_dc[keep_indices])
    #             self._opacity = self._opacity[keep_indices]    
    #     # 更新优化器中的参数
    #     self.update_optimizer()

    def density_control(self, iter):
        iter_threshold_remove = 4000  # 根据训练计划调整这个阈值
        if iter > iter_threshold_remove:
            return
        opacity = self._opacity
        _, sorted_indices = torch.sort(opacity)
        removal_rate_per_step = self.removal_rate/int(iter_threshold_remove/(self.densification_interval))
        if iter < iter_threshold_remove:
           
            remove_count = int(removal_rate_per_step * self.max_num_points)
            
            remove_indices = sorted_indices[:remove_count]

            
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
            # for param_group in self.optimizer.param_groups:
            #     for param in param_group['params']:
            #         print(param.size(), param.requires_grad)
            
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
        if (iter) % (self.densification_interval+1) == 0 and iter > 0 and isdensity:
            self.density_control(iter)
            # for param_group in self.optimizer.param_groups:
            #     for param in param_group['params']:
            #         print(param.size(), param.requires_grad)
            
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        
        self.scheduler.step()
        return loss, psnr,image
    

    
