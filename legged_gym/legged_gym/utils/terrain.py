# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        # Slightly increase non-stair terrain difficulty while keeping
        # stair-specific geometry unchanged.
        slope = difficulty * 0.7
        amplitude = 0.015 + 0.1 * difficulty
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.06 + difficulty * 0.2
        stepping_stones_size = 1.35 * (1.02 - difficulty)
        stone_distance = 0.08 if difficulty == 0 else 0.12
        gap_size = 1.15 * difficulty
        pit_depth = 1.15 * difficulty
        bridge_gap_size = 0.12 + 0.55 * difficulty    # 缺口随难度变大
        bridge_width = 0.75 - 0.3 * difficulty        # 桥宽随难度变窄
        if choice < self.proportions[0]:
            terrain.height_field_raw[:, :] = 0
        elif choice < self.proportions[1]:
            if choice < (self.proportions[0] + self.proportions[1]) / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[2]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude, max_height=amplitude, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[4]:
            if choice < self.proportions[3]:
                step_height *= -1

            # 楼梯宽度的课程学习逻辑
            # difficulty = 0 (Level 0) -> 宽度 0.3米
            # difficulty = 1 (Level 10) -> 宽度 0.2米

            # 使用线性插值
            current_step_width = 0.3 - 0.1 * difficulty
            # 确保不小于 0.2
            current_step_width = max(current_step_width, 0.2)

            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=0.3,
                step_height=step_height,
                platform_size=3.,
            )
        elif choice < self.proportions[5]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[6]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[7]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[8]:
            bridge_gap_options = getattr(self.cfg, "bridge_gap_options", None)
            bridge_plank_length_options = getattr(self.cfg, "bridge_plank_length_options", None)
            bridge_pit_depth_options = getattr(self.cfg, "bridge_pit_depth_options", None)

            if bridge_gap_options and bridge_plank_length_options:
                num_bridge_levels = min(len(bridge_gap_options), len(bridge_plank_length_options))
                bridge_idx = min(int(difficulty * num_bridge_levels), num_bridge_levels - 1)
                current_gap = bridge_gap_options[bridge_idx]
                current_plank_length = bridge_plank_length_options[bridge_idx]
                current_pit_depth = (
                    bridge_pit_depth_options[bridge_idx]
                    if bridge_pit_depth_options and bridge_idx < len(bridge_pit_depth_options)
                    else 2.0
                )
            else:
                # 默认连续桥课程，兼容原 black 任务
                current_gap = 0.12 + 0.24 * difficulty
                current_plank_length = 0.4
                current_pit_depth = 2.0

            plank_bridge_terrain(
                terrain,
                gap_size=current_gap,
                plank_length=current_plank_length,
                plank_width=getattr(self.cfg, "bridge_plank_width", 3.6),
                height=getattr(self.cfg, "bridge_height", 0.0),
                platform_len=getattr(self.cfg, "bridge_platform_len", 2.0),
                pit_depth=current_pit_depth,
            )
        else:
            wall_h = 0.12 + 0.34 * difficulty  # 0.12m 到 0.46m
            high_wall_terrain(
                terrain,
                height=wall_h,
                width_options=getattr(self.cfg, "high_wall_width_options", (0.0, 0.10)),
                num_walls=getattr(self.cfg, "high_wall_num_walls", 3),
                start_distance=getattr(self.cfg, "high_wall_start_distance", 1.2),
                spacing=getattr(self.cfg, "high_wall_spacing", 1.2),
                fill_full_block=getattr(self.cfg, "high_wall_fill_full_block", False),
                spawn_clearance=getattr(self.cfg, "high_wall_spawn_clearance", 0.8),
            )
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def broken_bridge_terrain(terrain, gap_size, platform_length, bridge_width, depth=1.0):
    """
    生成断桥地形：一条狭窄的路径，中间有随机或固定的缺口
    :param gap_size: 缺口的长度 [m]
    :param platform_length: 桥面每段的长度 [m]
    :param bridge_width: 桥的宽度 [m]
    :param depth: 缺口深度 [m] (实际上是负高度)
    """
    # 将物理尺寸转换为像素尺寸
    gap_pixels = int(gap_size / terrain.horizontal_scale)
    platform_pixels = int(platform_length / terrain.horizontal_scale)
    width_pixels = int(bridge_width / terrain.horizontal_scale)
    depth_raw = int(depth / terrain.vertical_scale)

    # 1. 先把整个区域挖成深坑 (depth)
    terrain.height_field_raw[:] = -depth_raw

    # 2. 计算桥的中心线
    mid_y = terrain.width // 2
    y_start = mid_y - width_pixels // 2
    y_end = mid_y + width_pixels // 2

    # 3. 沿 X 轴构建断续的桥面
    # 机器人从 x=0 向 x_end 移动
    current_x = 0
    while current_x < terrain.length:
        # 确保桥面在地图范围内
        p_end = min(current_x + platform_pixels, terrain.length)
        
        # 填充桥面高度为 0 (或者你可以设置为特定高度)
        terrain.height_field_raw[current_x:p_end, y_start:y_end] = 0
        
        # 移动到下一段（跳过缺口）
        current_x += platform_pixels + gap_pixels

def high_wall_terrain(
    terrain,
    height=1.0,
    width=0.2,
    distance=2.0,
    width_options=None,
    num_walls=1,
    start_distance=None,
    spacing=0.8,
    fill_full_block=False,
    spawn_clearance=0.8,
):
    """
    生成高墙地形
    :param terrain: 地形对象
    :param height: 墙的高度 [m]
    :param width: 墙的厚度 [m]，兼容旧单墙调用
    :param distance: 墙距离地图中心的距离 [m]，兼容旧单墙调用
    :param width_options: 多墙模式下每堵墙可选厚度 [m]；0 表示单行薄墙脊
    :param num_walls: 多墙模式下每块地形的墙数
    :param start_distance: 第一堵墙距离地图中心的距离 [m]
    :param spacing: 相邻高墙的固定间距 [m]
    :param fill_full_block: 是否按 spacing 在整块地形内周期铺墙
    :param spawn_clearance: fill_full_block 下跳过地图中心出生点附近的半径 [m]
    """
    h_raw = int(height / terrain.vertical_scale)
    center_x = terrain.length // 2
    terrain.height_field_raw[:, :] = 0

    if width_options is None:
        wall_specs = [(distance, width)]
    else:
        num_walls = max(1, int(num_walls))
        start_distance = distance if start_distance is None else start_distance
        if fill_full_block:
            spacing = max(float(spacing), terrain.horizontal_scale)
            spawn_clearance = max(float(spawn_clearance), 0.0)
            half_length = terrain.length * terrain.horizontal_scale * 0.5
            start_distance = abs(float(start_distance))
            if start_distance <= spawn_clearance:
                start_distance = spawn_clearance + spacing
            forward_distances = np.arange(start_distance, half_length, spacing)
            wall_distances = np.concatenate((-forward_distances[::-1], forward_distances))
            wall_distances = [d for d in wall_distances if abs(d) > spawn_clearance]
            wall_specs = []
            for wall_distance in wall_distances:
                phase_idx = int(round((abs(wall_distance) - start_distance) / spacing))
                wall_specs.append(
                    (float(wall_distance), float(width_options[phase_idx % len(width_options)]))
                )
        else:
            wall_specs = [
                (start_distance + wall_idx * spacing, float(width_options[wall_idx % len(width_options)]))
                for wall_idx in range(num_walls)
            ]

    for wall_distance, wall_width in wall_specs:
        dist_pixels = int(wall_distance / terrain.horizontal_scale)
        if wall_width <= 0.0:
            w_pixels = 1
        else:
            # Heightfield values live on vertices. One raised row is a ridge with nearly zero
            # plateau width, so a 0.10m wall at 0.10m resolution needs two raised rows.
            w_pixels = max(2, int(np.ceil(wall_width / terrain.horizontal_scale)) + 1)

        wall_start = center_x + dist_pixels
        wall_end = wall_start + w_pixels

        wall_start = np.clip(wall_start, 0, terrain.length)
        wall_end = np.clip(wall_end, 0, terrain.length)

        if wall_end > wall_start:
            terrain.height_field_raw[wall_start:wall_end, :] = h_raw

    return terrain

def plank_bridge_terrain(
    terrain,
    gap_size=0.15,
    plank_length=0.5,
    plank_width=1.0,
    height=0.5,
    platform_len=2.0,
    pit_depth=2.0,
):
    """
    生成木板桥地形，并在中心保留出生平台
    :param terrain: 地形对象
    :param gap_size: 木板间的空隙 [m]
    :param plank_length: 木板长度 [m]
    :param plank_width: 木板宽度 [m]
    :param height: 木板高度 [m]
    :param platform_len: 中心出生平台的长度 [m] (机器人出生在中心，必须留平地)
    """
    # 1. 坐标转换
    gap_pixels = int(gap_size / terrain.horizontal_scale)
    plank_len_pixels = int(plank_length / terrain.horizontal_scale)
    width_pixels = int(plank_width / terrain.horizontal_scale)
    height_raw = int(height / terrain.vertical_scale)
    platform_pixels = int(platform_len / terrain.horizontal_scale)
    
    # 2. 初始化深坑背景
    pit_depth_raw = int(pit_depth / terrain.vertical_scale)
    terrain.height_field_raw[:, :] = -pit_depth_raw

    # 3. 计算桥的Y轴范围 (宽度)
    mid_y = terrain.width // 2
    y_start = mid_y - width_pixels // 2
    y_end = mid_y + width_pixels // 2

    # 4. 铺设木板 (全图铺设)
    current_x = 0
    while current_x < terrain.length:
        plank_end = min(current_x + plank_len_pixels, terrain.length)
        # 填入木板高度
        terrain.height_field_raw[current_x:plank_end, y_start:y_end] = height_raw
        # 跳过木板+间隙
        current_x += plank_len_pixels + gap_pixels

    # 5. 强制填平中心区域作为出生点
    mid_x = terrain.length // 2
    plat_start = mid_x - platform_pixels // 2
    plat_end = mid_x + platform_pixels // 2
    
    # 边界保护
    plat_start = max(0, plat_start)
    plat_end = min(terrain.length, plat_end)
    
    # 将中心区域强制设为平地高度，覆盖掉可能存在的间隙
    terrain.height_field_raw[plat_start:plat_end, y_start:y_end] = height_raw

    return terrain
