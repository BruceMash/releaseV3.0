"""
智能空面协同任务博弈对抗规划系统 - ，有实时监控，修改触发机制，手动触发!
"""

import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Optional, Set
import time
from dataclasses import dataclass
import warnings
import os
import threading
import time
import sys
# 添加到文件最顶部的 import 区域
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'

@dataclass
class Config:
    """配置类"""
    MAP_SIZE_X: float = 100e3
    MAP_SIZE_Y: float = 100e3
    MAP_HEIGHT: float = 100e3

    N_PLATFORMS: int = 10
    N_TARGETS: int = 16

    PLATFORM_DIM: int = 11
    TARGET_DIM: int = 16
    HIDDEN_DIM: int = 256
    ACTION_DIM: int = 17  # 16 targets + 1 idle
    PENALTY_TEMPORAL: float = 100.0  # 时序违反惩罚
    REWARD_TEMPORAL: float = 10.0  # 时序满足奖励

    EXPLORATION_RESTART_EVERY: int = 600  # 每600回合重启探索
    RESTART_EPSILON: float = 0.5  # 重启时的探索率

    GAMMA: float = 0.99
    TAU: float = 0.005
    LR_ACTOR: float = 3e-4
    LR_CRITIC: float = 3e-4
    BUFFER_SIZE: int = 500000
    BATCH_SIZE: int = 128

    MAX_EPISODES: int = 500
    MAX_STEPS: int = 1
    WARMUP_STEPS: int = 200
    UPDATE_INTERVAL: int = 1

    EPSILON_START: float = 0.7
    EPSILON_END: float = 0.35
    EPSILON_DECAY: float = 0.998
    PARAM_NOISE_SCALE: float = 0.07  # 新增参数，重启时参数噪声强度
    ACTION_NOISE_TEMP: float = 1.5  # 新增参数，重启时动作选择温度（越高越随机）

    # 惩罚系数 - 用于约束检查（不影响奖励计算，仅用于统计）
    PENALTY_DUPLICATE: float = 500.0
    PENALTY_DUPLICATE_EXP: float = 2.0
    PENALTY_CAPABILITY: float = 50.0
    PENALTY_PAYLOAD: float = 30.0
    PENALTY_RANGE: float = 40.0

    # 奖励缩放因子（用于数值稳定性）
    REWARD_SCALE: float = 10.0  # 降低缩放因子
    REWARD_MAX: float = 50.0  # 奖励裁剪上限
    REWARD_MIN: float = -20.0  # 奖励裁剪下限
    # 添加归一化窗口
    REWARD_NORM_WINDOW: int = 1000

    CONVERGENCE_WINDOW: int = 50  # 检查窗口大小
    CONVERGENCE_THRESHOLD: float = 0.01  # 奖励变化阈值（1%）
    PATIENCE: int = 1000  # 容忍多少个窗口无改进就停止
    MIN_EPISODES: int = 1500  # 最少训练回合数（防止过早停止）
    SAVE_BEST: bool = True  # 是否保存最佳模型

config = Config()


class TerrainLoader:
    """地形加载器 - 支持纯模拟模式"""

    def __init__(self, csv_path='江宁.csv', use_simulation=False):
        # 如果指定使用模拟模式或CSV路径为None，直接生成模拟数据
        if use_simulation or csv_path is None:
            print("使用100km×100km模拟地形数据（无CSV文件依赖）")

            # 计算100km对应的经纬度范围
            # 纬度1度 ≈ 110.54km，经度1度 ≈ 111.32km × cos(纬度)
            center_lon, center_lat = 118.85, 31.85  # 模拟中心点（大致江宁区域）
            delta_lat = 100e3 / 110540  # 约0.904度
            delta_lon = 100e3 / (111320 * np.cos(np.radians(center_lat)))  # 约1.06度

            # 生成100×100的网格（覆盖100km×100km）
            n_points = 100
            lons = np.linspace(center_lon - delta_lon / 2, center_lon + delta_lon / 2, n_points)
            lats = np.linspace(center_lat - delta_lat / 2, center_lat + delta_lat / 2, n_points)
            # 模拟地形高程：0-500m随机，可改为特定模式
            elevs = np.random.uniform(0, 500, n_points * n_points)

            self.df = pd.DataFrame({
                'Longitude': np.repeat(lons, n_points),
                'Latitude': np.tile(lats, n_points),
                'Elevation': elevs
            })
        else:
            try:
                self.df = pd.read_csv(csv_path, usecols=['Longitude', 'Latitude', 'Elevation'])
                print(f"已加载真实地形数据: {csv_path}")
            except FileNotFoundError:
                # 如果文件不存在，自动回退到模拟数据（100km范围）
                print(f"未找到 {csv_path}，自动使用100km×100km模拟地形")
                center_lon, center_lat = 118.85, 31.85
                delta_lat = 100e3 / 110540
                delta_lon = 100e3 / (111320 * np.cos(np.radians(center_lat)))
                n_points = 100
                lons = np.linspace(center_lon - delta_lon / 2, center_lon + delta_lon / 2, n_points)
                lats = np.linspace(center_lat - delta_lat / 2, center_lat + delta_lat / 2, n_points)
                elevs = np.random.uniform(0, 500, n_points * n_points)
                self.df = pd.DataFrame({
                    'Longitude': np.repeat(lons, n_points),
                    'Latitude': np.tile(lats, n_points),
                    'Elevation': elevs
                })

        self.lons = self.df['Longitude'].values
        self.lats = self.df['Latitude'].values
        self.elevs = self.df['Elevation'].values
        self.lon_min, self.lon_max = self.lons.min(), self.lons.max()
        self.lat_min, self.lat_max = self.lats.min(), self.lats.max()

        # 构建插值器
        ul, ut = np.unique(self.lons), np.unique(self.lats)
        n_lon, n_lat = len(ul), len(ut)
        # lons/lats/elevs 本身就是按网格展平生成的，直接 reshape 即可
        grid = self.elevs.reshape(n_lat, n_lon)
        self.interp = RegularGridInterpolator((ut, ul), grid, method='linear',
                                              bounds_error=False, fill_value=0)
        self.grid_lons, self.grid_lats = ul, ut

    def get_elev(self, lon, lat):
        """获取海拔高度"""
        return float(self.interp([[lat, lon]])[0])

    def geo2local(self, lon, lat, elev=None):
        """地理坐标转局部坐标（以地图中心为原点）"""
        if elev is None:
            elev = self.get_elev(lon, lat)
        lc = (self.lon_min + self.lon_max) / 2
        tc = (self.lat_min + self.lat_max) / 2
        ec = (self.elevs.min() + self.elevs.max()) / 2
        # 转换为米：纬度1度≈110540m，经度1度≈111320m×cos(纬度)
        mr = 111320 * np.cos(np.radians(tc))
        return np.array([(lon - lc) * mr, (lat - tc) * 110540, elev - ec], dtype=np.float32)

class Threat:
    """威胁区域类"""

    def __init__(self, center: np.ndarray, radius: float, intensity: float = 1.0,
                 threat_type: str = 'radar'):
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.intensity = intensity
        self.type = threat_type

    def contains(self, point: np.ndarray) -> bool:
        """检查点是否在威胁区域内"""
        dist = np.linalg.norm(point[:2] - self.center[:2])
        return dist <= self.radius

class RoutePlanner:
    """
    知识数据联合驱动的路径规划器（对应申请书第3页、第13页）
    基于航程估计生成实际航路点，支持威胁规避和地形跟随
    """

    def __init__(self, route_estimator: 'RouteEstimator', terrain: TerrainLoader, threats: List[Threat]):
        self.estimator = route_estimator
        self.terrain = terrain
        self.threats = threats
        self.waypoint_cache = {}

        # 知识引导的规划参数（来自专家经验）
        self.knowledge_params = {
            'safety_margin': 2000.0,  # 威胁区安全余量(m)
            'min_turn_radius': 500.0,  # 最小转弯半径(m)
            'max_climb_rate': 30.0,  # 最大爬升率(m/s)
            'terrain_clearance': 100.0  # 地形安全高度(m)
        }

    def plan_path(self, start: np.ndarray, end: np.ndarray, platform_type: str = 'UAV',
                  n_waypoints: int = 5) -> List[np.ndarray]:
        """
        生成航路点序列（对应申请书第13-14页的路径规划）

        Returns:
            waypoints: 包含起点、中间航路点、终点的列表
        """
        cache_key = (tuple(start[:2]), tuple(end[:2]), platform_type)
        if cache_key in self.waypoint_cache:
            return self.waypoint_cache[cache_key]

        waypoints = [start.copy()]

        # 步骤1：二分搜索检测威胁和地形穿越点（复用RouteEstimator）
        mid_points = self._detect_critical_points(start, end)

        # 步骤2：基于知识规则生成规避航路点
        for point_type, point_pos in mid_points:
            if point_type == 'threat':
                # 知识规则：威胁规避（申请书第13页禁飞区规避）
                avoid_point = self._calculate_threat_avoidance(
                    waypoints[-1], end, point_pos,
                    self.knowledge_params['safety_margin']
                )
                waypoints.append(avoid_point)
            elif point_type == 'terrain':
                # 知识规则：地形跟随（申请书第13页地形穿越）
                climb_point = self._calculate_terrain_following(
                    point_pos, self.knowledge_params['terrain_clearance']
                )
                waypoints.append(climb_point)

        waypoints.append(end.copy())

        # 步骤3：路径平滑（Dubins曲线或B样条，符合飞行器动力学约束）
        smoothed_path = self._smooth_path_dubins(waypoints, platform_type)

        # 缓存结果
        self.waypoint_cache[cache_key] = smoothed_path
        return smoothed_path

    def _detect_critical_points(self, start: np.ndarray, end: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """二分搜索检测关键穿越点（威胁和地形）"""
        critical_points = []
        stack = [(start, end)]
        visited = set()

        while stack:
            p1, p2 = stack.pop()
            mid = (p1 + p2) / 2
            ds = np.linalg.norm(p2 - p1)

            if ds <= self.estimator.min_ds:
                # 检查威胁
                for threat in self.threats:
                    if threat.contains(mid) and tuple(mid) not in visited:
                        critical_points.append(('threat', mid.copy()))
                        visited.add(tuple(mid))
                        break

                # 检查地形
                elev = self.terrain.get_elev(mid[0], mid[1]) if len(mid) > 2 else 0
                if elev > 50 and tuple(mid) not in visited:  # 高地
                    critical_points.append(('terrain', mid.copy()))
                    visited.add(tuple(mid))
            else:
                stack.append((mid, p2))
                stack.append((p1, mid))

        return critical_points

    def _calculate_threat_avoidance(self, start: np.ndarray, end: np.ndarray,
                                    threat_center: np.ndarray, margin: float) -> np.ndarray:
        """计算威胁规避点（垂直于航线偏移）"""
        direction = end[:2] - start[:2]
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        # 垂直方向（左右规避，选择更近的方向）
        perp = np.array([-direction[1], direction[0]])
        to_threat = threat_center[:2] - start[:2]

        # 选择远离威胁中心的垂直方向
        if np.dot(perp, to_threat) > 0:
            perp = -perp

        avoid_pos = threat_center[:2] + perp * (margin + 1000)  # 1km额外余量
        height = max(100, self.terrain.get_elev(avoid_pos[0], avoid_pos[1]) + 200)

        return np.array([avoid_pos[0], avoid_pos[1], height])

    def _calculate_terrain_following(self, point: np.ndarray, clearance: float) -> np.ndarray:
        """地形跟随：保持安全高度"""
        if len(point) < 3:
            elev = self.terrain.get_elev(point[0], point[1])
            return np.array([point[0], point[1], elev + clearance])
        return point

    def _smooth_path_dubins(self, waypoints: List[np.ndarray], platform_type: str) -> List[np.ndarray]:
        """
        Dubins路径平滑（符合申请书第3页的"编队路径规划"要求）
        考虑平台最小转弯半径约束
        """
        if len(waypoints) <= 2:
            return waypoints

        # 根据平台类型设置转弯半径（知识驱动）
        turn_radius = {
            'UAV': 800.0,
            'UCAV': 1200.0,
            'USV': 2000.0,
            'UUV': 3000.0
        }.get(platform_type, 1000.0)

        smoothed = [waypoints[0]]

        # 简化的路径平滑：在转弯处插入圆弧路径点
        for i in range(1, len(waypoints) - 1):
            prev_p = waypoints[i - 1]
            curr_p = waypoints[i]
            next_p = waypoints[i + 1]

            # 计算转向角
            v1 = curr_p[:2] - prev_p[:2]
            v2 = next_p[:2] - curr_p[:2]
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)

            # 如果转向角过大，插入过渡点
            cos_angle = np.dot(v1_norm, v2_norm)
            if cos_angle < 0.9:  # 约大于26度角
                # 在转角处插入两个过渡点（简化的圆弧逼近）
                offset = turn_radius * 0.5
                mid1 = curr_p - v1_norm * offset
                mid2 = curr_p + v2_norm * offset
                smoothed.append(mid1)
                smoothed.append(mid2)
            else:
                smoothed.append(curr_p)

        smoothed.append(waypoints[-1])
        return smoothed

    def get_path_length(self, path: List[np.ndarray]) -> float:
        """计算路径总长度"""
        total = 0.0
        for i in range(len(path) - 1):
            total += np.linalg.norm(path[i + 1] - path[i])
        return total

class RouteEstimator:
    """
    二分搜索航程快速估计 - 对应申请书第13页
    计算包含地形和威胁惩罚的综合航程
    """

    def __init__(self, terrain: TerrainLoader, threats: List[Threat],
                 lambda_t: float = 0.3, lambda_n: float = 0.5, min_ds: float = 100.0):
        self.terrain = terrain
        self.threats = threats
        self.lambda_t = lambda_t  # 地形惩罚系数
        self.lambda_n = lambda_n  # 威胁惩罚系数
        self.min_ds = min_ds  # 最小二分判断长度

    def estimate_distance(self, start: np.ndarray, end: np.ndarray) -> float:
        """
        综合航程估计: d_total = d_base + λ_t*d_terrain + λ_n*d_threat
        对应申请书公式(3.3)
        """
        # 1. 基础欧氏距离
        d_base = np.linalg.norm(start[:3] - end[:3])

        # 2. 二分搜索计算地形穿越距离
        d_terrain = self._binary_search_terrain(start, end)

        # 3. 二分搜索计算威胁穿越距离
        d_threat = self._binary_search_threat(start, end)

        # 4. 综合航程（申请书第13页公式）
        total_dist = d_base + self.lambda_t * d_terrain + self.lambda_n * d_threat

        return float(total_dist)

    def _binary_search_terrain(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """二分搜索地形穿越点 - 迭代实现（避免递归深度限制）"""
        stack = [(p1, p2)]
        total_dist = 0.0

        while stack:
            start, end = stack.pop()
            ds = np.linalg.norm(end - start)

            if ds <= self.min_ds:
                # 到达最小分割长度，计算该段地形影响
                elev1 = self.terrain.get_elev(start[0], start[1]) if len(start) > 2 else 0
                elev2 = self.terrain.get_elev(end[0], end[1]) if len(end) > 2 else 0
                # 如果高度差大，认为穿越地形
                if abs(elev1 - elev2) > 50:
                    total_dist += ds * 0.5
            else:
                # 迭代二分
                mid = (start + end) / 2
                stack.append((mid, end))
                stack.append((start, mid))

        return total_dist

    def _binary_search_threat(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """二分搜索威胁区域穿越点，返回穿越距离 d^n"""
        mid = (p1 + p2) / 2
        ds = np.linalg.norm(p2 - p1)

        if ds <= self.min_ds:
            # 检查该点是否在威胁区域内
            in_threat = any(threat.contains(mid) for threat in self.threats)
            return ds if in_threat else 0.0

        # 递归二分
        left_dist = self._binary_search_threat(p1, mid)
        right_dist = self._binary_search_threat(mid, p2)
        return left_dist + right_dist




class Platform:
    """作战平台类"""
    TYPES = ['侦察', '打击', '察打一体']
    COLORS = {'侦察': '#1f77b4', '打击': '#d62728', '察打一体': '#2ca02c'}

    CAPABILITIES = {
        '侦察': ['侦察'],
        '打击': ['打击'],
        '察打一体': ['侦察', '打击', '察打一体']
    }

    DEFAULT_MAX_RANGE = {'侦察': 500e3, '打击': 800e3, '察打一体': 600e3}
    DEFAULT_SPEED = {'侦察': 300, '打击': 250, '察打一体': 280}
    DEFAULT_PAYLOAD = {'侦察': 4, '打击': 6, '察打一体': 5}

    def __init__(self, id: int, pos: np.ndarray, ptype: str = '侦察',
                 max_range: float = None, speed: float = None, payload: int = None):
        self.id = id
        self.position = pos.copy().astype(np.float32)
        self.type = ptype if ptype in self.TYPES else '侦察'

        # 如果外部传了参数，用外部的；没传就用默认值
        self.max_range = max_range if max_range is not None else self.DEFAULT_MAX_RANGE[self.type]
        self.speed = speed if speed is not None else self.DEFAULT_SPEED[self.type]
        self.max_payload = payload if payload is not None else self.DEFAULT_PAYLOAD[self.type]

        self.payload = self.max_payload
        self.assigned_targets = []
        self.current_fuel = self.max_range

    def get_state(self) -> np.ndarray:
        """获取平台状态向量"""
        type_enc = np.zeros(4, dtype=np.float32)
        type_enc[self.TYPES.index(self.type)] = 1.0

        return np.concatenate([
            self.position / [config.MAP_SIZE_X, config.MAP_SIZE_Y, config.MAP_HEIGHT],
            [self.payload / 8.0],
            [self.current_fuel / 1e6],
            type_enc,
            [len(self.assigned_targets) / config.N_TARGETS],
            [1.0 if self.payload > 0 else 0.0]
        ])

    def can_do(self, task_type: str) -> bool:
        """检查是否能执行某类任务"""
        return task_type in self.CAPABILITIES.get(self.type, [])

    def reset(self):
        """重置状态"""
        self.payload = self.max_payload
        self.current_fuel = self.max_range
        self.assigned_targets = []


class Target:
    """目标类"""
    TASKS = ['侦察', '打击', '察打一体']
    COLORS = {'侦察': '#ffd700', '打击': '#d62728', '察打一体': '#2ca02c'}
    MARKERS = {'侦察': 's', '打击': '^', '察打一体': 'd'}

    def __init__(self, id: int, pos: np.ndarray, task_type: str = '侦察',
                 value: float = 0.5, time_win_start: float = 0, time_win_end: float = 1000,
                 req_payload: int = 1, predecessors=None):
        self.id = id
        self.position = pos.copy().astype(np.float32)
        self.threat = random.uniform(0.1, 1.0)  # 威胁可以保留随机，或后面也开放输入
        self.value = value
        self.time_win_start = time_win_start
        self.time_win_end = time_win_end
        self.task_type = task_type if task_type in self.TASKS else '侦察'
        self.req_payload = req_payload
        self.completed = False
        self.assigned_platform = None
        self.priority = random.uniform(0.5, 1.0)
        self.predecessors = predecessors if predecessors else []

        # 时序边权重（用于GNN，值越小表示时间窗口约束越紧）
        if self.predecessors:
            self.temporal_weight = 1.0 / (self.time_win_start + 1.0)
        else:
            self.temporal_weight = 0.0

    def get_state(self) -> np.ndarray:
        """获取目标状态向量"""
        task_enc = np.zeros(3, dtype=np.float32)
        task_enc[self.TASKS.index(self.task_type)] = 1.0
        # 原有14维 + 新增2维（是否有前序、前序完成比例）
        has_pred = 1.0 if self.predecessors else 0.0
        pred_completion = self._get_pred_completion_ratio()

        return np.concatenate([
            self.position / [config.MAP_SIZE_X, config.MAP_SIZE_Y, config.MAP_HEIGHT],
            [self.threat],
            [self.value],
            [self.time_win_start / 1000, self.time_win_end / 1000],
            [self.req_payload / 3.0],
            task_enc,
            [1.0 if not self.completed else 0.0],
            [self.priority],
            [1.0 if self.assigned_platform is None else 0.0],
            # 新增时序约束特征：
            [has_pred],  # 是否有前序任务
            [pred_completion]  # 前序任务完成比例（0-1）
        ])

    def _get_pred_completion_ratio(self, env=None) -> float:
        """计算前序任务完成比例（需要外部提供env来计算）"""
        if not self.predecessors:
            return 1.0  # 无前序任务，视为已完成
        if env is None:
            return 0.0  # 默认返回0，实际计算时传入env
        completed = sum([1 for pid in self.predecessors if env.targets[pid].completed])
        return completed / len(self.predecessors)

    def check_predecessors_completed(self, env) -> bool:
        """检查所有前序任务是否已完成 - 核心约束检查"""
        if not self.predecessors:
            return True
        return all(env.targets[pid].completed for pid in self.predecessors)

    def reset(self):
        """重置状态"""
        self.completed = False
        self.assigned_platform = None





class Environment:
    """环境基类 - 修复：添加基类定义（原代码缺失此类）"""

    def __init__(self):
        self.platforms: List[Platform] = []
        self.targets: List[Target] = []
        self.threats: List[Threat] = []
        self.current_time = 0.0
        # 新增：跟踪已完成任务的集合（用于时序约束检查）
        self.completed_tasks: Set[int] = set()

    def check_constraints(self, assignment: Dict[int, List[int]], check_temporal: bool = True) -> Tuple[
        str, float, int]:
        if not assignment:
            return True, "OK", 0.0, 0

        penalty = 0.0
        reasons = []

        # 扁平化所有分配的内部节点
        all_assigned_nodes = []
        for tids in assignment.values():
            all_assigned_nodes.extend(tids)

        # 重复分配检查（基于内部节点ID）
        duplicates = len(all_assigned_nodes) - len(set(all_assigned_nodes))
        if duplicates > 0:
            penalty += config.PENALTY_DUPLICATE * duplicates
            reasons.append(f"Duplicate ({duplicates})")

        # 平台类型约束 + 时序约束
        for pid, tids in assignment.items():
            if pid >= len(self.platforms):
                continue
            for tid in tids:
                if tid >= len(self.targets):
                    continue
                platform = self.platforms[pid]
                target = self.targets[tid]

                if not platform.can_do(target.task_type):
                    penalty += config.PENALTY_CAPABILITY
                    reasons.append(f"P{pid} cannot do {target.task_type}")

                if check_temporal and not target.check_predecessors_completed(self):
                    penalty += config.PENALTY_TEMPORAL
                    reasons.append(f"Temporal violation: T{tid}")

        reason_str = "; ".join(reasons) if reasons else "OK"
        return penalty == 0, reason_str, penalty, duplicates

    def mark_target_completed(self, target_id: int):
        """标记任务已完成 - 用于时序约束传播"""
        if target_id < len(self.targets):
            self.targets[target_id].completed = True
            self.completed_tasks.add(target_id)

    def get_graph_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取图结构数据"""
        n_platforms = len(self.platforms)
        n_targets = len(self.targets)

        platform_states = torch.FloatTensor([p.get_state() for p in self.platforms])
        target_states = torch.FloatTensor([t.get_state() for t in self.targets])

        # 改为 numpy 数组进行计算，最后再转 tensor
        adj_matrix = np.zeros((n_platforms, n_targets), dtype=np.float32)
        temporal_matrix = np.zeros((n_targets, n_targets), dtype=np.float32)

        for i, p in enumerate(self.platforms):
            for j, t in enumerate(self.targets):
                if t.completed or not p.can_do(t.task_type):
                    continue
                dist = np.linalg.norm(p.position - t.position)
                adj_matrix[i, j] = 1.0 / (1.0 + dist / 1e5)

        for j, t_j in enumerate(self.targets):
            for k, t_k in enumerate(self.targets):
                if j in t_k.predecessors:
                    temporal_matrix[j, k] = 1.0 if t_j.completed else 0.0

        # 转回 torch tensor
        device = platform_states.device
        return platform_states, target_states, torch.FloatTensor(adj_matrix).to(device), torch.FloatTensor(
            temporal_matrix).to(device)

    def reset(self):
        for p in self.platforms:
            p.reset()
        for t in self.targets:
            t.reset()  # 确保completed=False
            # 关键：确保assigned_platform也重置
            t.assigned_platform = None
        self.completed_tasks.clear()  # 必须清空
        self.current_time = 0.0


class TerrainEnv(Environment):
    """地形环境类 - 默认使用100km×100km模拟地形"""

    def __init__(self, terrain_path=None, use_real_terrain=False):
        """
        参数:
            terrain_path: CSV文件路径（None表示使用模拟）
            use_real_terrain: 是否使用真实地形（默认False使用模拟）
        """
        super().__init__()
        # 默认使用模拟地形，除非指定使用真实地形
        self.terrain = TerrainLoader(
            csv_path=terrain_path if use_real_terrain else None,
            use_simulation=not use_real_terrain
        )

        # 更新地图尺寸（确保与Config一致）
        c1 = self.terrain.geo2local(self.terrain.lon_min, self.terrain.lat_min)
        c2 = self.terrain.geo2local(self.terrain.lon_max, self.terrain.lat_max)
        config.MAP_SIZE_X = abs(c2[0] - c1[0])
        config.MAP_SIZE_Y = abs(c2[1] - c1[1])

        print(f"模拟地形范围: {config.MAP_SIZE_X / 1000:.1f}km × {config.MAP_SIZE_Y / 1001:.1f}km")

        self._init_platforms()
        self._init_targets()
        self._init_threats()

    def setup_scenario(self, platform_configs: list, target_configs: list):
        """
        外部输入接口：
        - platform_configs: 平台配置列表
        - target_configs: 地理目标配置列表（16个）
          侦察目标 → 1个侦察任务节点
          察打一体目标 → 2个节点（侦察+打击，同一位置，时序绑定）
        内部实际任务节点数 = 6 + 10×2 = 26个
        """
        self.platforms.clear()
        self.targets.clear()
        self.geo_targets = []  # 记录16个地理目标与内部节点的映射

        # 创建平台
        for i, cfg in enumerate(platform_configs):
            pos = np.array(cfg['pos'], dtype=np.float32)
            p = Platform(
                id=i, pos=pos, ptype=cfg.get('type', '侦察'),
                max_range=cfg.get('max_range'),
                speed=cfg.get('speed'),
                payload=cfg.get('payload')
            )
            self.platforms.append(p)

        # 创建目标（察打一体自动拆分）
        node_id = 0
        for geo_id, cfg in enumerate(target_configs):
            pos = np.array(cfg['pos'], dtype=np.float32)
            ttype = cfg.get('type', '侦察')

            if ttype == '察打一体':
                # 阶段1：侦察（无前置）
                t_recon = Target(
                    id=node_id, pos=pos, task_type='侦察',
                    value=cfg.get('value', 0.5) * 0.4,
                    time_win_start=cfg.get('time_win_start', 0),
                    time_win_end=cfg.get('time_win_end', 1000),
                    req_payload=1, predecessors=[]
                )
                self.targets.append(t_recon)
                recon_id = node_id
                node_id += 1

                # 阶段2：打击（前置必须是本目标的侦察段）
                t_attack = Target(
                    id=node_id, pos=pos, task_type='打击',
                    value=cfg.get('value', 0.5) * 0.6,
                    time_win_start=cfg.get('time_win_start', 0) + 100,
                    time_win_end=cfg.get('time_win_end', 1000),
                    req_payload=cfg.get('req_payload', 2),
                    predecessors=[recon_id]
                )
                self.targets.append(t_attack)
                node_id += 1

                self.geo_targets.append({
                    'geo_id': geo_id, 'type': '察打一体',
                    'task_ids': [recon_id, recon_id + 1]
                })
            else:
                # 侦察目标
                t = Target(
                    id=node_id, pos=pos, task_type=ttype,
                    value=cfg.get('value', 0.5),
                    time_win_start=cfg.get('time_win_start', 0),
                    time_win_end=cfg.get('time_win_end', 1000),
                    req_payload=cfg.get('req_payload', 1),
                    predecessors=cfg.get('predecessors', [])
                )
                self.targets.append(t)
                self.geo_targets.append({
                    'geo_id': geo_id, 'type': '侦察',
                    'task_ids': [node_id]
                })
                node_id += 1

        # 更新全局配置（用实际节点数26，而非配置数16）
        config.N_PLATFORMS = len(platform_configs)
        config.N_TARGETS = len(self.targets)  # 26
        config.ACTION_DIM = len(self.targets) + 1  # 27


    def _init_platforms(self):
        return
        """初始化平台"""
        self.platforms = []
        types = ['UAV', 'UCAV', 'USV']
        for i in range(config.N_PLATFORMS):
            lon = random.uniform(self.terrain.lon_min, self.terrain.lon_max)
            lat = random.uniform(self.terrain.lat_min, self.terrain.lat_max)
            pos = self.terrain.geo2local(lon, lat)
            ptype = types[i % 3]
            pos[2] = 0 if ptype == 'USV' else random.uniform(1000, 80000)  # 1-80km高度分布
            self.platforms.append(Platform(i, pos, ptype))

    def _init_targets(self):
        return
        """初始化目标 - 修复版（解决IndexError）"""
        self.targets = []

        # 第一步：预生成所有目标的位置和任务类型（不创建对象）
        target_infos = []
        for i in range(config.N_TARGETS):
            lon = random.uniform(self.terrain.lon_min + 0.01, self.terrain.lon_max - 0.01)
            lat = random.uniform(self.terrain.lat_min + 0.01, self.terrain.lat_max - 0.01)
            pos = self.terrain.geo2local(lon, lat, self.terrain.get_elev(lon, lat))
            pos[2] = 0
            task_type = random.choice(Target.TASKS)
            target_infos.append({
                'pos': pos.copy(),
                'task_type': task_type,
                'predecessors': []
            })

        # 第二步：生成时序依赖关系（基于已收集的信息，确保无环）
        for i in range(config.N_TARGETS):
            task_type = target_infos[i]['task_type']

            if i < config.N_TARGETS * 0.3:
                target_infos[i]['predecessors'] = []
                continue

            if task_type == 'attack' and i > 0:
                # 攻击任务需要前置侦查任务（只能从前面目标中选择，确保无环）
                potential_preds = [j for j in range(i)
                                   if target_infos[j]['task_type'] == 'reconnaissance'and len(target_infos[j]['predecessors']) == 0]
                if potential_preds:
                    target_infos[i]['predecessors'] = random.sample(
                        potential_preds, min(1, len(potential_preds)))

            elif task_type == 'reconnaissance' and i > 0 and random.random() < 0.3:
                # 30%侦查任务需要电子战支援先完成
                potential_preds = [j for j in range(i)
                                   if target_infos[j]['task_type'] == 'electronic']
                if potential_preds:
                    target_infos[i]['predecessors'] = random.sample(
                        potential_preds, min(1, len(potential_preds)))

        # 第三步：创建Target对象（此时依赖关系已确定）
        for i in range(config.N_TARGETS):
            info = target_infos[i]
            target = Target(i, info['pos'], info['predecessors'])
            # 确保任务类型与生成依赖时一致（覆盖Target.__init__中的随机选择）
            target.task_type = info['task_type']
            self.targets.append(target)

        independent = sum(1 for t in self.targets if not t.predecessors)
        print(f"目标生成完成: {independent}/{config.N_TARGETS}个无依赖目标（确保初始可分配）")

        # 打印时序约束信息用于调试
        temporal_edges = [(t.predecessors, t.id) for t in self.targets if t.predecessors]
        if temporal_edges:
            print(f"Generated temporal constraints: {temporal_edges}")
            print(f"Targets with predecessors: {sum(1 for t in self.targets if t.predecessors)}/{config.N_TARGETS}")

    def _init_threats(self):
        """初始化威胁区域"""
        self.threats = []
        # 高地雷达
        for idx in np.argsort(self.terrain.elevs)[-3:]:
            center = self.terrain.geo2local(
                self.terrain.lons[idx],
                self.terrain.lats[idx],
                self.terrain.elevs[idx]
            )
            self.threats.append(Threat(
                center,
                random.uniform(8000, 15000),
                min(1.0, self.terrain.elevs[idx] / 100),
                'radar'
            ))
        # 随机禁飞区
        for _ in range(3):
            lon = random.uniform(self.terrain.lon_min, self.terrain.lon_max)
            lat = random.uniform(self.terrain.lat_min, self.terrain.lat_max)
            self.threats.append(Threat(
                self.terrain.geo2local(lon, lat, self.terrain.get_elev(lon, lat)),
                random.uniform(5000, 12000),
                1.5,
                'no_fly'
            ))

    def reset(self):
        """重置环境 - 重新生成目标位置确保多样性"""
        # 重置平台状态
        for p in self.platforms:
            p.reset()

        # 关键修复：重新随机生成目标位置（而非仅重置completed标志）
        # 保留原有依赖关系，但随机化位置和其他属性
        for i, t in enumerate(self.targets):
            t.reset()  # 重置completed和assigned_platform
            # 重新随机化位置（确保每轮episode环境不同）
            lon = random.uniform(self.terrain.lon_min + 0.01, self.terrain.lon_max - 0.01)
            lat = random.uniform(self.terrain.lat_min + 0.01, self.terrain.lat_max - 0.01)
            t.position = self.terrain.geo2local(lon, lat, self.terrain.get_elev(lon, lat))
            t.position[2] = 0
            # 可选：重新随机化其他属性增加多样性
            t.threat = random.uniform(0.1, 1.0)
            t.value = random.uniform(0.5, 1.0)

        self.completed_tasks.clear()
        self.current_time = 0.0

        # 调试：打印新生成的目标位置范围
        if random.random() < 0.01:  # 1%概率打印避免刷屏
            print(f"环境重置: 目标位置已重新随机化")


# ========== 1. 动态维度GNN编码器（替换原SimpleEncoder）==========
class GATEncoder(nn.Module):
    def __init__(self, hidden_dim=256, heads=4):
        super().__init__()
        self.W_u = nn.Linear(config.PLATFORM_DIM, hidden_dim)
        self.W_t = nn.Linear(config.TARGET_DIM, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        # 使用nn.Linear替代手工实现的注意力，可GPU加速
        self.attention_proj = nn.Linear(hidden_dim * 2, 1)

    def _compute_gat_attention(self, h_u, h_t, adj_matrix):
        """向量化GAT注意力计算"""
        n_p, n_t = h_u.size(0), h_t.size(0)

        # 使用广播机制替代expand
        h_u_exp = h_u.unsqueeze(1).expand(n_p, n_t, -1)  # [10, 16, 256]
        h_t_exp = h_t.unsqueeze(0).expand(n_p, n_t, -1)  # [10, 16, 256]

        concat = torch.cat([h_u_exp, h_t_exp], dim=-1)  # [10, 16, 512]

        # 使用Linear层替代matmul，更高效
        scores = self.attention_proj(concat).squeeze(-1)  # [10, 16]

        if adj_matrix is not None:
            scores = scores.masked_fill(adj_matrix == 0, float('-inf'))

            # 关键修复：检查哪些行全为 -inf（无可用目标），将其设为 0 避免 NaN
            # 如果一行全被 mask，则将该行所有分数设为 0（softmax 后变成均匀分布）
            valid_rows = (adj_matrix != 0).any(dim=-1, keepdim=True)  # [n_p, 1]
            scores = torch.where(valid_rows, scores, torch.zeros_like(scores))

        return F.softmax(F.leaky_relu(scores, 0.01), dim=-1)

    def forward(self, platform_states, target_states, adj_matrix):
        h_u = self.W_u(platform_states)  # [10, 256]
        h_t = self.W_t(target_states)  # [16, 256]

        attn = self._compute_gat_attention(h_u, h_t, adj_matrix)  # [10, 16]

        # 向量化GRU更新：批量处理所有平台
        # 计算每个平台的邻居聚合 [10, 16] @ [16, 256] = [10, 256]
        neighbor_agg = torch.matmul(attn, h_t)

        # 批量GRU更新（替代Python循环）
        h_u_new = self.gru(neighbor_agg, h_u)

        return h_u_new, h_t

class Actor(nn.Module):
    """支持动态动作维度的演员网络"""

    def __init__(self, action_dim=None):
        super().__init__()
        if action_dim is None:
            action_dim = config.ACTION_DIM

        self.net = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, action_dim)  # 动态输出维度
        )

    def forward(self, state, mask, temperature=1.0):
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            mask = mask.unsqueeze(0)
            squeeze = True

        logits = self.net(state)  # 输出可能是17维或更大

        # 如果网络输出和mask维度不匹配，进行截断/警告
        if logits.shape[-1] != mask.shape[-1]:
            min_dim = min(logits.shape[-1], mask.shape[-1])
            logits = logits[..., :min_dim]
            mask = mask[..., :min_dim]
            print(f"警告: Actor输出维度与mask不匹配，已自动对齐到{min_dim}")

        logits = logits.masked_fill(mask == 0, float('-inf'))
        probs = F.softmax(logits / temperature, dim=-1)

        if squeeze:
            probs = probs.squeeze(0)
        return probs

    def sample(self, state, mask, temperature=1.0):
        """采样动作"""
        probs = self.forward(state, mask, temperature)
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        if squeeze:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, action_dim=None, n_platforms=None):
        super().__init__()
        if action_dim is None:
            action_dim = config.ACTION_DIM
        if n_platforms is None:
            n_platforms = config.N_PLATFORMS  # 默认配置，但允许覆盖
        self.n_platforms = n_platforms
        self.action_dim = action_dim
        input_dim = n_platforms * config.HIDDEN_DIM + n_platforms * action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, 1)
        )

    def forward(self, states, actions):
        """
        states: [batch_size, n_platforms, hidden_dim]
        actions: [batch_size, n_platforms] 或 [batch_size, n_platforms, action_dim]
        """
        batch_size = states.size(0)

        # 动态获取实际维度（防御性编程）
        actual_n_platforms = states.size(1)  # 新增：记录实际平台数

        # 处理actions可能是one-hot（3维）或索引（2维）的情况
        if actions.dim() == 3:
            actual_action_dim = actions.size(2)  # 新增：提取实际动作维度
            actions_flat = actions.reshape(batch_size, -1)
        else:
            # 假设是索引，需要知道one-hot维度
            actual_action_dim = self.action_dim  # 新增：使用配置的动作维度
            actions_flat = actions.reshape(batch_size, -1)

        states_flat = states.reshape(batch_size, -1)
        expected_input = self.net[0].in_features  # 网络第一层期望的总输入维度
        actual_input = states_flat.shape[1] + actions_flat.shape[1]  # 实际输入总维度

        # 维度适配：填充或截断（确保与网络期望的输入维度一致）
        if actual_input != expected_input:
            # 分离状态和动作部分（关键修改：分别计算期望维度）
            expected_state_dim = self.n_platforms * config.HIDDEN_DIM  # 期望状态维度
            expected_action_dim = self.n_platforms * self.action_dim  # 期望动作维度

            # 适配状态部分（新增整块逻辑）
            if states_flat.shape[1] < expected_state_dim:
                padding = torch.zeros(batch_size, expected_state_dim - states_flat.shape[1],
                                      device=states.device, dtype=states.dtype)
                states_flat = torch.cat([states_flat, padding], dim=1)
            elif states_flat.shape[1] > expected_state_dim:
                states_flat = states_flat[:, :expected_state_dim]

            # 适配动作部分（保留原逻辑但优化）
            if actions_flat.shape[1] < expected_action_dim:
                padding = torch.zeros(batch_size, expected_action_dim - actions_flat.shape[1],
                                      device=actions.device, dtype=actions.dtype)
                actions_flat = torch.cat([actions_flat, padding], dim=1)
            elif actions_flat.shape[1] > expected_action_dim:
                actions_flat = actions_flat[:, :expected_action_dim]

        x = torch.cat([states_flat, actions_flat], dim=-1)
        return self.net(x)


class MASAC:
    """多智能体SAC算法"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.LR_ACTOR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)

        self.Transition = namedtuple('Transition', ['states', 'actions', 'rewards', 'next_states', 'dones'])
        self.buffer = deque(maxlen=config.BUFFER_SIZE)
        self.epsilon = config.EPSILON_START

    def select_action(self, state: np.ndarray, mask: np.ndarray, evaluate: bool = False,
                      temperature=1.0) -> int:
        """选择动作 - 支持动态动作维度"""
        state = torch.FloatTensor(state).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        if self.epsilon > 0.4:  # 探索阶段
            temperature = config.ACTION_NOISE_TEMP  # 1.5，增加随机性
        else:
            temperature = 0.8  # 利用阶段，更确定
        # 从mask获取实际动作维度
        actual_action_dim = mask.shape[0]

        with torch.no_grad():
            if not evaluate and random.random() < self.epsilon:
                # 探索阶段：混合策略（50%随机，50%按概率分布）
                if random.random() < 0.5:
                    # 纯随机
                    available_actions = torch.where(mask > 0)[0]
                    if len(available_actions) > 0:
                        action = random.choice(available_actions.cpu().numpy()).item()
                    else:
                        action = actual_action_dim - 1
                else:
                    # 按概率分布采样（带温度）
                    probs = self.actor(state, mask, temperature)
                    dist = Categorical(probs)
                    action = dist.sample().item()
            else:
                # 利用阶段
                probs = self.actor(state, mask, temperature)
                if evaluate:
                    action = torch.argmax(probs).item()
                else:
                    action, _ = self.actor.sample(state, mask, temperature)
                    action = action.item()
        return action

    def safe_select_action(self, state: np.ndarray, mask: np.ndarray,
                           temperature: float = 1.0) -> int:
        """安全动作选择 - 处理维度不匹配和NaN问题"""
        state = torch.FloatTensor(state).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)

        with torch.no_grad():
            # 动态获取当前网络输出的实际维度
            if state.dim() == 1:
                state_batch = state.unsqueeze(0)
            else:
                state_batch = state

            raw_logits = self.actor.net(state_batch)
            actual_dim = raw_logits.shape[-1]

            # 动态调整mask以匹配网络输出
            current_mask_dim = mask.shape[-1]
            if current_mask_dim != actual_dim:
                if current_mask_dim < actual_dim:
                    # mask太短，补1（允许选择新目标）
                    padding = torch.ones(actual_dim - current_mask_dim, device=self.device)
                    mask = torch.cat([mask, padding])
                else:
                    # mask太长，截断
                    mask = mask[:actual_dim]

            # 应用mask
            logits = raw_logits.masked_fill(mask == 0, float('-inf'))

            # 检查是否全被mask
            if torch.all(logits == float('-inf')):
                return actual_dim - 1  # 最后一个动作为Idle

            # 计算概率并检查NaN
            probs = F.softmax(logits / temperature, dim=-1)
            if torch.isnan(probs).any() or torch.sum(probs) < 0.99:  # 允许小误差
                # 回退到均匀分布（仅对可用动作）
                available = (mask > 0).float()
                probs = available / (torch.sum(available) + 1e-8)

            # 采样
            dist = Categorical(probs)
            action = dist.sample()

            return action.item() if action.dim() == 0 else action[0].item()

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)

    def store(self, joint_states, joint_actions, reward, joint_next_states, done):
        """存储经验"""
        self.buffer.append(self.Transition(joint_states, joint_actions, reward, joint_next_states, done))

    def update(self):
        """更新网络"""
        if len(self.buffer) < config.BATCH_SIZE:
            return {}

        batch = random.sample(self.buffer, config.BATCH_SIZE)
        if len(batch) == 0:  # 添加空检查
            return {}

        states = torch.FloatTensor(np.array([t.states for t in batch])).to(self.device)
        actions = torch.LongTensor(np.array([t.actions for t in batch])).to(self.device)
        if actions.numel() == 0:  # 添加空张量检查
            return {}
        rewards = torch.FloatTensor([t.rewards for t in batch]).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_states for t in batch])).to(self.device)
        dones = torch.FloatTensor([t.dones for t in batch]).unsqueeze(-1).to(self.device)

        max_action = config.ACTION_DIM  # 固定为17（16目标+1idle）

        # Critic当前Q值计算
        actions_onehot = F.one_hot(actions, num_classes=max_action).float()
        current_q = self.critic(states, actions_onehot)

        with torch.no_grad():
            next_actions = []
            for i in range(config.N_PLATFORMS):
                mask = torch.ones(config.BATCH_SIZE, max_action, device=self.device)  # 修改为max_action
                probs = self.actor_target(next_states[:, i, :], mask)
                dist = Categorical(probs)
                action = dist.sample()
                next_actions.append(action)
            next_actions = torch.stack(next_actions, dim=1)
            next_actions_onehot = F.one_hot(next_actions, num_classes=max_action)  # 统一使用max_action
            next_q = self.critic_target(next_states, next_actions_onehot)
            target_q = rewards + config.GAMMA * (1 - dones) * next_q

        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        new_actions_onehot = []
        for i in range(config.N_PLATFORMS):
            mask = torch.ones(config.BATCH_SIZE, config.ACTION_DIM, device=self.device)
            probs = self.actor(states[:, i, :], mask)  # [batch, action_dim]

            # Gumbel-Softmax：前向看起来是one_hot，但梯度走概率分布路径
            logits = torch.log(probs + 1e-10)
            gumbel = F.gumbel_softmax(logits, tau=0.2, hard=True)
            new_actions_onehot.append(gumbel)

        new_actions_onehot = torch.stack(new_actions_onehot, dim=1)  # [batch, n_platforms, action_dim]
        new_q = self.critic(states, new_actions_onehot)
        actor_loss = -new_q.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        for target, source in [(self.critic_target, self.critic), (self.actor_target, self.actor)]:
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(config.TAU * param.data + (1 - config.TAU) * target_param.data)

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}


# ========== 2. 逆强化学习奖励学习器（新增）==========
class IRLRewardLearner(nn.Module):
    """基于最大熵IRL的奖励学习网络"""

    def __init__(self, hidden_dim=256, action_dim=None):
        super().__init__()
        if action_dim is None:
            action_dim = config.ACTION_DIM
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # 使用原始特征维度而非GNN编码后的维度
        # 平台状态11维 + 目标状态16维 + 动作one-hot
        input_dim = config.PLATFORM_DIM + config.TARGET_DIM + action_dim

        # 从状态-动作特征推断奖励（替代手工设计的reward）
        self.reward_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def infer_reward(self, platform_state, target_state, action_onehot):
        """推断专家行为的隐式奖励"""
        combined = torch.cat([platform_state, target_state, action_onehot], dim=-1)
        return self.reward_net(combined)

    def compute_advantage(self, states, actions, next_states):
        """计算优势函数用于策略更新"""
        current_r = self.reward_net(states)
        next_r = self.reward_net(next_states)
        return current_r + config.GAMMA * next_r


class ConvergenceMonitor:
    """收敛监控器 - 实时检测训练平台期"""

    def __init__(self):
        self.reward_history = deque(maxlen=config.CONVERGENCE_WINDOW)
        self.best_reward = -1e10
        self.best_episode = 0
        self.patience_counter = 0
        self.converged = False
        self.convergence_episode = None
        self.start_time = time.time()
        self.episode_times = deque(maxlen=100)
        self.plateau_counter = 0  # 平台期计数器
        self.PLATEAU_WINDOW = 50  # 检查最近50回合
        self.PLATEAU_THRESHOLD = 0.005  # 变化小于0.5%视为平台期
        self.plateau_history = deque(maxlen=self.PLATEAU_WINDOW)
        self.last_checkpoint_mean = 0.0
        self.CONVERGENCE_WINDOW = config.CONVERGENCE_WINDOW  # 添加这行
        self.last_window_mean = 0.0  # 添加这行（在update方法底部使用）

    def update(self, episode: int, reward: float, n_assigned: int, assignment: Dict[int, int]) -> dict:
        # 初始化状态字典（确保包含新键）
        """
        更新监控状态 - 强制要求全平台分配
        返回: 状态字典，包含是否收敛、当前统计等
        """
        self.reward_history.append(reward)
        self.episode_times.append(time.time())
        self.plateau_history.append(reward)  # 新增：记录平台期检测历史

        # 真正分配了任务的平台数（排除空列表）
        n_platforms_assigned = len([pid for pid, tids in assignment.items()
                                    if isinstance(tids, list) and len(tids) > 0]) if assignment else 0

        # 真正覆盖的唯一目标数
        all_tids = []
        for tids in assignment.values():
            if isinstance(tids, list):
                all_tids.extend(tids)
            else:
                all_tids.append(tids)
        n_targets_assigned = len(set(all_tids))

        # 以目标全覆盖为判断标准（你的核心需求）
        full_coverage = (n_targets_assigned == config.N_TARGETS)

        # 初始化状态字典
        status = {
            'episode': episode,
            'current_reward': reward,
            'converged': False,
            'should_stop': False,
            'best_reward': self.best_reward,
            'progress': 0.0,
            'eta_seconds': 0,
            'n_platforms_assigned': n_platforms_assigned,
            'n_targets_assigned': n_targets_assigned,
            'full_coverage': full_coverage,
            'new_best': False,
            'window_mean': 0.0,
            'window_std': 0.0
        }

        # 记录历史
        self.reward_history.append(reward)
        self.episode_times.append(time.time())

        # 计算窗口统计
        if len(self.reward_history) > 0:
            current_mean = np.mean(list(self.reward_history))
            current_std = np.std(list(self.reward_history)) if len(self.reward_history) > 1 else 0.0
            status['window_mean'] = current_mean
            status['window_std'] = current_std

        # 最佳奖励更新（不再检查是否全分配！只看奖励高低）
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_episode = episode
            status['new_best'] = True
            self.patience_counter = 0
            print(f"🏆 回合{episode}: 新最佳模型! 奖励{reward:.3f}")
        else:
            self.patience_counter += 1

        # 收敛判断（只看出奖励是否长期不提升）
        if self.patience_counter >= config.PATIENCE:
            if episode >= config.MIN_EPISODES:
                self.converged = True
                status['converged'] = True
                status['should_stop'] = True
                status['stop_reason'] = f"连续{config.PATIENCE}个窗口无改进，已收敛"

        # 预计剩余时间
        if len(self.episode_times) > 1:
            avg_time = np.mean(np.diff(self.episode_times))
            status['eta_seconds'] = avg_time * (config.MAX_EPISODES - episode)
        is_plateau = False
        if len(self.plateau_history) >= self.PLATEAU_WINDOW:
            recent = list(self.plateau_history)
            mean_r = np.mean(recent)
            std_r = np.std(recent)
            # 标准差极小且不是负奖励崩溃状态
            if std_r < self.PLATEAU_THRESHOLD * (abs(mean_r) + 1e-3) and mean_r > -5:
                is_plateau = True
                self.plateau_counter += 1
            else:
                self.plateau_counter = 0

        # 崩溃检测：奖励极低或覆盖率不到一半
        is_collapse = (reward < -15) or (n_platforms_assigned < config.N_PLATFORMS * 0.5)

        status['is_plateau'] = is_plateau
        status['is_collapse'] = is_collapse
        status['plateau_counter'] = self.plateau_counter
        return status

    def get_trend(self) -> str:
        """判断近期趋势（上升/下降/平台）"""
        if len(self.reward_history) < 20:
            return "收集数据中..."

        # 将窗口分为两半比较
        half = len(self.reward_history) // 2
        first_half = np.mean(list(self.reward_history)[:half])
        second_half = np.mean(list(self.reward_history)[half:])

        diff = (second_half - first_half) / (abs(first_half) + 1e-6)

        if diff > 0.05:
            return f"↗ 上升({diff:.1%})"
        elif diff < -0.05:
            return f"↘ 下降({diff:.1%})"
        else:
            return f"→ 平台({diff:.1%})"

class TaskSystem:
    def __init__(self, env):
        self.env = env
        # 替换为动态编码器（支持变长输入）
        self.encoder = GATEncoder(config.HIDDEN_DIM)  # 修改这里
        self.masac = MASAC()
        self.device = self.masac.device
        self.encoder.to(self.device)
        self.route_est = RouteEstimator(self.env.terrain, self.env.threats)
        self.route_planner = RoutePlanner(self.route_est, self.env.terrain, self.env.threats)
        from collections import OrderedDict

        class LimitedSizeDict(OrderedDict):
            def __init__(self, size_limit=None):
                super().__init__()
                self.size_limit = size_limit

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if self.size_limit and len(self) > self.size_limit:
                    # 移除最旧的项
                    self.popitem(last=False)

        # 修改这里：限制缓存5000条（足够8-12平台×14-18目标使用）
        self._distance_cache = LimitedSizeDict(size_limit=5000)

        # 添加奖励归一化统计
        from collections import deque
        self.reward_history = deque(maxlen=config.REWARD_NORM_WINDOW)
        self.reward_mean = 0.0
        self.reward_std = 1.0

        # 诊断计数器
        self.diag = {
            'empty_assignments': 0,
            'duplicate_hits': 0,
            'constraint_violations': 0,
            'valid_rewards': 0
        }

        self.stats = {
            'violations': 0,
            'duplicates': 0,
            'times': [],
            'duplicate_history': [],
            'temporal_violations': 0  # 新增：时序违反统计
        }
        self.last_J_v = 0.0
        self.last_J_e = 0.0
        self.last_J_a = 0.0
        self.last_obj = 0.0

    def get_action_mask(self, assigned_targets: Set[int], completed_targets: Set[int],
                        current_platform_id: int, alive_platforms: List[int] = None,
                        current_assignment: Dict[int, List[int]] = None) -> torch.Tensor:
        """增强版动作掩码，支持指定可用平台"""
        n_targets = len(self.env.targets)
        action_dim = n_targets + 1
        mask = torch.zeros(action_dim, device=self.device)

        # 检查平台是否存活
        if alive_platforms is not None and current_platform_id not in alive_platforms:
            mask[-1] = 1.0  # 强制idle
            return mask

        if current_platform_id >= len(self.env.platforms):
            mask[-1] = 1.0
            return mask

        p = self.env.platforms[current_platform_id]
        rejected_reasons = {'assigned': 0, 'temporal': 0, 'completed': 0, 'payload': 0, 'capability': 0, 'time_window': 0, 'range': 0}

        for j, t in enumerate(self.env.targets):
            # 1. 已分配目标排除
            if j in assigned_targets:
                rejected_reasons['assigned'] += 1
                mask[j] = 0.0
                continue

            # 2. 时序约束：前序任务必须已完成（基于本地completed_targets）
            if t.predecessors:
                if not all(pid in completed_targets for pid in t.predecessors):
                    rejected_reasons['temporal'] += 1
                    mask[j] = 0.0
                    continue

            # 3. 关键修复：使用completed_targets（本次分配状态）而非t.completed（全局状态）
            if j in completed_targets:
                rejected_reasons['completed'] += 1
                continue

            if not p.can_do(t.task_type):
                rejected_reasons['capability'] += 1
                continue
            # ===== 时间窗口约束（仅对打击任务生效） =====
            # 统一计算当前平台到目标的距离（时间窗口和航程约束都需要）
            dist = np.linalg.norm(p.position - t.position)

            # ===== 时间窗口约束（仅对打击任务生效） =====
            if t.task_type == '打击':
                arrival_time = dist / max(p.speed, 1e-6)
                # 只约束最晚到达时间，允许提前到达（盘旋等待）
                if arrival_time > t.time_win_end:
                    rejected_reasons['time_window'] = rejected_reasons.get('time_window', 0) + 1
                    mask[j] = 0.0
                    continue

            # ===== 航程约束（累加已分配航段） =====
            total_dist = dist  # 当前这段距离

            # 如果该平台已有分配任务，累加之前的航段
            if current_assignment and current_platform_id in current_assignment:
                prev_pos = p.position
                for prev_tid in current_assignment[current_platform_id]:
                    if prev_tid < len(self.env.targets):
                        prev_target = self.env.targets[prev_tid]
                        total_dist += np.linalg.norm(prev_pos - prev_target.position)
                        prev_pos = prev_target.position

            if total_dist > p.max_range:
                rejected_reasons['range'] = rejected_reasons.get('range', 0) + 1
                mask[j] = 0.0
                continue
            mask[j] = 1.0  # 现在 j 最大为 n_targets-1，mask 大小为 n_targets+1，不会越界

            # 空闲动作始终可用（最后一个位置）
        mask[-1] = 1.0



        # 调试输出
        if current_platform_id < 1 and sum(mask[:-1]) == 0 and random.random() < 0.01:
            print(f"平台{current_platform_id}: 无可选目标! 统计: {rejected_reasons}")
            print(f"  completed_targets集合: {completed_targets}")

        return mask

    def get_action_mask_emergency(self, assigned_targets: Set[int], completed_targets: Set[int],
                                    current_platform_id: int) -> torch.Tensor:
        """应急情况下的动作掩码 - 更严格的约束检查"""
        # 应急情况下使用标准掩码逻辑，但可以更严格
        # 目前直接复用 get_action_mask 的逻辑
        return self.get_action_mask(assigned_targets, completed_targets, current_platform_id, None)

    def assign(self, evaluate: bool = False) -> Tuple[Dict[int, List[int]], List[int], np.ndarray]:
        """执行任务分配 - 两轮制：第一轮保底全覆盖，第二轮追加"""
        start_time = time.time()

        # 获取图数据
        platform_states, target_states, base_mask, temporal_matrix = self.env.get_graph_data()
        platform_states = platform_states.to(self.device)
        target_states = target_states.to(self.device)
        base_mask = base_mask.to(self.device)

        with torch.no_grad():
            hu, ht = self.encoder(platform_states, target_states, base_mask)

        # 初始化
        assignment: Dict[int, List[int]] = {i: [] for i in range(config.N_PLATFORMS)}
        assigned_targets = set()
        completed_targets = set()
        n_targets = len(self.env.targets)
        action_dim = n_targets + 1

        # 平台排序：侦察/打击优先，察打一体靠后
        platform_order = sorted(range(config.N_PLATFORMS),
                                key=lambda pid: 0 if self.env.platforms[pid].type in ['侦察', '打击'] else 1)
        critical_recon_ids = set()
        for t in self.env.targets:
            if t.task_type == '打击' and t.predecessors:
                critical_recon_ids.update(t.predecessors)

        # 按目标ID排序，确保稳定分配
        for crit_tid in sorted(critical_recon_ids):
            if crit_tid in assigned_targets:
                continue  # 已被预分配

            t = self.env.targets[crit_tid]
            candidates = []
            for pid in range(config.N_PLATFORMS):
                if len(assignment[pid]) > 0:
                    continue
                p = self.env.platforms[pid]
                if not p.can_do('侦察'):
                    continue
                # 【关键修改】预分配阶段跳过纯侦察平台，留给第一轮就近选择家门口目标
                # 理由：纯侦察平台只能做侦察，如果被绑去远处，家门口的纯侦察目标就空了
                if p.type == '侦察':
                    continue
                dist = np.linalg.norm(p.position - t.position)
                if dist > p.max_range:
                    continue
                candidates.append((dist, pid))

            if candidates:
                # 只按距离升序（越近的察打一体平台越优先）
                candidates.sort(key=lambda x: x[0])
                best_pid = candidates[0][1]

                assignment[best_pid].append(crit_tid)
                assigned_targets.add(crit_tid)
                completed_targets.add(crit_tid)
                self.env.mark_target_completed(crit_tid)

        # ========== 第一轮：每个平台只分配1个目标（确保12/12全覆盖）==========
        for i in platform_order:
            # 已有预分配任务的平台，不再抢占其他稀缺资源
            if len(assignment[i]) > 0:
                continue
            platform = self.env.platforms[i]

            action_mask = self.get_action_mask(
                assigned_targets, completed_targets, i,
                current_assignment=assignment
            ).cpu().numpy()

            # 无可选目标则跳过
            if np.sum(action_mask[:-1]) == 0:
                continue

            state = hu[i].cpu().numpy()
            temperature = 0.5 if evaluate else 1.0
            action = self.masac.select_action(state, action_mask, evaluate, temperature)

            if action < len(self.env.targets):
                valid_tids = np.where(action_mask[:-1] == 1)[0]
                valid_tids = [tid for tid in valid_tids if tid not in assigned_targets and tid != action]

                if len(valid_tids) > 0:
                    current_dist = np.linalg.norm(platform.position - self.env.targets[action].position)
                    nearest_tid = min(valid_tids,
                                      key=lambda tid: np.linalg.norm(
                                          platform.position - self.env.targets[tid].position))
                    nearest_dist = np.linalg.norm(platform.position - self.env.targets[nearest_tid].position)

                    # 替换条件：最近目标明显更近（<60% 或 绝对差>20km）
                    if nearest_dist < current_dist * 0.6 or (current_dist - nearest_dist) > 30e3:
                        action = nearest_tid

            if action < len(self.env.targets):
                target = self.env.targets[action]

                # 检查约束
                can_assign = True
                if target.predecessors and not all(pid in completed_targets for pid in target.predecessors):
                    can_assign = False
                if action in assigned_targets:
                    can_assign = False
                if target.req_payload > platform.payload:
                    can_assign = False

                if can_assign:
                    assignment[i].append(action)
                    assigned_targets.add(action)
                    completed_targets.add(action)
                    self.env.mark_target_completed(action)

                else:
                    # 网络选了非法目标，强制纠正为最近合法目标
                    available = np.where(action_mask[:-1] == 1)[0]
                    valid_alts = [tid for tid in available if tid not in assigned_targets]
                    if len(valid_alts) > 0:
                        best_tid = min(valid_alts,
                                       key=lambda tid: np.linalg.norm(
                                           platform.position - self.env.targets[tid].position))
                        alt_target = self.env.targets[best_tid]
                        assignment[i].append(best_tid)
                        assigned_targets.add(best_tid)
                        completed_targets.add(best_tid)
                        self.env.mark_target_completed(best_tid)
            else:
                # 网络选idle，但还有可用目标，强制分配最近合法目标
                available = np.where(action_mask[:-1] == 1)[0]
                valid_alts = [tid for tid in available if tid not in assigned_targets]
                if len(valid_alts) > 0:
                    best_tid = min(valid_alts,
                                   key=lambda tid: np.linalg.norm(
                                       platform.position - self.env.targets[tid].position))
                    alt_target = self.env.targets[best_tid]
                    assignment[i].append(best_tid)
                    assigned_targets.add(best_tid)
                    completed_targets.add(best_tid)
                    self.env.mark_target_completed(best_tid)

        # ========== 第二轮：有剩余载荷的平台继续分配额外目标 ==========
        for i in platform_order:
            platform = self.env.platforms[i]
            used_payload = sum(self.env.targets[tid].req_payload for tid in assignment[i])
            remaining_payload = platform.payload - used_payload

            while remaining_payload > 0 and len(assignment[i]) < 3:
                state = hu[i].cpu().numpy()

                action_mask = self.get_action_mask(
                    assigned_targets, completed_targets, i,
                    current_assignment=assignment
                ).cpu().numpy()

                if np.sum(action_mask[:-1]) == 0:
                    break

                temperature = 0.5 if evaluate else 1.0
                action = self.masac.select_action(state, action_mask, evaluate, temperature)

                if action < len(self.env.targets):
                    valid_tids = np.where(action_mask[:-1] == 1)[0]
                    valid_tids = [tid for tid in valid_tids if tid not in assigned_targets and tid != action]

                    if len(valid_tids) > 0:
                        # 第二轮：从当前末端位置计算距离（如果已有任务）
                        if assignment[i]:
                            prev_tid = assignment[i][-1]
                            if prev_tid < len(self.env.targets):
                                ref_pos = self.env.targets[prev_tid].position
                            else:
                                ref_pos = platform.position
                        else:
                            ref_pos = platform.position

                        current_dist = np.linalg.norm(ref_pos - self.env.targets[action].position)
                        nearest_tid = min(valid_tids,
                                          key=lambda tid: np.linalg.norm(ref_pos - self.env.targets[tid].position))
                        nearest_dist = np.linalg.norm(ref_pos - self.env.targets[nearest_tid].position)

                        if nearest_dist < current_dist * 0.6 or (current_dist - nearest_dist) > 20e3:
                            action = nearest_tid

                if action < len(self.env.targets):
                    target = self.env.targets[action]

                    can_assign = True
                    if target.predecessors and not all(pid in completed_targets for pid in target.predecessors):
                        can_assign = False
                    elif target.req_payload > remaining_payload:
                        can_assign = False
                    elif action in assigned_targets:
                        can_assign = False

                    if can_assign:
                        assignment[i].append(action)
                        assigned_targets.add(action)
                        completed_targets.add(action)
                        self.env.mark_target_completed(action)
                        remaining_payload -= target.req_payload
                    else:
                        # 尝试找一个小载荷替代目标
                        if target.req_payload > remaining_payload:
                            found = False
                            for alt_tid in range(n_targets):
                                if (action_mask[alt_tid] == 1 and
                                        alt_tid not in assigned_targets and
                                        self.env.targets[alt_tid].req_payload <= remaining_payload):
                                    alt_target = self.env.targets[alt_tid]
                                    assignment[i].append(alt_tid)
                                    assigned_targets.add(alt_tid)
                                    completed_targets.add(alt_tid)
                                    self.env.mark_target_completed(alt_tid)
                                    remaining_payload -= alt_target.req_payload
                                    found = True
                                    break
                            if not found:
                                break
                        else:
                            # ===== 修改3：训练引导（网络选的目标无效，强制纠正）=====
                            if not evaluate and remaining_payload > 0:
                                available = np.where(action_mask[:-1] == 1)[0]
                                valid_alts = [tid for tid in available
                                              if tid not in assigned_targets
                                              and self.env.targets[tid].req_payload <= remaining_payload]
                                if len(valid_alts) > 0:
                                    best_tid = min(valid_alts,
                                                   key=lambda tid: np.linalg.norm(
                                                       platform.position - self.env.targets[tid].position))
                                    alt_target = self.env.targets[best_tid]
                                    assignment[i].append(best_tid)
                                    assigned_targets.add(best_tid)
                                    completed_targets.add(best_tid)
                                    self.env.mark_target_completed(best_tid)
                                    remaining_payload -= alt_target.req_payload
                                    continue  # 继续while循环，尝试再分一个
                            break
                else:
                    # 网络选了idle，用距离最近合法目标兜底（训练+评估都生效）
                    if remaining_payload > 0:
                        available = np.where(action_mask[:-1] == 1)[0]
                        if len(available) > 0:
                            best_tid = min(available,
                                           key=lambda tid: np.linalg.norm(
                                               platform.position - self.env.targets[tid].position))
                            alt_target = self.env.targets[best_tid]
                            if alt_target.req_payload <= remaining_payload:
                                assignment[i].append(best_tid)
                                assigned_targets.add(best_tid)
                                completed_targets.add(best_tid)
                                self.env.mark_target_completed(best_tid)
                                remaining_payload -= alt_target.req_payload
                                continue  # 继续while循环，尝试再分一个
                    break

        # ========== 保底循环（安全网）==========
        for i in range(config.N_PLATFORMS):
            if len(assignment[i]) > 0:
                continue

            action_mask = self.get_action_mask(
                assigned_targets, completed_targets, i,
                current_assignment=assignment
            ).cpu().numpy()

            available_targets = np.where(action_mask[:-1] == 1)[0]
            if len(available_targets) == 0:
                continue

            p = self.env.platforms[i]
            best_tid = None
            best_dist = float('inf')
            for tid in available_targets:
                t = self.env.targets[tid]
                dist = np.linalg.norm(p.position - t.position)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                assignment[i].append(best_tid)
                assigned_targets.add(best_tid)
                completed_targets.add(best_tid)
                self.env.mark_target_completed(best_tid)

        all_target_ids = set(range(len(self.env.targets)))
        missing_targets = sorted(all_target_ids - assigned_targets)

        for tid in missing_targets:
            t = self.env.targets[tid]
            candidates = []

            for pid in range(config.N_PLATFORMS):
                p = self.env.platforms[pid]

                # 1. 类型约束
                if not p.can_do(t.task_type):
                    continue

                # 2. 载荷约束（累加已分配任务的用弹量）
                used_payload = sum(self.env.targets[x].req_payload for x in assignment[pid])
                if used_payload + t.req_payload > p.payload:
                    continue

                # 3. 时序约束
                if t.predecessors and not all(p_id in completed_targets for p_id in t.predecessors):
                    continue

                # 4. 航程约束（从当前末端位置到目标）
                if assignment[pid]:
                    prev_pos = self.env.targets[assignment[pid][-1]].position
                else:
                    prev_pos = p.position
                if np.linalg.norm(prev_pos - t.position) > p.max_range:
                    continue

                dist = np.linalg.norm(p.position - t.position)
                candidates.append((dist, pid))

            if candidates:
                candidates.sort()  # 按距离升序
                best_pid = candidates[0][1]
                assignment[best_pid].append(tid)
                assigned_targets.add(tid)
                completed_targets.add(tid)
                self.env.mark_target_completed(tid)
                print(f"补分: P{best_pid}({self.env.platforms[best_pid].type}) <- T{tid}({t.task_type})")

        # 重建actions列表（每个平台只记录第一个任务，用于训练）
        actions = []
        for pid in range(config.N_PLATFORMS):
            if assignment[pid]:
                actions.append(assignment[pid][0])
            else:
                actions.append(action_dim - 1)

        elapsed = time.time() - start_time
        self.stats['times'].append(elapsed)

        return assignment, actions, hu.cpu().numpy()

    def calc_reward(self, assignment: Dict[int, List[int]], completed_snapshot: set = None) -> float:
        """计算奖励 - 由航程代价J_v、时间代价J_e、任务收益J_a三部分组成"""
        if not assignment or all(len(tasks) == 0 for tasks in assignment.values()):
            return -10.0

        # ========== 1. 航程代价 J_v（归一化）==========
        J_v = 0.0
        map_diag = np.sqrt(config.MAP_SIZE_X ** 2 + config.MAP_SIZE_Y ** 2)

        # ========== 2. 时间代价 J_e（取所有平台最大完成时间，归一化）==========
        J_e = 0.0

        # ========== 3. 任务收益 J_a（与覆盖目标的价值、数量、匹配度正相关）==========
        J_a = 0.0
        assigned_target_values = []

        for pid, tids in assignment.items():
            if pid >= len(self.env.platforms):
                continue
            p = self.env.platforms[pid]

            prev_pos = p.position
            platform_dist = 0.0

            for tid in tids:
                if tid >= len(self.env.targets):
                    continue
                t = self.env.targets[tid]

                # J_v 项：航程估计（含地形+威胁惩罚，对应申请书公式3.3）
                dist = self.route_est.estimate_distance(prev_pos, t.position)
                platform_dist += dist
                prev_pos = t.position

                # J_a 项：收集该目标的价值和类型匹配收益
                target_value = t.value
                if p.type == t.task_type:
                    match_bonus = 1.0  # 完全匹配，高收益
                elif p.type == '察打一体' and t.task_type in ['侦察', '打击']:
                    match_bonus = 0.6  # 部分匹配，中收益
                else:
                    match_bonus = 0.2  # 勉强匹配，低收益
                assigned_target_values.append(target_value + match_bonus)

            # 归一化航程代价
            J_v += platform_dist / (map_diag + 1e-6)

            # 归一化时间代价（取所有平台的最大值）
            platform_time = platform_dist / max(p.speed, 1e-6)
            J_e = max(J_e, platform_time / 10000.0)

        # ========== 统计覆盖情况（只算一次）==========
        all_assigned_targets = []
        for tids in assignment.values():
            all_assigned_targets.extend(tids)
        unique_targets = len(set(all_assigned_targets))
        n_targets = len(self.env.targets)

        # 【新增】统计有多少平台实际分到了任务
        n_platforms_assigned = len([pid for pid, tids in assignment.items() if len(tids) > 0])
        n_platforms = len(self.env.platforms)

        # ========== 核心修改：覆盖率基础奖励（大幅提高）==========
        n_platforms = len(self.env.platforms)
        n_targets = len(self.env.targets)

        if unique_targets >= n_targets and n_platforms_assigned >= n_platforms:
            coverage_reward = 8.0  # 从80降到8
        elif unique_targets >= n_targets:
            coverage_reward = 5.0  # 从50降到5
        elif n_platforms_assigned >= n_platforms:
            coverage_reward = 3.0  # 从30降到3
        else:
            coverage_reward = unique_targets * 0.5  # 从3降到0.5

        platform_coverage_reward = n_platforms_assigned * 0.5  # 从5降到0.5

        n_uncovered = n_targets - unique_targets
        uncovered_penalty = n_uncovered * 40.0  # 每个未覆盖目标只罚20

        # 任务收益（与目标价值、类型匹配度相关）
        total_value = sum(assigned_target_values) if assigned_target_values else 0.0
        J_a = total_value * 10.0  # 从3.0提高到4.0，整体放大收益

        if unique_targets >= n_targets and n_platforms_assigned >= n_platforms:
            J_a += 30.0  # 从150降到30
        elif unique_targets >= n_targets:
            J_a += 15.0  # 从80降到15
        elif unique_targets >= n_targets * 0.9:
            J_a += 5.0  # 从30降到5

        # 航程代价 J_v、时间代价 J_e（适度惩罚，不要压过收益）
        cost = J_v * 10.0 + J_e * 3.0

        # ========== 约束违反惩罚（硬约束）==========
        penalty = 0.0

        # 1. 重复分配惩罚
        duplicates = len(all_assigned_targets) - len(set(all_assigned_targets))
        if duplicates > 0:
            penalty += duplicates * 50.0

        # 2. 时序约束惩罚
        temporal_violations = 0
        for pid, tids in assignment.items():
            for tid in tids:
                target = self.env.targets[tid]
                if target.predecessors:
                    check_set = completed_snapshot if completed_snapshot is not None else self.env.completed_tasks
                    if not all(p_id in check_set for p_id in target.predecessors):
                        temporal_violations += 1
        penalty += temporal_violations * 30.0

        # 3. 时间窗口约束 & 航程约束惩罚
        time_window_violations = 0
        range_violations = 0
        for pid, tids in assignment.items():
            if pid >= len(self.env.platforms):
                continue
            p = self.env.platforms[pid]
            prev_pos = p.position
            cumulative_dist = 0.0
            for tid in tids:
                if tid >= len(self.env.targets):
                    continue
                t = self.env.targets[tid]
                seg_dist = np.linalg.norm(prev_pos - t.position)
                cumulative_dist += seg_dist
                prev_pos = t.position
                if t.task_type == '打击':
                    arrival_time = cumulative_dist / max(p.speed, 1e-6)
                    if not (t.time_win_start <= arrival_time <= t.time_win_end):
                        time_window_violations += 1
            if cumulative_dist > p.max_range:
                range_violations += 1

        penalty += time_window_violations * 15.0
        penalty += range_violations * 25.0
        task_counts = [len(tids) for tids in assignment.values()]
        if len(task_counts) > 1:
            mean_tasks = np.mean(task_counts)
            std_tasks = np.std(task_counts)
            # 标准差>1.0时开始惩罚（比如1个平台4个任务、其他1个，标准差≈1.3）
            if std_tasks > 1.0:
                imbalance_penalty = (std_tasks - 1.0) ** 2 * 30.0  # 二次增长，惩罚力度大
            else:
                imbalance_penalty = 0.0
        else:
            imbalance_penalty = 0.0
        n_platforms_with_tasks = len([pid for pid, tids in assignment.items() if len(tids) > 0])

        if unique_targets >= n_targets and n_platforms_with_tasks >= n_platforms:
            perfect_bonus = 10.0
        else:
            perfect_bonus = 0.0

            # 空闲平台惩罚（只保留这一处，删除后面重复的定义）
        n_idle = n_platforms - n_platforms_assigned
        if n_idle > 0:
            idle_penalty = 50.0 * n_idle  # 线性惩罚，不再指数增长
        else:
            idle_penalty = 0.0
        # ========== 最终奖励：覆盖率奖励 + 平台覆盖奖励 + 收益 - 代价 - 惩罚 - 空闲惩罚 ==========
        if unique_targets == 0:
            final_reward = -10.0
        else:
            final_reward = (coverage_reward + platform_coverage_reward + J_a
                            - cost - penalty - idle_penalty - uncovered_penalty
                            - imbalance_penalty
                            + perfect_bonus)

        final_reward = max(final_reward, -50.0)

        return final_reward

    def train_step(self):
        """执行一步训练 - 多任务版本"""
        assignment, actions, joint_states = self.assign(evaluate=False)
        completed_snapshot = set(self.env.completed_tasks)
        raw_reward = self.calc_reward(assignment, completed_snapshot)

        # 奖励裁剪与缩放，防止Critic损失爆炸（必须放在reward定义之后！）
        clipped_reward = float(np.clip(raw_reward / config.REWARD_SCALE, config.REWARD_MIN, config.REWARD_MAX))

        # 存储经验（使用扁平化actions）
        joint_next = joint_states + np.random.normal(0, 0.01, joint_states.shape).astype(np.float32)
        self.masac.store(joint_states, np.array(actions), clipped_reward, joint_next, False)
        self.masac.decay_epsilon()

        # 网络更新
        losses = {}
        if (len(self.masac.buffer) > config.WARMUP_STEPS and
                len(self.masac.buffer) % config.UPDATE_INTERVAL == 0):
            losses = self.masac.update()

        # 关键修改：把分配结果和原始奖励返回给主循环复用，避免外面再调一次 assign()
        return losses, assignment, raw_reward

    def format_assignment(self, assignment: Dict[int, List[int]]) -> str:
        """格式化输出分配结果"""
        lines = []
        lines.append("=" * 60)
        lines.append("任务分配结果")
        lines.append("=" * 60)

        total_cost = 0.0
        assigned_targets = set()

        for pid, tids in sorted(assignment.items()):
            p = self.env.platforms[pid]
            lines.append(f"\n平台{pid}({p.type}):")

            prev_pos = p.position
            platform_dist = 0.0
            platform_time = 0.0

            for idx, tid in enumerate(tids):
                t = self.env.targets[tid]
                dist = np.linalg.norm(prev_pos - t.position)
                time = dist / max(p.speed, 1e-6)
                platform_dist += dist
                platform_time += time
                prev_pos = t.position

                lines.append(f"  → 目标{tid}({t.task_type}) | 距离:{dist / 1000:.1f}km | 用时:{time:.0f}s")
                assigned_targets.add(tid)

            lines.append(f"  小计: 航程{platform_dist / 1000:.1f}km | 总时间{platform_time:.0f}s")
            total_cost += platform_dist

        # 未分配目标
        all_targets = set(range(len(self.env.targets)))
        unassigned = sorted(all_targets - assigned_targets)

        lines.append("\n" + "=" * 60)
        lines.append(f"总航程代价: {total_cost / 1000:.1f} km")

        if unassigned:
            lines.append(f"⚠️ 未完全覆盖！未分配目标: {unassigned}")
        else:
            lines.append(f"✅ 完全覆盖！所有{len(self.env.targets)}个目标已分配")

        lines.append("=" * 60)
        return "\n".join(lines)
# ========== 3. 在线微调管理器（新增）==========
class OnlineFineTuner:
    """支持动态环境的在线参数微调 - 对应申请书第20页"""

    def __init__(self, system, reallocator):  # 添加 reallocator 参数
        self.system = system
        self.reallocator = reallocator  # 保存引用
        self.base_lr = config.LR_ACTOR
        self.adaptation_steps = 20
        self.dimension_changed = False  # ✅ 新增：标记维度是否刚改变

    def _fast_policy_update(self, assignment: Dict[int, int], reward: float):
        """
        快速策略更新（仅更新最后一层，冻结前层）
        确保总时间<0.5秒（申请书技术指标）
        """
        import time
        start_t = time.time()

        if len(assignment) == 0:
            return

        # 1. 构造单步训练数据
        platform_states = []
        target_states = []
        actions = []

        for pid, tid in assignment.items():
            p_state = self.system.env.platforms[pid].get_state()
            t_state = self.system.env.targets[tid].get_state()

            platform_states.append(p_state)
            target_states.append(t_state)
            actions.append(tid)

        # 转换为tensor
        p_tensor = torch.FloatTensor(np.array(platform_states)).to(self.system.device)
        t_tensor = torch.FloatTensor(np.array(target_states)).to(self.system.device)
        a_tensor = torch.LongTensor(actions).to(self.system.device)

        # 2. 仅优化Actor最后一层（冻结网络前层已实现于freeze_base_layers）
        optimizer = torch.optim.Adam(
            self.system.masac.actor.net[-1].parameters(),  # 只优化最后一层
            lr=self.base_lr * 5  # 更高学习率加速收敛
        )

        # 3. 执行20步快速梯度上升（最大化奖励）
        for step in range(self.adaptation_steps):
            optimizer.zero_grad()

            # 前向传播获取动作概率
            with torch.no_grad():
                h_u, h_t = self.system.encoder(p_tensor, t_tensor, None)

            # 计算当前策略的log概率
            n_targets = len(self.system.env.targets)
            action_dim = n_targets + 1
            masks = torch.ones(len(actions), action_dim).to(self.system.device)
            logits = self.system.masac.actor.net(h_u)  # 仅最后一层前向
            logits = logits.masked_fill(masks == 0, float('-inf'))
            log_probs = F.log_softmax(logits, dim=-1)

            # 选择对应动作的log概率
            selected_log_probs = log_probs.gather(1, a_tensor.unsqueeze(1))

            # 策略梯度：最大化奖励（梯度上升）
            loss = -(selected_log_probs * reward).mean()
            loss.backward()

            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(
                self.system.masac.actor.net[-1].parameters(), 1.0
            )
            optimizer.step()

            # 检查时间限制（<0.5秒）
            if time.time() - start_t > 0.4:  # 留0.1s余量
                print(f"警告: 在线微调接近时间限制，已执行{step + 1}步")
                break

        elapsed = time.time() - start_t
        print(f"在线微调完成: {min(self.adaptation_steps, step + 1)}步, 耗时{elapsed * 1000:.1f}ms")

    def freeze_base_layers(self):
        """冻结前层网络，只保留最后一层可训练"""
        # 冻结GNN编码器前层（适配GATEncoder结构）
        for param in [self.system.encoder.W_u.weight, self.system.encoder.W_u.bias,
                      self.system.encoder.W_t.weight, self.system.encoder.W_t.bias]:
            param.requires_grad = False

        # 冻结注意力层和GRU
        for param in self.system.encoder.attention_proj.parameters():
            param.requires_grad = False
        for param in self.system.encoder.gru.parameters():
            param.requires_grad = False

        # 冻结Actor前层，只保留最后一层
        for param in self.system.masac.actor.net[:-1].parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """恢复所有参数可训练（正常训练模式）"""
        for param in self.system.encoder.parameters():
            param.requires_grad = True
        for param in self.system.masac.actor.parameters():
            param.requires_grad = True

    def adapt_to_new_size(self, n_platforms: int, n_targets: int):
        # ===== 新增：安全检查 =====
        # 检查变化范围（您的场景：平台8-12，目标14-18，变化±2）
        current_n_platforms = len(self.system.env.platforms)
        current_n_targets = len(self.system.env.targets)

        platform_change = abs(n_platforms - current_n_platforms)
        target_change = abs(n_targets - current_n_targets)

        if platform_change > 2 or target_change > 4:  # 放宽到4，允许一次+2后再次+2
            print(f"❌ 拒绝调整：变化过大（平台±{platform_change}, 目标±{target_change}），超过安全阈值(2,4)")
            print(f"   建议：重启训练或使用更保守的应急策略")
            return

        if target_change > 2:
            print(f"⚠️  警告：目标变化{target_change}较大，可能影响模型稳定性")

        # 清空所有缓存（防止旧维度数据污染新网络）
        self.system._distance_cache.clear()
        if hasattr(self.system, 'route_planner') and hasattr(self.system.route_planner, 'waypoint_cache'):
            self.system.route_planner.waypoint_cache.clear()
        new_action_dim = n_targets + 1  # 包含idle动作

        # ===== 1. 更新Actor网络（已有逻辑优化） =====
        new_actor = Actor(new_action_dim).to(self.system.device)

        with torch.no_grad():
            # 复制共享层权重（输入层和隐藏层）
            old_actor = self.system.masac.actor
            for i, (old_layer, new_layer) in enumerate(zip(old_actor.net, new_actor.net)):
                if isinstance(old_layer, nn.Linear) and isinstance(new_layer, nn.Linear):
                    # 如果是最后一层（输出层），维度可能不同，需要特殊处理
                    if i == len(old_actor.net) - 1:  # 最后一层
                        # 只复制输入维度部分的权重
                        min_out = min(old_layer.weight.shape[0], new_layer.weight.shape[0])
                        min_in = min(old_layer.weight.shape[1], new_layer.weight.shape[1])
                        new_layer.weight[:min_out, :min_in].copy_(old_layer.weight[:min_out, :min_in])
                        if old_layer.bias is not None:
                            new_layer.bias[:min_out].copy_(old_layer.bias[:min_out])
                    else:
                        # 中间层直接复制
                        if old_layer.weight.shape == new_layer.weight.shape:
                            new_layer.weight.copy_(old_layer.weight)
                            if old_layer.bias is not None:
                                new_layer.bias.copy_(old_layer.bias)

        self.system.masac.actor = new_actor
        self.system.masac.actor_target = Actor(new_action_dim).to(self.system.device)
        self.system.masac.actor_target.load_state_dict(new_actor.state_dict())

        # 关键：重新初始化Actor优化器（确保优化的是新参数）
        self.system.masac.actor_opt = torch.optim.Adam(
            new_actor.parameters(),
            lr=config.LR_ACTOR
        )

        # ===== 2. 关键修复：同步更新Critic网络（新增） =====
        # Critic输入维度 = N_PLATFORMS * HIDDEN_DIM + N_PLATFORMS * action_dim
        if n_platforms != current_n_platforms:
            # 平台数变化：必须重建Critic以匹配新的输入维度
            new_critic = Critic(new_action_dim, n_platforms=n_platforms).to(self.system.device)

            with torch.no_grad():
                old_critic = self.system.masac.critic
                # 平台数变化时，第一层输入维度改变，无法直接复制权重，重新初始化
                # 但保留其他层的训练经验（如果有隐藏层）
                for i, (old_layer, new_layer) in enumerate(zip(old_critic.net, new_critic.net)):
                    if isinstance(old_layer, nn.Linear) and isinstance(new_layer, nn.Linear):
                        # 只复制维度匹配层的权重（通常是输出层）
                        if old_layer.weight.shape == new_layer.weight.shape:
                            new_layer.weight.copy_(old_layer.weight)
                            if old_layer.bias is not None:
                                new_layer.bias.copy_(old_layer.bias)
                        # 第一层维度不匹配时，使用Xavier初始化（默认）

            print(f"平台数变化：{current_n_platforms} → {n_platforms}，Critic已重建")
        else:
            # 仅目标数变化：复用原有权重迁移逻辑
            new_critic = Critic(new_action_dim, n_platforms=n_platforms).to(self.system.device)

            with torch.no_grad():
                old_critic = self.system.masac.critic
                for old_layer, new_layer in zip(old_critic.net, new_critic.net):
                    if isinstance(old_layer, nn.Linear) and isinstance(new_layer, nn.Linear):
                        if old_layer.weight.shape == new_layer.weight.shape:
                            new_layer.weight.copy_(old_layer.weight)
                            if old_layer.bias is not None:
                                new_layer.bias.copy_(old_layer.bias)
                        else:
                            # 维度不匹配（通常是第一层），部分复制状态权重
                            state_dim = n_platforms * config.HIDDEN_DIM
                            if old_layer.weight.shape[1] > state_dim and new_layer.weight.shape[1] > state_dim:
                                new_layer.weight[:, :state_dim].copy_(old_layer.weight[:, :state_dim])
                                nn.init.xavier_uniform_(new_layer.weight[:, state_dim:])
                                if old_layer.bias is not None:
                                    new_layer.bias.copy_(old_layer.bias)

            # 同步更新Critic目标网络（关键！）
        self.system.masac.critic = new_critic
        self.system.masac.critic_target = Critic(new_action_dim, n_platforms=n_platforms).to(self.system.device)
        # 平台数变化时，目标网络也重新初始化；仅动作维变化时，复制权重
        if n_platforms == current_n_platforms:
            self.system.masac.critic_target.load_state_dict(new_critic.state_dict())

        # 重新初始化优化器（确保优化新参数）
        self.system.masac.critic_opt = torch.optim.Adam(
            new_critic.parameters(),
            lr=config.LR_CRITIC
        )

        # ===== 3. 同步更新配置（关键！） =====
        # 更新全局配置，确保后续创建的mask等使用新维度
        config.ACTION_DIM = new_action_dim

        # ===== 4. 清空经验回放缓冲区（可选但建议） =====
        # 因为旧的经验数据动作维度与新维度不匹配，建议清空或标记
        if len(self.system.masac.buffer) > 0:
            print(f"警告: 清空经验缓冲区（{len(self.system.masac.buffer)}条），因动作维度已改变")
            self.system.masac.buffer.clear()
        # ===== 5. 同步更新IRL奖励学习器（关键修复） =====
        # 重新初始化IRL网络以匹配新的动作维度
        new_irl = IRLRewardLearner(config.HIDDEN_DIM, new_action_dim).to(self.system.device)
        with torch.no_grad():
            old_irl = self.reallocator.irl_learner  # 现在可以直接访问
            # 复制隐藏层权重（输入层维度变化无法复制）
            for i, (old_layer, new_layer) in enumerate(zip(old_irl.reward_net, new_irl.reward_net)):
                if isinstance(old_layer, nn.Linear) and isinstance(new_layer, nn.Linear):
                    # 跳过输入层（维度已改变：platform_dim + target_dim + action_dim）
                    if i == 0:
                        continue
                    # 复制维度匹配的层（通常是 hidden 和 output 层）
                    if old_layer.weight.shape == new_layer.weight.shape:
                        new_layer.weight.copy_(old_layer.weight)
                        if old_layer.bias is not None:
                            new_layer.bias.copy_(old_layer.bias)
        self.reallocator.irl_learner = new_irl

        old_epsilon = self.system.masac.epsilon
        self.system.masac.epsilon = 0.4  # 重置到中间探索水平
        print(f"🔄 探索率重置: {old_epsilon:.3f} → 0.300 (环境变化后重新探索)")

        # ✅ 修改4：清空失效的历史最佳模型（防止维度不匹配时无法加载）
        # 通知训练器需要清空best_model_state
        self.dimension_changed = True  # 添加标志位，在类__init__中初始化

    def online_finetune(self, emergency_buffer, n_steps=None):
        """在线微调（仅更新最后一层）"""
        if n_steps is None:
            n_steps = self.adaptation_steps

        if len(emergency_buffer) < 5:
            return

        # 只优化未冻结的参数（最后一层）
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad,
                   list(self.system.masac.actor.parameters()) +
                   list(self.system.encoder.parameters())),
            lr=self.base_lr * 5  # 更高学习率加速适应
        )

        for _ in range(n_steps):
            batch = random.sample(emergency_buffer, min(5, len(emergency_buffer)))
            # 快速更新...
            # 这里简化处理，实际使用IRL学习的奖励

# ================== 新增：自适应探索管理器（位置：GuaranteedAssignment 之后） ==================
class AdaptiveExplorer:
    """应急探索管理器 - 防止应急后探索率过低导致策略崩溃"""

    def __init__(self, masac):
        self.masac = masac
        self.normal_epsilon = config.EPSILON_END  # 通常是0.15
        self.emergency_epsilon = 0.5  # 应急后初始探索率（提高）
        self.recovery_steps = 0
        self.max_recovery = 50  # 50回合恢复期

    def on_emergency(self):
        """应急事件触发时调用"""
        old_eps = self.masac.epsilon
        self.masac.epsilon = self.emergency_epsilon
        self.recovery_steps = self.max_recovery
        print(f"🔄 应急探索重置: {old_eps:.3f} → {self.emergency_epsilon} (恢复周期:{self.max_recovery}回合)")

    def step(self):
        """每回合调用 - 在train循环中"""
        if self.recovery_steps > 0:
            self.recovery_steps -= 1
            # 线性衰减回正常水平
            decay = (self.emergency_epsilon - self.normal_epsilon) / self.max_recovery
            self.masac.epsilon = max(self.normal_epsilon,
                                     self.masac.epsilon - decay)

            if self.recovery_steps == 0:
                print(f"✅ 探索率恢复正常水平: {self.masac.epsilon:.3f}")


# ================== 结束插入 ==================

# ========== 4. 动态应急重分配器（新增）==========
class DynamicReallocator:
    """修复版：约束满足的应急重分配器"""

    def __init__(self, env, system):
        self.env = env
        self.system = system
        self.fine_tuner = OnlineFineTuner(system, self)
        self.emergency_buffer = deque(maxlen=1000)
        self.irl_learner = IRLRewardLearner(config.HIDDEN_DIM, config.ACTION_DIM).to(system.device)
        self.emergency_count = 0
        self.explorer = AdaptiveExplorer(system.masac)  # 添加探索管理

    def handle_emergency(self, event_type: str, event_data: dict):
        """统一入口"""
        self.emergency_count += 1
        self.explorer.on_emergency()  # 重置探索率
        print(f"\n[紧急事件 #{self.emergency_count}] 类型: {event_type}")

        if event_type == 'crash':
            result = self._reallocate_after_crash(event_data['platform_id'])
        elif event_type == 'new_target':
            result = self._reallocate_with_new_targets(event_data['targets'])
        elif event_type == 'target_lost':
            result = self._reallocate_after_target_loss(event_data.get('targets', []))
        else:
            result = {}

        # 验证结果
        if result:
            self._validate_emergency_result(result)
        return result

    def _reallocate_after_crash(self, crashed_id: int):
        """带约束保障的应急重构"""
        start_time = time.time()
        alive_platforms = [i for i in range(len(self.env.platforms)) if i != crashed_id]

        # 阶段1：尝试快速微调（原有逻辑）
        self.fine_tuner.freeze_base_layers()
        best_assignment = {}
        best_reward = -1e10

        for step in range(self.fine_tuner.adaptation_steps):
            assignment = self._partial_assign(alive_platforms)
            reward = self._evaluate_assignment(assignment)

            if len(assignment) > len(best_assignment) or \
                    (len(assignment) == len(best_assignment) and reward > best_reward):
                best_assignment = assignment.copy()
                best_reward = reward

            # 如果已达成全分配，提前停止
            if len(assignment) == len(alive_platforms):
                print(f"✅ 微调成功：回合{step}达成全分配")
                break

        # 阶段2：约束保障回退（关键补丁）
        if len(best_assignment) < len(alive_platforms):
            print(f"⚠️ 神经网络仅分配{len(best_assignment)}/{len(alive_platforms)}，启动强制回退...")
            # 将优质强制分配加入经验池
            self._store_demonstration(best_assignment, reward=10.0)

        self.fine_tuner.unfreeze_all()

        elapsed = time.time() - start_time
        print(f"重构完成: 耗时{elapsed * 1000:.1f}ms，分配{len(best_assignment)}个任务")
        return best_assignment

    def _reallocate_with_new_targets(self, new_targets: List[np.ndarray]) -> Dict[int, int]:
        """
        处理新增目标的应急重分配 - 与坠毁场景类似，但所有平台可用

        Args:
            new_targets: 新增目标的位置列表（已由handle_emergency传入，环境已更新）

        Returns:
            assignment: 平台到目标的分配字典
        """
        start_time = time.time()

        # 新增目标场景：所有平台都可用（无需排除坠毁平台）
        alive_platforms = list(range(len(self.env.platforms)))

        print(f"   新增目标数: {len(new_targets)}，当前总目标数: {len(self.env.targets)}")
        print(f"   存活平台: {len(alive_platforms)}个")

        # 阶段1：尝试快速微调（复用 crash 场景的逻辑）
        self.fine_tuner.freeze_base_layers()
        best_assignment = {}
        best_reward = -1e10

        for step in range(self.fine_tuner.adaptation_steps):
            assignment = self._partial_assign(alive_platforms)
            reward = self._evaluate_assignment(assignment)

            # 优先选择分配数量多的，其次选择奖励高的
            if len(assignment) > len(best_assignment) or \
                    (len(assignment) == len(best_assignment) and reward > best_reward):
                best_assignment = assignment.copy()
                best_reward = reward

            # 如果已达成全分配，提前停止
            if len(assignment) == len(alive_platforms):
                print(f"✅ 微调成功：回合{step}达成全分配")
                break

        # 阶段2：约束保障回退（关键补丁：确保100%分配）
        if len(best_assignment) < len(alive_platforms):
            print(f"⚠️ 神经网络仅分配{len(best_assignment)}/{len(alive_platforms)}，启动强制回退...")
            # 去除强制保底：神经网络分配多少就是多少，不再强制补全
            print(f"   神经网络分配结果: {len(best_assignment)}个平台有任务")
            # 将优质强制分配加入经验池用于后续学习
            self._store_demonstration(best_assignment, reward=10.0)

        self.fine_tuner.unfreeze_all()

        elapsed = time.time() - start_time
        print(f"新增目标重构完成: 耗时{elapsed * 1000:.1f}ms，分配{len(best_assignment)}个任务")

        # 验证结果（可选，调试用）
        if best_assignment:
            self._validate_emergency_result(best_assignment)

        return best_assignment

    def _reallocate_after_target_loss(self, lost_target_ids: List[int] = None) -> Dict[int, int]:
        """
        处理目标丢失/减少后的应急重分配

        Args:
            lost_target_ids: 丢失的目标ID列表（仅用于日志记录）

        Returns:
            assignment: 平台到目标的分配字典
        """
        start_time = time.time()

        # 目标丢失场景：所有平台都可用，但可用目标池减少了
        alive_platforms = list(range(len(self.env.platforms)))

        if lost_target_ids:
            print(f"   标记为完成的目标: {lost_target_ids}")
        print(f"   存活平台: {len(alive_platforms)}个")
        remaining = sum(1 for t in self.env.targets if not t.completed)
        print(f"   剩余可用目标: {remaining}/{len(self.env.targets)}")

        # 阶段1：尝试快速微调（复用既有逻辑）
        self.fine_tuner.freeze_base_layers()
        best_assignment = {}
        best_reward = -1e10

        for step in range(self.fine_tuner.adaptation_steps):
            assignment = self._partial_assign(alive_platforms)
            reward = self._evaluate_assignment(assignment)

            # 优先选择分配数量多的，其次选择奖励高的
            if len(assignment) > len(best_assignment) or \
                    (len(assignment) == len(best_assignment) and reward > best_reward):
                best_assignment = assignment.copy()
                best_reward = reward

            # 如果已达成全分配，提前停止
            if len(assignment) == len(alive_platforms):
                print(f"✅ 微调成功：回合{step}达成全分配")
                break

        # 阶段2：约束保障回退（确保100%平台都有任务）
        if len(best_assignment) < len(alive_platforms):
            print(f"⚠️ 神经网络仅分配{len(best_assignment)}/{len(alive_platforms)}，启动强制回退...")
            # 去除强制保底：神经网络分配多少就是多少，不再强制补全
            print(f"   神经网络分配结果: {len(best_assignment)}个平台有任务")
            self._store_demonstration(best_assignment, reward=10.0)

        self.fine_tuner.unfreeze_all()

        elapsed = time.time() - start_time
        print(f"目标丢失重构完成: 耗时{elapsed * 1000:.1f}ms，分配{len(best_assignment)}个任务")

        if best_assignment:
            self._validate_emergency_result(best_assignment)

        return best_assignment

    def _partial_assign(self, alive_platforms: List[int]):
        """部分分配 - 使用应急mask"""
        assignment = {}
        assigned_targets = set()
        completed_targets = set(self.env.completed_tasks)

        # 获取编码器输出
        platform_states, target_states, base_mask, _ = self.env.get_graph_data()
        platform_states = platform_states.to(self.system.device)
        target_states = target_states.to(self.system.device)
        base_mask = base_mask.to(self.system.device)

        with torch.no_grad():
            hu, ht = self.system.encoder(platform_states, target_states, base_mask)

        for i in alive_platforms:
            if i >= len(self.env.platforms):
                continue

            state = hu[i].cpu().numpy()
            # 使用应急mask（更严格）
            mask = self.system.get_action_mask_emergency(
                assigned_targets, completed_targets, i
            ).cpu().numpy()

            # 安全选择（补丁2）
            action = self.system.masac.safe_select_action(state, mask, temperature=0.8)

            if action < len(self.env.targets):
                target = self.env.targets[action]
                # 双重检查时序（补丁3）
                if target.predecessors and not all(pid in completed_targets for pid in target.predecessors):
                    continue  # 跳过违反时序的

                if action not in assigned_targets:
                    assignment[i] = action
                    assigned_targets.add(action)
                    completed_targets.add(action)  # 立即更新时序状态

        return assignment

    def _evaluate_assignment(self, assignment: Dict[int, int]) -> float:
        """综合评估分配质量"""
        if not assignment:
            return -100.0

        reward = 0.0
        # 基础：分配数量（鼓励多分配）
        reward += len(assignment) * 5.0

        # 效率：航程估计
        for pid, tid in assignment.items():
            p, t = self.env.platforms[pid], self.env.targets[tid]
            dist = np.linalg.norm(p.position - t.position)
            reward -= dist / 1e5  # 归一化惩罚

        # 约束满足奖励
        targets = list(assignment.values())
        if len(targets) == len(set(targets)):
            reward += 10.0  # 无重复

        return reward

    def _validate_emergency_result(self, assignment: Dict[int, int]):
        """最终验证"""
        print(f"\n🔍 应急结果验证:")
        print(f"   ├─ 分配平台数: {len(assignment)}")
        print(f"   ├─ 唯一目标数: {len(set(assignment.values()))}")

        # 检查时序
        violations = 0
        for pid, tid in assignment.items():
            t = self.env.targets[tid]
            if t.predecessors:
                if not all(self.env.targets[p].completed for p in t.predecessors):
                    violations += 1
                    print(f"   ├─ ❌ 时序违反: T{tid}")

        if violations == 0:
            print(f"   └─ ✅ 所有约束满足")
        else:
            print(f"   └─ ⚠️ 存在{violations}个时序违反（不应出现）")

    def _store_demonstration(self, assignment: Dict[int, int], reward: float):
        """存储强制分配作为专家示范"""
        # 构造经验数据
        platform_states, target_states, _, _ = self.env.get_graph_data()

        joint_states = []
        actions = []
        for pid, tid in assignment.items():
            # 这里简化处理，实际应存储编码后的状态
            joint_states.append(platform_states[pid].numpy())
            actions.append(tid)

        # 存储到特殊缓冲区
        if len(actions) == len(self.env.platforms) - (1 if self.emergency_count > 0 else 0):
            # 只有全分配才存储
            self.emergency_buffer.append({
                'states': np.array(joint_states),
                'actions': np.array(actions),
                'reward': reward
            })
            print(f"   💾 存储专家示范({len(actions)}个分配)，奖励{reward}")

class Trainer:
    """训练器基类 - 修复：正确定义基类"""

    def __init__(self):
        self.history = {
            'rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'duplicate_rate': [],
            'unique_assignments': [],
            'critic_losses': [],
            'actor_losses': [],
            'platform_coverage': [],  # 记录每回合的平台分配数
            'target_coverage': [],  # 记录每回合的目标分配数
            'epsilons': [],  # 记录探索率变化
            'assignments_detail': [],  # 每回合的完整assignment字典
            'temporal_violations': [],  # 每回合的时序违反次数
            'task_completion_order': []  # 任务完成顺序（用于时序验证）
        }

    def train(self, episodes=1000):
        """训练循环 - 需要在子类中实现"""
        raise NotImplementedError("Subclasses must implement train method")

    def plot_results(self):
        """绘制训练结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 奖励曲线
        if self.history['rewards']:
            ax = axes[0, 0]
            ax.plot(self.history['rewards'], alpha=0.3, label='Raw')
            if len(self.history['rewards']) >= 100:
                smoothed = np.convolve(self.history['rewards'], np.ones(100) / 100, mode='valid')
                ax.plot(smoothed, label='Smoothed')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Training Rewards')
            ax.legend()
            ax.grid(True)

        # 平均奖励
        if self.history['avg_rewards']:
            ax = axes[0, 1]
            ax.plot(self.history['avg_rewards'], color='orange')
            ax.set_xlabel('Episode (x100)')
            ax.set_ylabel('Average Reward')
            ax.set_title('Average Rewards per 100 Episodes')
            ax.grid(True)

        # 重复率
        if self.history['duplicate_rate']:
            ax = axes[1, 0]
            ax.plot(self.history['duplicate_rate'], color='red')
            ax.set_xlabel('Episode (x100)')
            ax.set_ylabel('Duplicate Rate')
            ax.set_title('Duplicate Assignment Rate')
            ax.grid(True)

        # 损失
        if self.history['critic_losses']:
            ax = axes[1, 1]
            ax.plot(self.history['critic_losses'], label='Critic', alpha=0.7)
            ax.plot(self.history['actor_losses'], label='Actor', alpha=0.7)
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.set_title('Network Losses')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig('training_results_v4_fixed.png', dpi=300, bbox_inches='tight')
        print("Saved: training_results_v4_fixed.png")
        plt.show()

class ManualTrigger:
    """手动触发管理器 - 纯标准库版本（无需pynput）"""

    def __init__(self, trigger_file='trigger_emergency.txt'):
        self.trigger_file = trigger_file
        self.manual_triggered = False
        self.trigger_count = 0
        self.last_trigger_episode = -50
        self.min_interval = 50

        # 启动后台监听线程（使用input或文件检测）
        self.listener_thread = threading.Thread(target=self._listen_keyboard, daemon=True)
        self.listener_thread.start()

        # 启动文件监控线程
        self.file_thread = threading.Thread(target=self._monitor_file, daemon=True)
        self.file_thread.start()

        print(f"手动触发系统已启动:")
        print(f"  - 键盘触发: 在控制台按 'E' + Enter 触发")
        print(f"  - 文件触发: 创建 '{trigger_file}' 文件触发")
        print(f"  - 最小间隔: {self.min_interval} 回合")

    def _listen_keyboard(self):
        """后台监听键盘输入（非阻塞）"""
        while True:
            try:
                # 使用sys.stdin读取，兼容Windows和Linux
                user_input = sys.stdin.readline().strip().lower()
                if user_input == 'e' and not self.manual_triggered:
                    self.manual_triggered = True
                    print(f"\n[手动触发] 检测到键盘输入 'E'，准备执行应急事件...")
            except Exception:
                time.sleep(0.1)

    def _monitor_file(self):
        """后台监控触发文件"""
        while True:
            try:
                if os.path.exists(self.trigger_file):
                    with open(self.trigger_file, 'r') as f:
                        content = f.read().strip()
                    try:
                        os.remove(self.trigger_file)
                        if content.upper() == 'E' or content == '':
                            self.manual_triggered = True
                            print(f"\n[手动触发] 检测到触发文件，准备执行应急事件...")
                    except Exception as e:
                        print(f"触发文件处理失败: {e}")
                time.sleep(0.5)  # 每0.5秒检查一次
            except Exception:
                time.sleep(1)

    def check_trigger(self, current_episode: int):
        """检查是否触发"""
        if current_episode - self.last_trigger_episode < self.min_interval:
            return False, None

        if self.manual_triggered:
            self.manual_triggered = False
            self.last_trigger_episode = current_episode
            self.trigger_count += 1
            return True, self._generate_event()

        return False, None

    def _generate_event(self):
        """生成随机应急事件"""
        import random
        import numpy as np

        event_type = random.choice(['crash', 'new_target', 'new_target'])

        if event_type == 'crash':
            crash_id = random.randint(0, 9)  # config.N_PLATFORMS - 1
            return event_type, {'platform_id': crash_id}
        else:
            n_new = random.randint(1, 2)
            new_positions = [
                np.array([random.uniform(20e3, 80e3),
                          random.uniform(20e3, 80e3), 0])
                for _ in range(n_new)
            ]
            return event_type, {'targets': new_positions}

    def get_status(self):
        return {
            'trigger_count': self.trigger_count,
            'last_trigger_episode': self.last_trigger_episode,
            'ready': True
        }


class TerrainTrainer(Trainer):
    def __init__(self, use_real_terrain=False, platform_configs=None, target_configs=None):
        super().__init__()
        self.env = TerrainEnv(use_real_terrain=use_real_terrain)

        # 关键修复：先加载场景配置，再创建TaskSystem，确保网络维度与实际目标数一致
        if platform_configs is not None and target_configs is not None:
            self.env.setup_scenario(platform_configs, target_configs)

        self.terrain = self.env.terrain
        self.system = TaskSystem(self.env)
        self.reallocator = DynamicReallocator(self.env, self.system)
        self.manual_trigger = ManualTrigger()
        self.emergency_cooldown = 0
        self.min_cooldown = 100
        # 添加滑动窗口最佳模型跟踪（新增）
        self.recent_best_reward = -1e10  # 最近500回合窗口内的最佳奖励
        self.recent_best_episode = 0
        self.recent_best_assignment = None
        self.window_size = 500  # 窗口大小
        self.reward_window = deque(maxlen=self.window_size)  # 记录最近500回合的奖励
        self.recent_checkpoints = []  # 记录近期保存的检查点文件，用于清理
        self.best_composite_score = -1e10  # 新增：覆盖率优先的综合评分
        print(f"当前配置: {config.N_PLATFORMS}平台, {config.N_TARGETS}目标")
        print("使用模式: 100km×100km纯模拟地形")
        print("等待手动触发应急事件...")

    def _add_parameter_noise(self, scale=0.02):
        """
        向Actor网络参数添加高斯噪声
        原理：轻微扰动网络权重，使策略产生变化，帮助跳出局部最优陷阱
        """
        with torch.no_grad():
            noise_count = 0
            for param in self.system.masac.actor.parameters():
                # 生成与参数同形状的高斯噪声
                noise = torch.randn_like(param) * scale
                param.add_(noise)  # 原地添加噪声
                noise_count += 1

        print(f"   └─ 已注入参数噪声到{noise_count}层 (scale={scale})")

    def _reset_exploration(self, episode, reason="周期性重启"):
        """
        强化的探索重启策略 - 确保每次重启都有明显波动
        """
        old_eps = self.system.masac.epsilon
        new_eps = config.RESTART_EPSILON  # 0.6

        # 1. 重置探索率到高位
        self.system.masac.epsilon = new_eps
        print(f"\n🔄 回合{episode}: {reason}")
        print(f"   探索率: {old_eps:.3f} → {new_eps:.3f} (强制探索阶段)")

        # 2. 注入高强度参数噪声（关键：破坏当前局部最优）
        with torch.no_grad():
            noise_count = 0
            for name, param in self.system.masac.actor.named_parameters():
                # 对最后一层（输出层）施加更强噪声
                if 'net.2' in name or 'net.3' in name:  # 输出层权重
                    noise_scale = config.PARAM_NOISE_SCALE * 2  # 0.1
                else:
                    noise_scale = config.PARAM_NOISE_SCALE  # 0.05

                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)
                noise_count += 1

        print(f"   已注入参数噪声到{noise_count}层 (scale={config.PARAM_NOISE_SCALE})")

        # 3. 临时提升学习率（激进探索阶段）
        for param_group in self.system.masac.actor_opt.param_groups:
            param_group['lr'] = config.LR_ACTOR * 5.0  # 原为2.0，提升到5倍
        for param_group in self.system.masac.critic_opt.param_groups:
            param_group['lr'] = config.LR_CRITIC * 3.0  # Critic也提升

        print(f"   Actor学习率临时提升至: {config.LR_ACTOR * 5:.2e} (5倍)")
        print(f"   Critic学习率临时提升至: {config.LR_CRITIC * 3:.2e} (3倍)")

        # 4. 清空近期经验缓冲区（消除导致平台期的旧数据污染）
        if len(self.system.masac.buffer) > 2000:
            print(f"   清空经验缓冲区（保留最新2000条）...")
            buffer_list = list(self.system.masac.buffer)
            self.system.masac.buffer.clear()
            self.system.masac.buffer.extend(buffer_list[-2000:])  # 只保留最新2000条

        # 5. 重置优化器状态（清除动量积累）
        self.system.masac.actor_opt = torch.optim.Adam(
            self.system.masac.actor.parameters(),
            lr=config.LR_ACTOR * 5.0
        )
        self.system.masac.critic_opt = torch.optim.Adam(
            self.system.masac.critic.parameters(),
            lr=config.LR_CRITIC * 3.0
        )
        print(f"   优化器状态已重置（清除历史动量）")

    def train_with_emergency_simulation(self, episodes=1000):
        """训练过程中模拟突发事件，增强鲁棒性"""
        for episode in range(episodes):
            self.env.reset()

            # 每50轮模拟一次突发事件（训练应急能力）
            if episode > 200 and episode % 50 == 0:
                # 随机坠毁一架无人机
                crash_id = random.randint(0, config.N_PLATFORMS - 1)
                emergency_assignment = self.reallocator.handle_emergency(
                    'crash', {'platform_id': crash_id}
                )
                print(f"Episode {episode}: 模拟坠毁，重构分配完成")

    def train(self, episodes=2000):
        print(f"\n开始训练 {episodes} 个回合...")
        print(f"设备: {self.system.masac.device}")
        print(f"地图尺寸: {config.MAP_SIZE_X / 1000:.1f}km x {config.MAP_SIZE_Y / 1000:.1f}km")
        print(f"模式: 手动触发应急事件（自动触发已禁用）")
        print(f"当前固定配置: {config.N_PLATFORMS}平台, {config.N_TARGETS}目标")
        print("等待手动触发应急事件...")

        self.explorer = AdaptiveExplorer(self.system.masac)
        collapse_recovery = 0
        consecutive_collapse = 0
        monitor = ConvergenceMonitor()
        best_model_state = None
        last_restart_episode = -100
        last_convergence_status = False

        # 手动触发冷却计数器
        emergency_cooldown = 0
        min_cooldown = 50  # 触发后至少等待50回合才能再次触发

        for episode in range(episodes):
            self.env.reset()
            episode_reward = 0
            episode_losses = {'critic_loss': 0, 'actor_loss': 0}
            n_updates = 0
            self.explorer.step()

            if episode > 0 and episode % config.EXPLORATION_RESTART_EVERY == 0:
                if episode - last_restart_episode >= 1000:  # 至少间隔1000回合才重启
                    self._reset_exploration(episode, "周期性探索重启")
                    last_restart_episode = episode
            # ========== 手动触发检查（替代原有的自动触发）==========
            should_trigger_emergency = False
            trigger_reason = ""

            # 检查手动触发（优先级最高）
            is_triggered, event_info = self.manual_trigger.check_trigger(episode)

            if is_triggered:
                should_trigger_emergency = True
                event_type, event_data = event_info
                trigger_reason = f"手动触发 #{self.manual_trigger.trigger_count}（回合{episode}）"

                print(f"\n{'!' * 60}")
                print(f"【手动应急事件】{trigger_reason}")
                print(f"{'!' * 60}")
                # 保存重规划前的分配
                old_assignment, _, _ = self.system.assign(evaluate=True)
                old_count = len(old_assignment)

                # 执行应急
                emergency_assignment = self.reallocator.handle_emergency(event_type, event_data)

                # 重规划后对比
                print("\n" + "=" * 60)
                print("【应急重规划对比】")
                print(f"重规划前: {old_count}个平台有任务")
                print(f"重规划后: {len(emergency_assignment)}个平台有任务")
                print(f"变化: {len(emergency_assignment) - old_count}")
                print("=" * 60)
                # 执行应急事件
                if event_type == 'crash':
                    crash_id = event_data['platform_id']
                    emergency_assignment = self.reallocator.handle_emergency(
                        'crash', {'platform_id': crash_id}
                    )
                    if len(emergency_assignment) < len(self.env.platforms) - 1:
                        print(f"⚠️ 应急分配不足({len(emergency_assignment)}个)，触发崩溃恢复协议...")
                        consecutive_collapse += 1
                        self.system.masac.epsilon = 0.5
                        if len(self.system.masac.buffer) > 5000:
                            print("   └─ 清空经验缓冲区...")
                            self.system.masac.buffer.clear()
                    else:
                        consecutive_collapse = 0
                    print(f"事件: 平台{crash_id}坠毁，重构完成，"
                          f"新分配{len(emergency_assignment)}个任务")
                else:  # new_target
                    new_positions = event_data['targets']
                    n_new = len(new_positions)
                    print(
                        f"事件: 新增{n_new}个临时目标（当前目标数: {len(self.env.targets)} → {len(self.env.targets) + n_new}）")

                    # 添加新目标到环境
                    for i, pos in enumerate(new_positions):
                        new_id = len(self.env.targets)
                        # 生成简化的前序依赖（30%概率需要电子战支援）
                        predecessors = []
                        if random.random() < 0.3 and new_id > 0:
                            potential = [j for j in range(new_id)
                                         if self.env.targets[j].task_type == 'electronic']
                            if potential:
                                predecessors = random.sample(potential, min(1, len(potential)))

                        new_target = Target(new_id, pos, predecessors)
                        # 确保新目标的任务类型随机但合理
                        new_target.task_type = random.choice(['attack', 'reconnaissance', 'electronic'])
                        self.env.targets.append(new_target)

                    # 调整网络维度适应新目标数
                    self.reallocator.fine_tuner.adapt_to_new_size(
                        len(self.env.platforms),
                        len(self.env.targets)
                    )

                    emergency_assignment = self.reallocator.handle_emergency(
                        'new_target', {'targets': new_positions}
                    )

                    if len(emergency_assignment) < len(self.env.platforms):
                        print(f"⚠️ 应急分配不足({len(emergency_assignment)}/{len(self.env.platforms)})，触发崩溃恢复...")
                        consecutive_collapse += 1
                        self.system.masac.epsilon = 0.4
                    else:
                        consecutive_collapse = 0

                # 设置冷却期
                emergency_cooldown = min_cooldown

            else:
                # 正常冷却递减（非手动触发时）
                if emergency_cooldown > 0:
                    emergency_cooldown -= 1

            # 正常训练步骤
            temp_assignment = {}  # 用于保存本回合最终分配结果
            for step in range(config.MAX_STEPS):
                # 关键修改：接收 train_step() 返回的三元组
                losses, assignment, raw_reward = self.system.train_step()
                if losses:
                    episode_losses['critic_loss'] += losses.get('critic_loss', 0)
                    episode_losses['actor_loss'] += losses.get('actor_loss', 0)
                    n_updates += 1

                # 直接复用 train_step() 里已经算好的分配结果和奖励
                # 不再重复调用 self.system.assign() 和 self.system.calc_reward()
                reward = float(np.clip(raw_reward / config.REWARD_SCALE, config.REWARD_MIN, config.REWARD_MAX))
                episode_reward += reward
                temp_assignment = assignment  # 保存最后一次分配，供后续统计复用

            avg_episode_reward = episode_reward / config.MAX_STEPS

            # === 更新收敛监控 ===
            # 关键修改：直接复用 temp_assignment，不再额外调用 assign(evaluate=True)
            n_assigned = len(temp_assignment) if temp_assignment else 0
            status = monitor.update(episode, avg_episode_reward, n_assigned, temp_assignment)
            if not status.get('new_best') and (episode - monitor.best_episode) >= 200:
                if episode - last_restart_episode >= 50:  # 至少间隔50回合，避免连续震荡
                    print(f"\n🚨 回合{episode}: 已连续{episode - monitor.best_episode}回合未突破最佳，强制重启探索...")
                    self._reset_exploration(episode, "强制重启：找回历史最佳")
                    last_restart_episode = episode
                    monitor.patience_counter = 0  # 重置耐心计数
            # ========== 新增：检测到平台期时强制重启探索 ==========
            if status.get('is_plateau') and episode - last_restart_episode >= 100:
                print(f"\n🚨 回合{episode}: 检测到局部最优平台期({monitor.plateau_counter}回合无改进)")
                print(f"   当前奖励: {status['window_mean']:.3f} vs 历史最佳: {monitor.best_reward:.3f}")

                # 执行强制重启
                self._reset_exploration(episode, "强制重启：突破局部最优")

                # 可选：清空近期经验缓冲区（消除导致平台期的旧数据）
                if len(self.system.masac.buffer) > 5000:
                    print(f"   清空经验缓冲区（保留{len(self.system.masac.buffer) - 2000}条最新数据）...")
                    # 只保留最新的2000条经验
                    buffer_list = list(self.system.masac.buffer)
                    self.system.masac.buffer.clear()
                    self.system.masac.buffer.extend(buffer_list[-2000:])

                last_restart_episode = episode
                # 重置平台期计数器
                monitor.plateau_counter = 0

            # 崩溃检测与恢复
            if status.get('is_collapse'):
                consecutive_collapse += 1
                print(f"🚨 检测到策略崩溃（第{consecutive_collapse}次连续）...")

                if consecutive_collapse >= 3:
                    print("🚨🚨 连续3次崩溃，执行强制恢复协议：")
                    print("   1. 探索率强制提升至0.6")
                    print("   2. 清空经验缓冲区（消除污染数据）")
                    print("   3. 重置网络权重（重新初始化Actor最后层）")

                    self.system.masac.epsilon = 0.6
                    if len(self.system.masac.buffer) > 1000:
                        self.system.masac.buffer.clear()

                    with torch.no_grad():
                        for param in self.system.masac.actor.net[-1].parameters():
                            param.data.normal_(0, 0.1)

                    consecutive_collapse = 0
                    collapse_recovery += 1
                else:
                    self.system.masac.epsilon = 0.5
            else:
                consecutive_collapse = 0

            if not status.get('converged'):
                last_convergence_status = False

            # ========== 实时保存当前最优模型（使用原始奖励评估质量）==========
            if config.SAVE_BEST and episode >= 100:
                current_unique_targets = len(
                    set(tid for tids in temp_assignment.values() for tid in tids)) if temp_assignment else 0
                n_platforms_assigned = status.get('n_platforms_assigned', 0)

                # 【关键修改】使用未裁剪的原始奖励来评估模型真实质量
                raw_reward = self.system.calc_reward(temp_assignment)

                # 综合评分：原始奖励占90%（包含航程、时间、收益），覆盖率占10%
                coverage_rate = current_unique_targets / max(config.N_TARGETS, 1)
                platform_rate = n_platforms_assigned / max(config.N_PLATFORMS, 1)

                # 覆盖率部分最多贡献10分，原始奖励贡献主体
                coverage_part = (coverage_rate * 10.0 + platform_rate * 5.0)
                composite_score = raw_reward * 0.9 + coverage_part

                if composite_score > self.best_composite_score:
                    self.best_composite_score = composite_score
                    best_model_state = {
                        'actor': self.system.masac.actor.state_dict(),
                        'critic': self.system.masac.critic.state_dict(),
                        'encoder': self.system.encoder.state_dict(),
                        'episode': episode,
                        'reward': float(raw_reward),  # 保存原始奖励
                        'window_mean': float(status['window_mean']),
                        'unique_targets': current_unique_targets,
                        'n_platforms': n_platforms_assigned,
                        'assignment': temp_assignment,
                    }
                    torch.save(best_model_state, 'best_model_checkpoint.pth')
                    print(f"\n💾 新最佳模型保存: 回合{episode}, 原始奖励{raw_reward:.1f}, "
                          f"覆盖{current_unique_targets}/{config.N_TARGETS}, "
                          f"平台{n_platforms_assigned}/{config.N_PLATFORMS} "
                          f"[综合评分:{composite_score:.1f}]")
                    for param_group in self.system.masac.actor_opt.param_groups:
                        param_group['lr'] = config.LR_ACTOR

            # 记录历史数据
            self.history['rewards'].append(avg_episode_reward)
            self.history['platform_coverage'].append(len(temp_assignment))
            unique_targets_now = len(set(tid for tids in temp_assignment.values()
                                         for tid in
                                         (tids if isinstance(tids, list) else [tids]))) if temp_assignment else 0
            self.history['target_coverage'].append(unique_targets_now)
            self.history['epsilons'].append(self.system.masac.epsilon)
            self.history['assignments_detail'].append(temp_assignment.copy())

            # 时序违反统计
            temporal_viol = 0
            for pid, tids in temp_assignment.items():
                # 适配多任务：tids 可能是列表
                if not isinstance(tids, list):
                    tids = [tids]
                for tid in tids:
                    target = self.env.targets[tid]
                    if target.predecessors:
                        if not all(p in self.env.completed_tasks for p in target.predecessors):
                            temporal_viol += 1
            self.history['temporal_violations'].append(temporal_viol)

            if n_updates > 0:
                self.history['critic_losses'].append(episode_losses['critic_loss'] / n_updates)
                self.history['actor_losses'].append(episode_losses['actor_loss'] / n_updates)

            # 每100回合统计与打印
            if (episode + 1) % 20 == 0:  # 每20回合打印一次，更密集监控
                avg_reward = np.mean(self.history['rewards'][-20:])  # 最近20回合平均
                self.history['avg_rewards'].append(avg_reward)

                # 重复率计算
                recent_dups = self.history.get('duplicate_history', [])[-100:]
                dup_rate = np.mean(recent_dups) / max(config.N_PLATFORMS, 1) if recent_dups else 0

                avg_time = np.mean(self.system.stats['times'][-100:]) * 1000 if self.system.stats['times'] else 0

                # 获取触发器状态
                trigger_status = self.manual_trigger.get_status()
                episodes_since_trigger = episode - trigger_status['last_trigger_episode']

                eps_status = "🔍探索中" if self.system.masac.epsilon > 0.3 else "⚡利用中"
                plateau_info = f"平台期:{monitor.plateau_counter}" if hasattr(monitor, 'plateau_counter') else ""
                recent_assignments = self.history['assignments_detail'][-20:]
                avg_unique_targets = np.mean([
                    len(set(tid for tids in a.values() for tid in (tids if isinstance(tids, list) else [tids])))
                    for a in recent_assignments
                ])

                print(f"\n{'=' * 60}")
                print(f"【回合 {episode + 1}/{episodes}】{eps_status} ε={self.system.masac.epsilon:.3f}")
                print(f"  窗口平均奖励: {status.get('window_mean', 0):.3f} {plateau_info}")
                print(f"  当前规模: {len(self.env.platforms)}平台 × {len(self.env.targets)}目标")
                coverage_info = "✅" if status.get('full_coverage') else "❌"
                print(f"  分配覆盖: {status.get('n_platforms_assigned', 0)}/{config.N_PLATFORMS} {coverage_info}")
                current_unique = len(set(tid for tids in temp_assignment.values()
                                         for tid in
                                         (tids if isinstance(tids, list) else [tids]))) if temp_assignment else 0
                print(f"  目标分配: {current_unique}/{config.N_TARGETS} (窗口平均: {avg_unique_targets:.1f})")
                print(f"  历史最佳: {monitor.best_reward:.3f} @回合{monitor.best_episode}")

                # 手动触发状态显示
                print(f"\n  【手动触发状态】")
                print(f"  触发次数: {trigger_status['trigger_count']} | "
                      f"上次触发: {episodes_since_trigger}回合前 | "
                      f"冷却: {emergency_cooldown}/{min_cooldown}")
                print(f"  操作方式: 按键盘'E'键 或 创建文件'{self.manual_trigger.trigger_file}'")

                if status['converged']:
                    print(f"  ⚠️ 检测到收敛平台期（但需手动触发应急事件）")
                print(f"{'=' * 60}")
                if len(self.history['rewards']) >= 40:
                    recent_std = np.std(self.history['rewards'][-40:])
                    if recent_std < 0.1:  # 40回合内标准差<0.1视为平台期
                        print(f"⚠️ 警告：检测到平台期（最近40回合标准差:{recent_std:.3f}）")
                # 定期生成详细报告（每500回合）
                # 定期生成详细报告（每500回合）
                if (episode + 1) % 500 == 0:
                    current_ep = episode + 1
                    print(f"\n📊 生成回合 {current_ep} 综合训练报告...")
                    self._plot_training_report(current_ep, monitor)
                    self._plot_assignment_gantt(current_ep, recent_episodes=500)

                    # ========== 新增：滑动窗口最优模型评估与保存 ==========

                    # 1. 评估当前策略（进行10次独立评估取平均，减少随机性）
                    print(f"\n🔍 正在评估回合{current_ep}的策略质量...")
                    eval_rewards = []
                    eval_unique_targets = []
                    eval_platform_coverage = []

                    # 临时降低探索率进行公平评估
                    old_epsilon = self.system.masac.epsilon
                    self.system.masac.epsilon = 0.05  # 低探索率评估

                    for eval_i in range(10):
                        self.env.reset()
                        test_assign, _, _ = self.system.assign(evaluate=True)
                        raw_test_reward = self.system.calc_reward(test_assign)
                        # 【修改】评估也用原始奖励，与保存模型的逻辑口径统一
                        eval_rewards.append(raw_test_reward)
                        eval_unique_targets.append(len(set(tid for tids in test_assign.values() for tid in tids)) if test_assign else 0)
                        eval_platform_coverage.append(
                            len([pid for pid, tids in test_assign.items() if len(tids) > 0])
                        )

                    # 恢复探索率
                    self.system.masac.epsilon = old_epsilon

                    # 计算评估指标
                    avg_reward = np.mean(eval_rewards)
                    std_reward = np.std(eval_rewards)
                    avg_targets = np.mean(eval_unique_targets)
                    avg_coverage = np.mean(eval_platform_coverage)

                    # 综合评分：奖励为主，目标覆盖率为辅
                    # 如果奖励相近，优先选择目标分配更多的模型
                    composite_score = avg_reward + (avg_targets / config.N_TARGETS) * 2.0

                    print(f"   评估结果: 奖励={avg_reward:.3f}±{std_reward:.3f}, "
                          f"目标数={avg_targets:.1f}, 覆盖率={avg_coverage:.1f}")
                    print(f"   综合评分: {composite_score:.3f}")

                    # 2. 更新滑动窗口历史
                    self.reward_window.append({
                        'episode': current_ep,
                        'avg_reward': avg_reward,
                        'std_reward': std_reward,
                        'avg_targets': avg_targets,
                        'composite_score': composite_score,
                        'assignment': temp_assignment.copy()
                    })

                    # 3. 判断是否为"窗口内最佳"（最近500回合的最佳）
                    window_best = max(self.reward_window, key=lambda x: x['composite_score'])
                    is_window_best = (current_ep == window_best['episode'])

                    # 4. 保存逻辑：同时维护全局最佳和窗口最佳
                    saved_models = []

                    # A. 保存全局最佳模型（原有逻辑）
                    if status.get('new_best') and config.SAVE_BEST:
                        global_path = 'best_model_checkpoint.pth'
                        torch.save({
                            'actor': self.system.masac.actor.state_dict(),
                            'critic': self.system.masac.critic.state_dict(),
                            'encoder': self.system.encoder.state_dict(),
                            'episode': current_ep,
                            'reward': avg_reward,
                            'avg_targets': avg_targets,
                            'assignment': temp_assignment,
                            'type': 'global_best'
                        }, global_path)
                        saved_models.append(f"全局最佳: {global_path}")

                    # B. 保存窗口最佳模型（新增核心逻辑）
                    if is_window_best:
                        self.recent_best_reward = avg_reward
                        self.recent_best_episode = current_ep
                        self.recent_best_assignment = temp_assignment.copy()

                        # 使用带评分的文件名，方便识别
                        window_path = f'recent_best_ep{current_ep}_r{avg_reward:.2f}_t{avg_targets:.0f}.pth'
                        torch.save({
                            'actor': self.system.masac.actor.state_dict(),
                            'critic': self.system.masac.critic.state_dict(),
                            'encoder': self.system.encoder.state_dict(),
                            'episode': current_ep,
                            'reward': avg_reward,
                            'std_reward': std_reward,
                            'avg_targets': avg_targets,
                            'avg_coverage': avg_coverage,
                            'assignment': temp_assignment,
                            'optimizer_actor': self.system.masac.actor_opt.state_dict(),
                            'optimizer_critic': self.system.masac.critic_opt.state_dict(),
                            'window_start': current_ep - 500,  # 记录窗口范围
                            'window_end': current_ep,
                            'type': 'window_best'
                        }, window_path)
                        saved_models.append(f"窗口最佳: {window_path}")
                        print(f"\n🏆 新窗口最佳模型! (最近{self.window_size}回合内最佳)")

                    # C. 每500回合都保存一个"当前状态"检查点（可选，用于恢复训练）
                    checkpoint_path = f'checkpoint_ep{current_ep}.pth'
                    torch.save({
                        'actor': self.system.masac.actor.state_dict(),
                        'critic': self.system.masac.critic.state_dict(),
                        'encoder': self.system.encoder.state_dict(),
                        'episode': current_ep,
                        'current_reward': avg_reward,
                        'epsilon': self.system.masac.epsilon,
                        'optimizer_actor': self.system.masac.actor_opt.state_dict(),
                        'optimizer_critic': self.system.masac.critic_opt.state_dict(),
                        'monitor_state': {
                            'best_reward': monitor.best_reward,
                            'patience_counter': monitor.patience_counter,
                            'plateau_counter': getattr(monitor, 'plateau_counter', 0)
                        },
                        'type': 'regular_checkpoint'
                    }, checkpoint_path)
                    saved_models.append(f"常规检查点: {checkpoint_path}")
                    self.recent_checkpoints.append(checkpoint_path)

                    # 5. 打印保存摘要
                    print(f"\n💾 模型保存摘要 (回合 {current_ep}):")
                    for model_info in saved_models:
                        print(f"   ├─ {model_info}")

                    # 6. 清理旧文件：只保留最近3个窗口最佳和最近2个常规检查点
                    import glob
                    import os

                    # 清理旧窗口最佳（保留最近3个）
                    window_models = sorted(glob.glob('recent_best_ep*.pth'),
                                           key=lambda x: int(x.split('_')[2].replace('ep', '')))
                    if len(window_models) > 3:
                        for old_model in window_models[:-3]:
                            try:
                                os.remove(old_model)
                                print(f"   └─ 清理旧窗口模型: {old_model}")
                            except:
                                pass

                    # 清理旧常规检查点（保留最近2个）
                    if len(self.recent_checkpoints) > 2:
                        old_ckpt = self.recent_checkpoints.pop(0)
                        try:
                            if os.path.exists(old_ckpt) and old_ckpt != checkpoint_path:
                                os.remove(old_ckpt)
                                print(f"   └─ 清理旧检查点: {old_ckpt}")
                        except:
                            pass

                    # 7. 打印窗口统计
                    if len(self.reward_window) > 50:
                        window_rewards = [r['avg_reward'] for r in self.reward_window]
                        window_targets = [r['avg_targets'] for r in self.reward_window]
                        print(f"\n📈 滑动窗口统计 (最近{len(self.reward_window)}回合):")
                        print(f"   ├─ 奖励范围: {min(window_rewards):.2f} ~ {max(window_rewards):.2f}")
                        print(f"   ├─ 平均目标数: {np.mean(window_targets):.1f}/{config.N_TARGETS}")
                        print(f"   ├─ 窗口最佳回合: {window_best['episode']}")
                        print(f"   └─ 窗口最佳评分: {window_best['composite_score']:.2f}")

                    # ================================================================

                    if (episode + 1) % 1000 == 0:
                        temp_assign, _, _ = self.system.assign(evaluate=True)
                        self.visualize_3d(temp_assign)

                # 实时进度图（每100回合）
                self._plot_realtime(monitor, episode)

            # 早停判断（仅当手动触发过且模型稳定时）
            if status.get('should_stop', False) and episode >= config.MIN_EPISODES:
                print(f"✅ 训练完成（回合{episode}），使用当前最新模型进行最终评估")
                break

        # 训练结束：保存最终的窗口最佳模型（最近500回合的最佳）
        if self.recent_best_episode > 0:
            final_window_path = 'final_window_best.pth'
            print(f"\n🏆 训练结束，保存最终窗口最佳模型:")
            print(f"   来自回合: {self.recent_best_episode}")
            print(f"   评估奖励: {self.recent_best_reward:.3f}")
            print(
                f"   目标分配: {len(set(tid for tids in self.recent_best_assignment.values() for tid in tids)) if self.recent_best_assignment else 0}/{config.N_TARGETS}")

            # 创建一个指向最近最佳的快捷方式/副本
            latest_window = max(glob.glob('recent_best_ep*.pth'),
                                key=lambda x: int(x.split('_')[2].replace('ep', ''))) if glob.glob(
                'recent_best_ep*.pth') else None
            if latest_window and os.path.exists(latest_window):
                import shutil
                shutil.copy(latest_window, final_window_path)
                print(f"   已复制到: {final_window_path}")

        print("\n训练完成!")
        self.plot_results()
        final_assignment = self._final_evaluation()
        print(self.system.format_assignment(final_assignment))
        # 最终评估
        return self._final_evaluation()

    def _final_evaluation(self):
        """最终评估（评估20次采样取最佳，防止偶然性）"""
        print(f"\n{'=' * 60}")
        print("执行最终评估（20次采样取最佳）...")

        best_final_assignment = None
        best_coverage = 0
        best_reward = -1e10

        for eval_i in range(20):
            self.env.reset()
            assignment, _, _ = self.system.assign(evaluate=True)
            reward = self.system.calc_reward(assignment)
            coverage = len(set(tid for tids in assignment.values() for tid in tids)) if assignment else 0

            # 优先选覆盖目标多的；覆盖相同则选奖励高的
            if coverage > best_coverage or (coverage == best_coverage and reward > best_reward):
                best_coverage = coverage
                best_reward = reward
                best_final_assignment = assignment.copy()

        # 用最好的结果输出
        coverage = len(best_final_assignment) if best_final_assignment else 0
        unique_targets = best_coverage
        print(f"评估结果: {coverage}个平台有任务, 覆盖{unique_targets}/{len(self.env.targets)}个目标")

        # 打印分配详情
        for pid in range(len(self.env.platforms)):
            if pid in best_final_assignment:
                tids = best_final_assignment[pid]
                if not isinstance(tids, list):
                    tids = [tids]
                for tid in tids:
                    target = self.env.targets[tid]
                    print(f"   P{pid}({self.env.platforms[pid].type}) → T{tid}({target.task_type})")
            else:
                print(f"   P{pid}({self.env.platforms[pid].type}) → [未分配]")
        print(f"{'=' * 60}")

        # 生成图表（保持不变）
        total_episodes = len(self.history['assignments_detail'])
        display_episodes = min(2000, total_episodes)
        self._plot_assignment_gantt(total_episodes, recent_episodes=display_episodes)
        self._plot_temporal_dependencies_simple(best_final_assignment, total_episodes)
        self.visualize_3d(best_final_assignment)

        return best_final_assignment

    def _plot_training_report(self, episode: int, monitor: ConvergenceMonitor):
        """生成每500回合的综合训练报告图表"""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec

        # 创建大图 (16x12英寸，高清)
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 颜色主题
        colors = {
            'primary': '#2E86AB',  # 蓝色
            'secondary': '#A23B72',  # 紫红色
            'success': '#06A77D',  # 绿色
            'warning': '#F18F01',  # 橙色
            'danger': '#C1121F'  # 红色
        }

        # ===== 子图1: 奖励曲线 =====
        ax1 = fig.add_subplot(gs[0, :])  # 占据第一行整行
        rewards = self.history['rewards']

        if len(rewards) > 0:
            episodes_axis = np.arange(len(rewards))
            ax1.plot(episodes_axis, rewards, alpha=0.3, color=colors['primary'], label='原始奖励', linewidth=0.5)

            # 移动平均平滑
            if len(rewards) >= 50:
                window = min(100, len(rewards) // 5)
                smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
                ax1.plot(np.arange(window - 1, len(rewards)), smoothed,
                         color=colors['primary'], linewidth=2, label=f'{window}回合平滑')

            # 标记最佳奖励
            if monitor.best_reward > -1e9:
                ax1.axhline(y=monitor.best_reward, color=colors['success'],
                            linestyle='--', linewidth=1.5, alpha=0.7,
                            label=f'历史最佳: {monitor.best_reward:.3f}')

            # 标记当前窗口平均
            if len(monitor.reward_history) > 0:
                current_mean = np.mean(list(monitor.reward_history))
                ax1.axhline(y=current_mean, color=colors['secondary'],
                            linestyle=':', linewidth=1.5, alpha=0.7,
                            label=f'当前窗口: {current_mean:.3f}')

        ax1.set_xlabel('回合数', fontsize=11)
        ax1.set_ylabel('奖励', fontsize=11)
        ax1.set_title(f'训练奖励曲线 (截至回合 {episode})', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ===== 子图2: 平台覆盖率 =====
        ax2 = fig.add_subplot(gs[1, 0])
        coverage = self.history['platform_coverage']

        if len(coverage) > 0:
            ax2.plot(coverage, color=colors['warning'], linewidth=1.5)
            ax2.fill_between(range(len(coverage)), coverage, alpha=0.3, color=colors['warning'])

            # 标记全分配线
            ax2.axhline(y=config.N_PLATFORMS, color=colors['success'],
                        linestyle='--', linewidth=2, label=f'目标: {config.N_PLATFORMS}平台')

            # 标记当前值
            current_coverage = coverage[-1]
            color = colors['success'] if current_coverage == config.N_PLATFORMS else colors['danger']
            ax2.scatter([len(coverage) - 1], [current_coverage], color=color, s=100, zorder=5)

        ax2.set_xlabel('回合数', fontsize=11)
        ax2.set_ylabel('分配平台数', fontsize=11)
        ax2.set_title('平台分配覆盖率', fontsize=12, fontweight='bold')
        ax2.set_ylim(-0.5, config.N_PLATFORMS + 1)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ===== 子图3: 目标分配数 =====
        ax3 = fig.add_subplot(gs[1, 1])
        target_cov = self.history['target_coverage']

        if len(target_cov) > 0:
            ax3.plot(target_cov, color=colors['secondary'], linewidth=1.5)
            ax3.fill_between(range(len(target_cov)), target_cov, alpha=0.3, color=colors['secondary'])

            # 标记目标线
            ax3.axhline(y=config.N_TARGETS, color=colors['success'],
                        linestyle='--', linewidth=2, label=f'目标: {config.N_TARGETS}目标')

            current_t = target_cov[-1]
            ax3.scatter([len(target_cov) - 1], [current_t], color=colors['primary'], s=100, zorder=5)

        ax3.set_xlabel('回合数', fontsize=11)
        ax3.set_ylabel('分配目标数', fontsize=11)
        ax3.set_title('目标分配数量', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # ===== 子图4: 网络损失 =====
        ax4 = fig.add_subplot(gs[2, 0])
        if len(self.history['critic_losses']) > 0:
            ax4.plot(self.history['critic_losses'], label='Critic损失',
                     color=colors['primary'], linewidth=1.5)
            ax4.plot(self.history['actor_losses'], label='Actor损失',
                     color=colors['secondary'], linewidth=1.5, alpha=0.8)
            ax4.set_yscale('log')  # 对数坐标更好观察
        ax4.set_xlabel('更新步数', fontsize=11)
        ax4.set_ylabel('损失 (log)', fontsize=11)
        ax4.set_title('网络训练损失', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)

        # ===== 子图5: 探索率衰减 =====
        ax5 = fig.add_subplot(gs[2, 1])
        if len(self.history['epsilons']) > 0:
            ax5.plot(self.history['epsilons'], color=colors['danger'], linewidth=2)
            ax5.fill_between(range(len(self.history['epsilons'])),
                             self.history['epsilons'], alpha=0.2, color=colors['danger'])

            # 标记当前探索率
            current_eps = self.history['epsilons'][-1]
            ax5.text(len(self.history['epsilons']) - 1, current_eps,
                     f'{current_eps:.3f}', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax5.set_xlabel('回合数', fontsize=11)
        ax5.set_ylabel('Epsilon', fontsize=11)
        ax5.set_title('探索率衰减曲线', fontsize=12, fontweight='bold')
        ax5.set_ylim(0, 0.6)
        ax5.grid(True, alpha=0.3)

        # 添加总标题
        fig.suptitle(f'智能空面协同任务分配 - 训练进度报告 (回合 {episode})',
                     fontsize=15, fontweight='bold', y=0.98)

        # 添加统计摘要文本框
        stats_text = (
            f"最近100回合统计:\n"
            f"• 平均奖励: {np.mean(rewards[-100:]):.3f}\n"
            f"• 平台覆盖: {np.mean(coverage[-100:]):.1f}/{config.N_PLATFORMS}\n"
            f"• 目标分配: {np.mean(target_cov[-100:]):.1f}/{config.N_TARGETS}\n"
            f"• 探索率(Epsilon): {self.history['epsilons'][-1]:.3f}\n"
            f"• 历史最佳: {monitor.best_reward:.3f} (回合{monitor.best_episode})"
        )

        fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

        # 保存图片
        filename = f'training_report_ep{episode}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"\n📊 训练报告已生成: {filename}")
        print(f"   └─ 包含奖励曲线、覆盖率、损失、探索率等{len(self.history)}项指标")

    def _plot_realtime(self, monitor, episode):
        """实时绘制训练曲线 - 第四步新增"""
        if episode % 100 != 0 or episode == 0:
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            plt.figure(figsize=(12, 4))

            # 奖励曲线
            plt.subplot(1, 3, 1)
            rewards = list(monitor.reward_history)
            if len(rewards) > 0:
                plt.plot(rewards, alpha=0.3, color='blue', label='Raw')
                if len(rewards) > 20:
                    # 移动平均
                    window = min(20, len(rewards) // 2)
                    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
                    plt.plot(range(window - 1, len(rewards)), smoothed, 'r-', linewidth=2, label=f'MA({window})')
                plt.axhline(y=monitor.best_reward, color='green', linestyle='--',
                            label=f'Best:{monitor.best_reward:.2f}')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title(f'Reward Trend (Last {len(rewards)})')
                plt.legend()
                plt.grid(True)

            # 收敛判断
            plt.subplot(1, 3, 2)
            if len(rewards) > 0:
                current_mean = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                plt.bar(['Current', 'Best'], [current_mean, monitor.best_reward],
                        color=['blue', 'green'])
                plt.ylabel('Reward')
                plt.title(f'Convergence: {monitor.get_trend()}')

            # 耐心计数器
            plt.subplot(1, 3, 3)
            patience_pct = monitor.patience_counter / config.PATIENCE
            colors = ['green' if patience_pct < 0.5 else 'yellow' if patience_pct < 0.8 else 'red']
            plt.bar(['Patience'], [patience_pct], color=colors)
            plt.ylim(0, 1.2)
            plt.axhline(y=1.0, color='red', linestyle='--', label='Stop Threshold')
            plt.title(f'Patience: {monitor.patience_counter}/{config.PATIENCE}')
            plt.legend()

            plt.tight_layout()
            filename = f'training_progress_ep{episode + 1}.png'
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"📊 进度图已保存: {filename}")
        except Exception as e:
            # 如果绘图失败不影响训练
            pass

    def visualize_3d(self, assignment: Dict[int, List[int]]):
        """3D可视化 - 多任务版本"""
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')

        # 设置背景
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)

        # 绘制简化地形
        try:
            step = max(1, len(self.terrain.grid_lons) // 50)
            lg, lt = np.meshgrid(self.terrain.grid_lons[::step], self.terrain.grid_lats[::step])
            X, Y, Z = np.zeros_like(lg), np.zeros_like(lt), np.zeros_like(lg)
            for i in range(lg.shape[0]):
                for j in range(lg.shape[1]):
                    local = self.terrain.geo2local(lg[i, j], lt[i, j])
                    X[i, j], Y[i, j], Z[i, j] = local[0], local[1], local[2]
            ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.3, rstride=1, cstride=1)
        except Exception as e:
            print(f"地形绘制警告: {e}")

        # 绘制威胁区域
        for threat in self.env.threats:
            if threat.type == 'radar':
                color, alpha = '#ff4444', 0.08
            else:
                color, alpha = '#888888', 0.12

            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi / 2, 15)
            x = threat.center[0] + threat.radius * np.outer(np.cos(u), np.sin(v))
            y = threat.center[1] + threat.radius * np.outer(np.sin(u), np.sin(v))
            z = threat.center[2] + threat.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, alpha=alpha, color=color, rstride=2, cstride=2)

        # 绘制平台
        for p in self.env.platforms:
            color = Platform.COLORS.get(p.type, 'blue')
            ax.scatter(*p.position, c=color, marker='o', s=400,
                       edgecolors='black', linewidth=2.5, alpha=0.95, zorder=10)

            n_tasks = len(assignment.get(p.id, []))  # ← 获取任务数
            label = f'P{p.id}\n({n_tasks}tasks)'  # ← 显示任务数
            ax.text(p.position[0], p.position[1], p.position[2] + 8000,
                    label, fontsize=10, fontweight='bold', zorder=15,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color, alpha=0.9))

        # 绘制目标
        for t in self.env.targets:
            color = Target.COLORS.get(t.task_type, 'gray')
            marker = Target.MARKERS.get(t.task_type, 'o')
            size = 200 + int(t.value * 200)
            ax.scatter(*t.position, c=color, marker=marker, s=size,
                       edgecolors='black', linewidth=2, alpha=0.9, zorder=10)
            ax.text(t.position[0], t.position[1], t.position[2] + 5000,
                    f'T{t.id}', fontsize=10, fontweight='bold', zorder=15,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, alpha=0.8))

        for t in self.env.targets:
            if t.predecessors:
                for pred_id in t.predecessors:
                    if pred_id < len(self.env.targets):
                        pred_pos = self.env.targets[pred_id].position
                        curr_pos = t.position
                        # 绘制从目标指向其前序任务的虚线（表示依赖关系）
                        ax.plot([curr_pos[0], pred_pos[0]],
                                [curr_pos[1], pred_pos[1]],
                                [curr_pos[2], pred_pos[2]],
                                'g--', linewidth=1.5, alpha=0.5, zorder=3)

        # 绘制分配连线
        for pid, tids in assignment.items():  # ← 遍历目标列表
            if pid >= len(self.env.platforms):
                continue
            p = self.env.platforms[pid]

            prev_pos = p.position  # 从平台位置开始
            for idx, tid in enumerate(tids):  # ← 遍历多个任务
                if tid >= len(self.env.targets):
                    continue
                t = self.env.targets[tid]

                # 绘制连线（从上一位置到当前目标）
                ax.plot([prev_pos[0], t.position[0]],
                        [prev_pos[1], t.position[1]],
                        [prev_pos[2], t.position[2]],
                        'k-', linewidth=2, alpha=0.6, zorder=5)

                # 绘制箭头
                mid = (prev_pos + t.position) / 2
                direction = (t.position - prev_pos)
                direction = direction / (np.linalg.norm(direction) + 1e-6) * 5000
                ax.quiver(mid[0], mid[1], mid[2],
                          direction[0], direction[1], direction[2],
                          color='darkred', arrow_length_ratio=0.3, linewidth=2, zorder=8)

                prev_pos = t.position  # ← 更新位置，准备绘制下一段
                direction = direction / (np.linalg.norm(direction) + 1e-6) * 8000
                ax.quiver(mid[0], mid[1], mid[2],
                          direction[0], direction[1], direction[2],
                          color='darkred', arrow_length_ratio=0.4, linewidth=3, zorder=8)

        # 设置坐标轴
        ax.set_xlim(0, config.MAP_SIZE_X)
        ax.set_ylim(0, config.MAP_SIZE_Y)
        ax.set_zlim(0, config.MAP_HEIGHT)
        ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=14, fontweight='bold')

        # 标题
        avg_time = np.mean(self.system.stats['times']) * 1000 if self.system.stats['times'] else 0
        total_tasks = sum(len(tids) for tids in assignment.values())
        unique_tasks = len(set(tid for tids in assignment.values() for tid in tids))

        title = (f'智能空面协同任务分配 - 多任务模式\n'
                 f'平台数: {len(assignment)} | 总分配: {total_tasks} | 唯一目标: {unique_tasks}/{config.N_TARGETS}')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # 图例
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=Platform.COLORS['侦察'],
                   markersize=12, label='侦察 (UAV)', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=Platform.COLORS['察打一体'],
                   markersize=12, label='察打一体 (UCAV)', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=Platform.COLORS['打击'],
                   markersize=12, label='打击 (USV)', markeredgecolor='black'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=Target.COLORS['打击'],
                   markersize=12, label='打击目标', markeredgecolor='black'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=Target.COLORS['侦察'],
                   markersize=12, label='侦察目标', markeredgecolor='black'),
            Line2D([0], [0], marker='d', color='w', markerfacecolor=Target.COLORS['察打一体'],
                   markersize=12, label='察打一体目标', markeredgecolor='black'),
            Line2D([0], [0], color='red', lw=3, alpha=0.3, label='威胁区域'),
            Line2D([0], [0], color='black', lw=2, label='分配路径'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()
        plt.savefig('assignment_3d_v4_fixed.png', dpi=300, bbox_inches='tight')
        print("Saved: assignment_3d_v4_fixed.png")
        plt.show()
        plt.close(fig)  # 关键：释放内存，防止泄漏

    def _plot_temporal_dependencies_simple(self, assignment: Dict[int, int], episode: int):
        """
        简化版任务时序依赖验证图（无需NetworkX）
        展示任务分配情况及前序-后续关系
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if not assignment:
            print("⚠️ 无分配数据，跳过时序依赖图")
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        assigned_targets = []
        platform_to_target = {}
        for pid, tids in assignment.items():
            if isinstance(tids, list):
                assigned_targets.extend(tids)
                for tid in tids:
                    platform_to_target[tid] = pid
            else:
                assigned_targets.append(tids)
                platform_to_target[tids] = pid

        # 按任务类型分组布局
        task_types = ['electronic', 'reconnaissance', 'attack']  # 战术顺序：电子→侦查→攻击
        y_positions = {t: i for i, t in enumerate(task_types)}
        colors = {
            'electronic': '#9467bd',  # 紫色
            'reconnaissance': '#ffd700',  # 黄色
            'attack': '#d62728'  # 红色
        }

        # 统计每个任务类型的数量用于X轴分布
        type_counts = {t: 0 for t in task_types}
        type_indices = {t: [] for t in task_types}

        for tid in assigned_targets:
            t_type = self.env.targets[tid].task_type
            if t_type in type_counts:
                type_counts[t_type] += 1
                type_indices[t_type].append(tid)

        # 绘制任务节点（按类型分层）
        node_positions = {}  # 记录每个任务的位置用于连线

        for t_type in task_types:
            tids = type_indices[t_type]
            n = len(tids)
            if n == 0:
                continue

            # 在Y轴上均匀分布
            y_base = y_positions[t_type]

            for i, tid in enumerate(sorted(tids)):
                # X轴根据索引均匀分布，避免重叠
                x = i * (10.0 / max(n, 1)) + 1
                y = y_base + (i % 2) * 0.1  # 轻微交错避免重叠

                node_positions[tid] = (x, y)

                # 查找分配平台
                assigned_platform = None
                for pid, t in assignment.items():
                    if t == tid:
                        assigned_platform = pid
                        break

                # 绘制节点
                color = colors.get(t_type, 'gray')
                ax.scatter([x], [y], s=800, c=color, alpha=0.8,
                           edgecolors='black', linewidth=2, zorder=5)

                # 标注：任务ID + 平台ID
                label = f'T{tid}\nP{assigned_platform}'
                ax.annotate(label, (x, y), ha='center', va='center',
                            fontsize=9, fontweight='bold', zorder=6)

                # 标注前序任务（如果有）
                target = self.env.targets[tid]
                if target.predecessors:
                    pred_text = f"前序: {target.predecessors}"
                    ax.annotate(pred_text, (x, y - 0.25), ha='center', va='top',
                                fontsize=7, color='darkblue', alpha=0.7)

        # 绘制依赖连线（前序→当前）
        for tid in assigned_targets:
            target = self.env.targets[tid]
            if target.predecessors and tid in node_positions:
                x_end, y_end = node_positions[tid]

                for pred_id in target.predecessors:
                    if pred_id in node_positions:  # 只画已分配的依赖
                        x_start, y_start = node_positions[pred_id]

                        # 判断约束是否满足（前序是否已完成）
                        is_satisfied = self.env.targets[pred_id].completed
                        line_color = 'green' if is_satisfied else 'red'
                        line_style = '-' if is_satisfied else '--'
                        line_width = 2 if is_satisfied else 2.5

                        # 绘制箭头（从下到上，表示时序流向）
                        ax.annotate('', xy=(x_end, y_end + 0.15), xytext=(x_start, y_start - 0.15),
                                    arrowprops=dict(arrowstyle='->', color=line_color,
                                                    lw=line_width, ls=line_style,
                                                    connectionstyle='arc3,rad=0.1'),
                                    zorder=4)

                        # 添加约束标签（仅对违反的）
                        if not is_satisfied:
                            mid_x = (x_start + x_end) / 2
                            mid_y = (y_start + y_end) / 2
                            ax.text(mid_x, mid_y, '未满足!', fontsize=8,
                                    color='red', ha='center',
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        # 设置坐标轴
        ax.set_xlim(-1, 12)
        ax.set_ylim(-0.5, 3.5)
        ax.set_yticks(range(3))
        ax.set_yticklabels(['电子战任务\n(Electronic)', '侦查任务\n(Reconnaissance)', '攻击任务\n(Attack)'])
        ax.set_ylabel('任务类型层级（战术逻辑：电子→侦查→攻击）', fontsize=12)
        ax.set_xlabel('任务分布（同一类型内均匀分布）', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        # 添加战术流程说明
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=1.5, color='gray', linestyle=':', alpha=0.5)

        # 标题和图例
        ax.set_title(f'任务时序依赖验证图 (回合 {episode})\n'
                     f'共分配{len(assignment)}个平台→{len(set(assigned_targets))}个唯一目标 | '
                     f'连线表示"必须在...之前完成"',
                     fontsize=14, fontweight='bold', pad=20)

        legend_elements = [
            mpatches.Patch(facecolor=colors['electronic'], label='电子战', edgecolor='black'),
            mpatches.Patch(facecolor=colors['reconnaissance'], label='侦查', edgecolor='black'),
            mpatches.Patch(facecolor=colors['attack'], label='攻击', edgecolor='black'),
            plt.Line2D([0], [0], color='green', linewidth=2, label='时序约束已满足'),
            plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='时序约束违反')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

        # 添加统计文本框
        satisfied_count = 0
        total_constraints = 0
        for tid in assigned_targets:
            target = self.env.targets[tid]
            if target.predecessors:
                for pid in target.predecessors:
                    total_constraints += 1
                    if self.env.targets[pid].completed:
                        satisfied_count += 1

        satisfaction_rate = satisfied_count / total_constraints if total_constraints > 0 else 1.0

        stats_text = (f'时序统计:\n'
                      f'• 总约束数: {total_constraints}\n'
                      f'• 已满足: {satisfied_count}\n'
                      f'• 满足率: {satisfaction_rate:.1%}\n'
                      f'• 分配方案: {len(assignment)}平台/{len(set(assigned_targets))}目标')

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        filename = f'temporal_dependency_ep{episode}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def _plot_assignment_gantt(self, episode: int, recent_episodes: int = 500):
        """
        生成任务分配时序甘特图
        展示最近N个回合中各平台的任务分配历史
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        # 获取最近的数据
        start_idx = max(0, len(self.history['assignments_detail']) - recent_episodes)
        end_idx = len(self.history['assignments_detail'])

        if start_idx >= end_idx:
            print("数据不足，无法生成甘特图")
            return

        recent_assignments = self.history['assignments_detail'][start_idx:end_idx]
        episodes_range = list(range(start_idx, end_idx))

        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})

        # 颜色映射
        task_colors = {
            'attack': '#d62728',  # 红色
            'reconnaissance': '#ffd700',  # 黄色
            'electronic': '#9467bd'  # 紫色
        }

        # ===== 子图1: 平台-任务分配时序图 =====
        ax1 = axes[0]

        # 为每个平台绘制时间线
        for platform_id in range(config.N_PLATFORMS):
            y_pos = platform_id

            # 收集该平台的所有分配事件
            assignments_x = []  # 回合数
            tasks_color = []  # 颜色
            task_ids = []  # 任务ID

            for idx, assignment in enumerate(recent_assignments):
                episode_num = start_idx + idx
                if platform_id in assignment:
                    target_ids = assignment[platform_id]
                    if not isinstance(target_ids, list):
                        target_ids = [target_ids]
                    for target_id in target_ids:
                        if target_id < len(self.env.targets):
                            target = self.env.targets[target_id]
                            assignments_x.append(episode_num)
                            tasks_color.append(task_colors.get(target.task_type, 'gray'))
                            task_ids.append(target_id)
            if assignments_x:
                # 绘制散点（甘特风格）
                y_positions = [y_pos] * len(assignments_x)
                ax1.scatter(assignments_x, y_positions,
                            c=tasks_color, s=120, alpha=0.8,
                            edgecolors='black', linewidth=0.5, zorder=5)

                # 稀疏标注任务ID（每3个点标注一次）
                for i, (x, tid) in enumerate(zip(assignments_x, task_ids)):
                    if i % 3 == 0 and len(assignments_x) > 5:  # 稀疏标注避免拥挤
                        ax1.annotate(f'T{tid}', (x, y_pos),
                                     textcoords="offset points",
                                     xytext=(0, 12), ha='center',
                                     fontsize=7, alpha=0.8)
                    elif len(assignments_x) <= 5:  # 点数少时全标注
                        ax1.annotate(f'T{tid}', (x, y_pos),
                                     textcoords="offset points",
                                     xytext=(0, 12), ha='center',
                                     fontsize=7, alpha=0.8)

        # 添加平台分隔线
        for i in range(config.N_PLATFORMS - 1):
            ax1.axhline(y=i + 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

        # 标记当前回合
        ax1.axvline(x=episode, color='red', linestyle='-', linewidth=2, alpha=0.6,
                    label='当前回合', zorder=1)

        # 设置标签
        ax1.set_xlabel('训练回合', fontsize=12)
        ax1.set_ylabel('平台ID', fontsize=12)
        ax1.set_title(f'平台任务分配时序图 (回合 {start_idx}-{episode})\n'
                      f'每个点代表该平台在该回合被分配的任务',
                      fontsize=14, fontweight='bold')
        ax1.set_yticks(range(config.N_PLATFORMS))
        ax1.set_yticklabels([f'P{i} ({self.env.platforms[i].type})'
                             for i in range(config.N_PLATFORMS)])
        ax1.grid(True, alpha=0.3, axis='x')

        # 图例
        legend_elements = [
            mpatches.Patch(facecolor=task_colors['attack'], label='攻击任务', edgecolor='black'),
            mpatches.Patch(facecolor=task_colors['reconnaissance'], label='侦查任务', edgecolor='black'),
            mpatches.Patch(facecolor=task_colors['electronic'], label='电子战', edgecolor='black'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='当前回合位置')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

        # ===== 子图2: 约束满足率与覆盖率 =====
        ax2 = axes[1]

        # 计算每回合的指标
        coverage_rates = []
        temporal_violations = []

        for idx in range(start_idx, end_idx):
            assignment = self.history['assignments_detail'][idx]
            # 平台覆盖率
            coverage = len(assignment) / config.N_PLATFORMS
            coverage_rates.append(coverage)

            # 时序违反数（从history获取）
            if idx < len(self.history['temporal_violations']):
                temporal_violations.append(self.history['temporal_violations'][idx])
            else:
                temporal_violations.append(0)

        x_axis = episodes_range

        # 绘制平台覆盖率
        ax2.plot(x_axis, coverage_rates, label='平台覆盖率',
                 color='#2E86AB', linewidth=2, marker='o', markersize=4)
        ax2.fill_between(x_axis, coverage_rates, alpha=0.2, color='#2E86AB')

        # 绘制时序违反数（次坐标轴）
        ax2_twin = ax2.twinx()
        ax2_twin.bar(x_axis, temporal_violations, alpha=0.3, color='#C1121F',
                     label='时序违反数', width=1.0)
        ax2_twin.set_ylabel('时序违反次数', fontsize=11, color='#C1121F')

        # 标记100%线
        ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='100%覆盖')

        ax2.set_xlabel('训练回合', fontsize=12)
        ax2.set_ylabel('平台覆盖率', fontsize=12)
        ax2.set_title('约束满足监控（覆盖率应趋近100%，违反数应趋近0）',
                      fontsize=12, fontweight='bold')
        ax2.set_ylim(-0.05, 1.05)

        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax2.grid(True, alpha=0.3)

        # 添加统计摘要
        recent_coverage = np.mean(coverage_rates[-100:]) if len(coverage_rates) >= 100 else np.mean(coverage_rates)
        recent_violations = np.mean(temporal_violations[-100:]) if len(temporal_violations) >= 100 else np.mean(
            temporal_violations)

        stats_text = (f'最近{min(len(coverage_rates), 100)}回合平均:\n'
                      f'• 平台覆盖率: {recent_coverage:.1%}\n'
                      f'• 时序违反: {recent_violations:.1f}次/回合')

        ax2.text(0.02, 0.05, stats_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        filename = f'assignment_gantt_ep{episode}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("=" * 70)
    print("Intelligent Air-Surface Coordination System (Fixed v4.3)")
    print("=" * 70)

    # ========== 外部输入示例 ==========
    platforms_input = [
        {'type': '侦察', 'payload': 6, 'max_range': 1000e3, 'pos': [66000, 46000, 13000]},
        {'type': '侦察', 'payload': 6, 'max_range': 1000e3, 'pos': [45000, 45000, 13000]},
        {'type': '侦察', 'payload': 6, 'max_range': 1000e3, 'pos': [58000, 48000, 13000]},
        {'type': '打击', 'payload': 8, 'max_range': 1000e3, 'pos': [20000, 60000, 12000]},
        {'type': '打击', 'payload': 8, 'max_range': 1000e3, 'pos': [51000, 31000, 12000]},
        {'type': '察打一体', 'payload': 7, 'max_range': 1000e3, 'pos': [32000, 52000, 12500]},
        {'type': '察打一体', 'payload': 7, 'max_range': 1000e3, 'pos': [10000, 20000, 12500]},
        {'type': '察打一体', 'payload': 7, 'max_range': 1000e3, 'pos': [71000, 21000, 12500]},
        {'type': '察打一体', 'payload': 7, 'max_range': 1000e3, 'pos': [43000, 13000, 12500]},
        {'type': '察打一体', 'payload': 7, 'max_range': 1000e3, 'pos': [34000, 64000, 12500]},
    ]

    targets_input = [
        {'type': '侦察', 'value': 0.8, 'pos': [50000, 60000, 200], 'req_payload': 1, 'time_win_start': 0,
         'time_win_end': 500, 'predecessors': []},
        {'type': '侦察', 'value': 0.5, 'pos': [20000, 70000, 200], 'req_payload': 1, 'time_win_start': 0,
         'time_win_end': 300, 'predecessors': []},
        {'type': '侦察', 'value': 0.55, 'pos': [75000, 50000, 200], 'req_payload': 1, 'time_win_start': 0,
         'time_win_end': 450, 'predecessors': []},
        {'type': '侦察', 'value': 0.45, 'pos': [25000, 55000, 200], 'req_payload': 1, 'time_win_start': 0,
         'time_win_end': 350, 'predecessors': []},
        {'type': '侦察', 'value': 0.6, 'pos': [50000, 85000, 200], 'req_payload': 1, 'time_win_start': 0,
         'time_win_end': 400, 'predecessors': []},
        {'type': '侦察', 'value': 0.6, 'pos': [30000, 85000, 200], 'req_payload': 1, 'time_win_start': 0,
         'time_win_end': 400, 'predecessors': []},
        {'type': '察打一体', 'value': 0.9, 'pos': [60000, 70000, 200], 'req_payload': 2, 'time_win_start': 100,
         'time_win_end': 800, 'predecessors': []},
        {'type': '察打一体', 'value': 0.85, 'pos': [70000, 30000, 200], 'req_payload': 2, 'time_win_start': 200,
         'time_win_end': 900, 'predecessors': []},
        {'type': '察打一体', 'value': 0.95, 'pos': [55000, 20000, 200], 'req_payload': 2, 'time_win_start': 300,
         'time_win_end': 1000, 'predecessors': []},
        {'type': '察打一体', 'value': 0.8, 'pos': [45000, 35000, 200], 'req_payload': 2, 'time_win_start': 250,
         'time_win_end': 850, 'predecessors': []},
        {'type': '察打一体', 'value': 0.88, 'pos': [85000, 45000, 200], 'req_payload': 2, 'time_win_start': 150,
         'time_win_end': 750, 'predecessors': []},
        {'type': '察打一体', 'value': 0.7, 'pos': [40000, 50000, 200], 'req_payload': 2, 'time_win_start': 50,
         'time_win_end': 600, 'predecessors': []},
        {'type': '察打一体', 'value': 0.75, 'pos': [80000, 80000, 200], 'req_payload': 2, 'time_win_start': 150,
         'time_win_end': 700, 'predecessors': []},
        {'type': '察打一体', 'value': 0.65, 'pos': [35000, 80000, 200], 'req_payload': 2, 'time_win_start': 100,
         'time_win_end': 550, 'predecessors': []},
        {'type': '察打一体', 'value': 0.7, 'pos': [65000, 65000, 200], 'req_payload': 2, 'time_win_start': 50,
         'time_win_end': 500, 'predecessors': []},
        {'type': '察打一体', 'value': 0.72, 'pos': [15000, 45000, 200], 'req_payload': 2, 'time_win_start': 100,
         'time_win_end': 600, 'predecessors': []},
    ]

    # 创建环境并加载场景
    trainer = TerrainTrainer(use_real_terrain=False,
                             platform_configs=platforms_input,
                             target_configs=targets_input)
    # 开始训练
    final_assignment = trainer.train(episodes=2000)

    print("\n" + "=" * 70)
    print("Results generated:")
    print(" 1. training_results_v4_fixed.png - Training curves")
    print(" 2. assignment_3d_v4_fixed.png - 3D visualization")
    print("=" * 70)



