import json
import os

import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv

from Envs.terrain_heightmap import CSVTerrainMap

'''
直接输出路点的路径规划脚本,本脚本单位为米,面向小范围避障场景

3.20前的对应学习环境
'''

DEFAULT_FIXED_TARGET_AGL_KM = 0.2


class CylinderObstacle:
    def __init__(self, x, y, radius, height):
        self.kind = "cylinder"
        self.x = x
        self.y = y
        self.radius = radius
        self.height = height
        self.xy_extent = radius

    def contains(self, px, py, pz):
        if pz < 0 or pz > self.height:
            return False
        dist_sq = (px - self.x) ** 2 + (py - self.y) ** 2
        return dist_sq <= self.radius ** 2


class EllipsoidalHemisphereObstacle:
    def __init__(self, x, y, radius, height):
        self.kind = "hemisphere"
        self.x = x
        self.y = y
        self.z = 0.0
        self.radius = radius
        self.height = height
        self.xy_extent = radius

    def contains(self, px, py, pz):
        if pz < 0.0:
            return False
        normalized = (
            ((px - self.x) / max(self.radius, 1e-8)) ** 2
            + ((py - self.y) / max(self.radius, 1e-8)) ** 2
            + ((pz - self.z) / max(self.height, 1e-8)) ** 2
        )
        return normalized <= 1.0


class UAVWaypoint3DMAPFEnv(ParallelEnv):
    metadata = {"name": "uav_3d_mapf_waypoint_v0"}

    def __init__(
        self,
        space_dim=(100.0, 100.0, 50.0),
        n_agents=3,
        max_steps=200,
        dt=0.1,
        num_obstacles=3,
        hemisphere_fraction=0.5,

        obs_radius_range=(2.0, 8.0),
        obs_height_range=(10.0, 40.0),
        hemisphere_radius_range=(4.0, 10.0),
        hemisphere_height_range=None,

        waypoint_step_size=3.0, #步长

        #lidar模拟
        lidar_rays=24,
        lidar_range=20.0,

        #奖励函数相关
        goal_threshold=2.0,
        step_penalty=-0.01,
        progress_reward_scale=1.0,
        obstacle_collision_penalty=3.0,
        agent_collision_penalty=3.0,
        goal_reward=30.0,
        timeout_penalty_scale=2.0,
        collision_radius=2.0,
        progress_normalizer=None,
        render_mode=None,
        scenario_json="",
        terrain_enabled=True,
        terrain_csv_path=os.path.join("Envs", "jiangning_mountain_simplified_smoother.csv"),
        terrain_grid_resolution=1.0,
        terrain_origin=(0.0, 0.0),
        terrain_min_clearance=0.0,
        terrain_z_scale=0.001,
    ):
        super().__init__()
        self.X, self.Y, self.Z = space_dim
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.dt = dt
        self.num_obstacles = num_obstacles
        self.hemisphere_fraction = float(np.clip(hemisphere_fraction, 0.0, 1.0))
        self.obs_radius_range = obs_radius_range
        self.obs_height_range = obs_height_range
        self.hemisphere_radius_range = (
            obs_radius_range if hemisphere_radius_range is None else hemisphere_radius_range
        )
        self.hemisphere_height_range = (
            (obs_height_range[0] * 0.5, obs_height_range[1] * 0.5)
            if hemisphere_height_range is None
            else hemisphere_height_range
        )
        self.waypoint_step_size = waypoint_step_size
        self.lidar_rays = lidar_rays
        self.lidar_range = lidar_range
        self.goal_threshold = goal_threshold
        self.step_penalty = step_penalty
        self.progress_reward_scale = progress_reward_scale
        self.obstacle_collision_penalty = obstacle_collision_penalty
        self.agent_collision_penalty = agent_collision_penalty
        self.goal_reward = goal_reward
        self.timeout_penalty_scale = timeout_penalty_scale
        self.collision_radius = collision_radius
        self.render_mode = render_mode
        self.scenario_json = str(scenario_json or "").strip()
        self._scenario_payload = None

        self.terrain_enabled = bool(terrain_enabled)
        self.terrain_csv_path = terrain_csv_path
        self.terrain_grid_resolution = float(terrain_grid_resolution)
        self.terrain_origin = tuple(float(item) for item in terrain_origin)
        self.terrain_min_clearance = float(terrain_min_clearance)
        self.terrain_z_scale = float(terrain_z_scale)
        self.terrain = CSVTerrainMap(
            csv_path=self.terrain_csv_path,
            enabled=self.terrain_enabled,
            resolution=self.terrain_grid_resolution,
            origin=self.terrain_origin,
            min_clearance=self.terrain_min_clearance,
            target_extent=(self.X, self.Y),
            target_height=None,
            z_scale=self.terrain_z_scale,
        )

        self.progress_normalizer = (
            max(self.waypoint_step_size, 1e-6)
            if progress_normalizer is None
            else max(progress_normalizer, 1e-6)
        )
        self.workspace_diagonal = max(
            np.linalg.norm(np.array([self.X, self.Y, self.Z], dtype=np.float32)),
            1e-6,
        )

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = self.possible_agents.copy()

        # 动作为当前位置的角度偏移再乘以步长
        self.act_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_spaces = {agent: self.act_space for agent in self.possible_agents}

        # Observation: lidar + pos + last_motion + rel_target + others_rel + norm_target_dist.
        obs_dim = self.lidar_rays + 3 + 3 + 3 + (self.n_agents - 1) * 3 + 1
        self.obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_spaces = {agent: self.obs_space for agent in self.possible_agents}

        indices = np.arange(0, self.lidar_rays, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / self.lidar_rays)
        theta = np.pi * (1 + 5**0.5) * indices
        self.ray_dirs = np.vstack(
            [
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ]
        ).T

        self.step_cnt = 0
        self.obstacles = []
        self.all_obstacles = []
        self.start_target_min_distance = 10.0
        self.spawn_min_separation = 6.0

        self.agent_positions = {}
        self.agent_last_motion = {}
        self.agent_targets = {}
        self.agent_reached_targets = {}
        self.agent_dones = {}
        self.agent_obstacle_collisions = {}
        self.agent_inter_agent_collisions = {}

    def _load_scenario_payload(self):
        if self._scenario_payload is not None:
            return self._scenario_payload
        if not self.scenario_json:
            self._scenario_payload = {}
            return self._scenario_payload
        with open(self.scenario_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("scenario_json 必须是 JSON 对象。")
        self._scenario_payload = payload
        return self._scenario_payload

    def _build_obstacles_from_payload(self, obstacle_payloads):
        all_obstacles = []
        active_obstacles = []
        for item in list(obstacle_payloads or [])[: self.num_obstacles]:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", item.get("type", ""))).strip().lower()
            if kind == "cylinder":
                obstacle = CylinderObstacle(
                    float(item["x"]),
                    float(item["y"]),
                    float(item["radius"]),
                    float(item["height"]),
                )
                all_obstacles.append(obstacle)
                active_obstacles.append(obstacle)
            elif kind == "hemisphere":
                obstacle = EllipsoidalHemisphereObstacle(
                    float(item["x"]),
                    float(item["y"]),
                    float(item["radius"]),
                    float(item["height"]),
                )
                obstacle.z = float(item.get("z", 0.0))
                all_obstacles.append(obstacle)
        return all_obstacles, active_obstacles

    def _build_obstacles_from_xy(self, obstacle_xy):
        all_obstacles = []
        active_obstacles = []
        default_radius = float(sum(self.obs_radius_range) / 2.0)
        default_height = float(sum(self.obs_height_range) / 2.0)
        for point in list(obstacle_xy or [])[: self.num_obstacles]:
            if len(point) < 2:
                continue
            obstacle = CylinderObstacle(
                float(point[0]),
                float(point[1]),
                default_radius,
                default_height,
            )
            all_obstacles.append(obstacle)
            active_obstacles.append(obstacle)
        return all_obstacles, active_obstacles

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def _is_obstacle_overlapping(self, x, y, xy_extent):
        for obs in self.obstacles:
            min_dist = xy_extent + obs.xy_extent
            if (x - obs.x) ** 2 + (y - obs.y) ** 2 < min_dist ** 2:
                return True
        return False

    def _generate_obstacles(self):
        scenario_payload = self._load_scenario_payload()
        obstacle_payloads = scenario_payload.get("obstacles")
        if isinstance(obstacle_payloads, list) and obstacle_payloads:
            self.all_obstacles, self.obstacles = self._build_obstacles_from_payload(obstacle_payloads)
            return
        obstacle_xy = scenario_payload.get("obstacle_xy")
        if isinstance(obstacle_xy, list) and obstacle_xy:
            self.all_obstacles, self.obstacles = self._build_obstacles_from_xy(obstacle_xy)
            return

        self.obstacles = []
        self.all_obstacles = []
        max_attempts = max(100, self.num_obstacles * 50)
        attempts = 0
        num_hemispheres = int(round(self.num_obstacles * self.hemisphere_fraction))
        num_cylinders = self.num_obstacles - num_hemispheres
        obstacle_plan = (["cylinder"] * num_cylinders) + (["hemisphere"] * num_hemispheres)
        np.random.shuffle(obstacle_plan)
        plan_index = 0
        while plan_index < len(obstacle_plan) and attempts < max_attempts:
            attempts += 1
            obstacle_kind = obstacle_plan[plan_index]

            if obstacle_kind == "cylinder":
                radius = np.random.uniform(*self.obs_radius_range)
                height = np.random.uniform(*self.obs_height_range)
                x = np.random.uniform(radius, self.X - radius)
                y = np.random.uniform(radius, self.Y - radius)
                if self._is_obstacle_overlapping(x, y, radius):
                    continue
                obstacle = CylinderObstacle(x, y, radius, height)
                self.obstacles.append(obstacle)
                self.all_obstacles.append(obstacle)
            else:
                radius = np.random.uniform(*self.hemisphere_radius_range)
                height = np.random.uniform(*self.hemisphere_height_range)
                x = np.random.uniform(radius, self.X - radius)
                y = np.random.uniform(radius, self.Y - radius)
                if self._is_obstacle_overlapping(x, y, radius):
                    continue
                obstacle = EllipsoidalHemisphereObstacle(x, y, radius, height)
                self.obstacles.append(obstacle)
                self.all_obstacles.append(obstacle)

            plan_index += 1

    def _is_collision(self, px, py, pz):
        if px < 0 or px > self.X or py < 0 or py > self.Y or pz < 0 or pz > self.Z:
            return True
        if self.terrain_enabled and self.terrain.collides(px, py, pz):
            return True
        for obs in self.obstacles:
            if obs.contains(px, py, pz):
                return True
        return False

    def _get_random_free_pos(self):
        while True:
            x = np.random.uniform(0, self.X)
            y = np.random.uniform(0, self.Y)
            z_min = 0.0
            if self.terrain_enabled:
                z_min, _ = self.terrain.sample_height_bounds(x, y, clearance=0.0)
            z_min = max(float(z_min), 0.0)
            if z_min >= self.Z:
                continue
            z = np.random.uniform(z_min, self.Z)
            if not self._is_collision(x, y, z):
                return np.array([x, y, z], dtype=np.float32)

    def _sample_free_position(self, min_distance=0.0, existing_positions=None, max_attempts=500):
        existing_positions = existing_positions or []
        for _ in range(max_attempts):
            candidate = self._get_random_free_pos()
            if all(np.linalg.norm(candidate - pos) >= min_distance for pos in existing_positions):
                return candidate
        return self._get_random_free_pos()

    def _lift_position_to_free_height(self, position, step_size=1.0):
        candidate = np.asarray(position, dtype=np.float32).copy()
        candidate[0] = float(np.clip(candidate[0], 0.0, self.X))
        candidate[1] = float(np.clip(candidate[1], 0.0, self.Y))
        candidate[2] = float(np.clip(candidate[2], 0.0, self.Z))
        if not self._is_collision(candidate[0], candidate[1], candidate[2]):
            return candidate

        z = float(candidate[2])
        while z <= self.Z:
            if not self._is_collision(candidate[0], candidate[1], z):
                candidate[2] = float(z)
                return candidate
            z += max(float(step_size), 1e-3)

        raise ValueError("给定位置沿高度方向无法找到安全点。")

    def _build_agent_info(self, agent):
        remaining_distance = None
        if agent in self.agent_positions and agent in self.agent_targets:
            remaining_distance = float(
                np.linalg.norm(self.agent_targets[agent] - self.agent_positions[agent])
            )
        return {
            "reached_goal": self.agent_dones.get(agent, False),
            "collision_with_obstacle": self.agent_obstacle_collisions.get(agent, 0),
            "collision_with_agent": self.agent_inter_agent_collisions.get(agent, 0),
            "steps": self.step_cnt,
            "remaining_distance": remaining_distance,
            "all_agents_reached": all(self.agent_dones.values()) if self.agent_dones else False,
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents.copy()
        self.step_cnt = 0
        self.terrain.generate()
        self._generate_obstacles()
        scenario_payload = self._load_scenario_payload()

        self.agent_positions = {}
        self.agent_last_motion = {}
        self.agent_targets = {}
        self.agent_reached_targets = {}
        self.agent_dones = {}
        self.agent_obstacle_collisions = {}
        self.agent_inter_agent_collisions = {}

        existing_starts = []
        existing_targets = []
        configured_starts = scenario_payload.get("start_positions") if isinstance(scenario_payload, dict) else None
        configured_targets = None
        if isinstance(scenario_payload, dict):
            configured_targets = scenario_payload.get("target_positions")
            if configured_targets is None:
                configured_targets = scenario_payload.get("task_positions")
        task_height_mode = str(
            scenario_payload.get("task_height_mode", "absolute")
            if isinstance(scenario_payload, dict)
            else "absolute"
        ).strip().lower()
        if isinstance(configured_starts, list) and len(configured_starts) < len(self.agents):
            raise ValueError("scenario_json 中的 start_positions 数量不足。")
        if isinstance(configured_targets, list) and len(configured_targets) < len(self.agents):
            raise ValueError("scenario_json 中的 target_positions 数量不足。")
        for agent in self.agents:
            agent_idx = self.possible_agents.index(agent)
            if isinstance(configured_starts, list):
                pos = np.asarray(configured_starts[agent_idx], dtype=np.float32).copy()
                pos = self._lift_position_to_free_height(pos)
            else:
                pos = self._sample_free_position(
                    min_distance=self.spawn_min_separation,
                    existing_positions=existing_starts,
                )

            if isinstance(configured_targets, list):
                target = np.asarray(configured_targets[agent_idx], dtype=np.float32).copy()
                terrain_height = (
                    float(self.terrain.surface_height_at(target[0], target[1]))
                    if self.terrain_enabled
                    else 0.0
                )
                target[2] = terrain_height + float(DEFAULT_FIXED_TARGET_AGL_KM)
                target = self._lift_position_to_free_height(target)
            else:
                target = self._sample_free_position(
                    min_distance=self.spawn_min_separation,
                    existing_positions=existing_targets,
                )
            while np.linalg.norm(pos - target) < self.start_target_min_distance:
                target = self._sample_free_position(
                    min_distance=self.spawn_min_separation,
                    existing_positions=existing_targets,
                )

            self.agent_positions[agent] = pos
            self.agent_last_motion[agent] = np.zeros(3, dtype=np.float32)
            self.agent_targets[agent] = target
            self.agent_reached_targets[agent] = False
            self.agent_dones[agent] = False
            self.agent_obstacle_collisions[agent] = 0
            self.agent_inter_agent_collisions[agent] = 0
            existing_starts.append(pos)
            existing_targets.append(target)

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: self._build_agent_info(agent) for agent in self.agents}
        return observations, infos

    def _cast_rays(self, agent_pos):
        distances = np.full(self.lidar_rays, self.lidar_range, dtype=np.float32)
        step_size = 1.0
        num_steps = int(self.lidar_range / step_size)

        for i, direction in enumerate(self.ray_dirs):
            for step in range(1, num_steps + 1):
                test_pos = agent_pos + direction * (step * step_size)
                if self._is_collision(test_pos[0], test_pos[1], test_pos[2]):
                    distances[i] = step * step_size
                    break

        return distances / self.lidar_range

    def observe(self, agent):
        pos = self.agent_positions[agent]
        target = self.agent_targets[agent]
        lidar_data = self._cast_rays(pos)

        norm_pos = pos / np.array([self.X, self.Y, self.Z], dtype=np.float32)
        norm_motion = self.agent_last_motion[agent] / self.progress_normalizer
        target_vec = target - pos
        rel_target = target_vec / np.array([self.X, self.Y, self.Z], dtype=np.float32)
        target_dist = np.linalg.norm(target_vec)
        norm_target_dist = np.array([target_dist / self.workspace_diagonal], dtype=np.float32)

        others_rel = []
        for other in self.possible_agents:
            if other == agent:
                continue
            if other in self.agents:
                other_pos = self.agent_positions[other]
                others_rel.extend(
                    (other_pos - pos) / np.array([self.X, self.Y, self.Z], dtype=np.float32)
                )
            else:
                others_rel.extend([0.0, 0.0, 0.0])

        obs = np.concatenate(
            [
                lidar_data,
                norm_pos,
                norm_motion,
                rel_target,
                others_rel,
                norm_target_dist,
            ]
        ).astype(np.float32)
        return obs

    def step(self, actions):
        
        # 存储各种信息
        rewards = {
            agent: 0.0 if self.agent_reached_targets.get(agent, False) else self.step_penalty
            for agent in self.agents
        }
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        collision_happened = {agent: False for agent in self.agents}

        # 获取相对距离
        self.step_cnt += 1
        dist_before = {
            agent: np.linalg.norm(self.agent_targets[agent] - self.agent_positions[agent])
            for agent in self.agents
        }

        # 更新每个agent的状态
        for agent in self.agents:

            if self.agent_reached_targets.get(agent, False):    #到打目标
                self.agent_last_motion[agent] = np.zeros(3, dtype=np.float32)
                continue
            
            #输出路点
            action = np.asarray(actions[agent], dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            motion = action * self.waypoint_step_size
            target_waypoint = self.agent_positions[agent] + motion
            target_waypoint = np.clip(
                target_waypoint,
                [0.0, 0.0, 0.0],
                [self.X, self.Y, self.Z],
            )
            actual_motion = target_waypoint - self.agent_positions[agent]
            new_pos = self.agent_positions[agent] + actual_motion

            if self._is_collision(new_pos[0], new_pos[1], new_pos[2]):
                rewards[agent] -= self.obstacle_collision_penalty
                self.agent_obstacle_collisions[agent] += 1
                self.agent_last_motion[agent] = np.zeros(3, dtype=np.float32)
                collision_happened[agent] = True
            else:
                self.agent_positions[agent] = new_pos
                self.agent_last_motion[agent] = actual_motion.astype(np.float32)

        agent_list = list(self.agents)
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a1 = agent_list[i]
                a2 = agent_list[j]
                dist = np.linalg.norm(self.agent_positions[a1] - self.agent_positions[a2])
                if dist < self.collision_radius:
                    rewards[a1] -= self.agent_collision_penalty
                    rewards[a2] -= self.agent_collision_penalty
                    self.agent_inter_agent_collisions[a1] += 1
                    self.agent_inter_agent_collisions[a2] += 1
                    collision_happened[a1] = True
                    collision_happened[a2] = True
                    self.agent_last_motion[a1] = np.zeros(3, dtype=np.float32)
                    self.agent_last_motion[a2] = np.zeros(3, dtype=np.float32)

        for agent in self.agents:
            dist_after = np.linalg.norm(self.agent_targets[agent] - self.agent_positions[agent])
            step_progress = dist_before[agent] - dist_after
            progress = np.clip(step_progress / self.progress_normalizer, -1.0, 1.0)

            if not collision_happened[agent] and not self.agent_reached_targets.get(agent, False):
                rewards[agent] += self.progress_reward_scale * progress

            if (
                not self.agent_reached_targets.get(agent, False)
                and dist_after < self.goal_threshold
            ):
                rewards[agent] += self.goal_reward
                self.agent_reached_targets[agent] = True
                self.agent_dones[agent] = True
                self.agent_last_motion[agent] = np.zeros(3, dtype=np.float32)

        if all(self.agent_reached_targets.get(agent, False) for agent in self.possible_agents):
            for agent in self.agents:
                terminations[agent] = True

        if self.step_cnt >= self.max_steps: #触发truncated
            for agent in self.agents:
                if not self.agent_reached_targets.get(agent, False):
                    final_dist = np.linalg.norm(
                        self.agent_targets[agent] - self.agent_positions[agent]
                    )
                    rewards[agent] -= (
                        self.timeout_penalty_scale * final_dist / self.workspace_diagonal
                    )
                truncations[agent] = True

        self.agents = [
            agent for agent in self.agents if not terminations[agent] and not truncations[agent]
        ]

        for agent in infos:
            infos[agent] = self._build_agent_info(agent)

        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations, rewards, terminations, truncations, infos


# Backward-compatible alias so existing training scripts can switch files
# without needing to rename the imported class immediately.
UAVContinuous3DMAPFEnv = UAVWaypoint3DMAPFEnv
