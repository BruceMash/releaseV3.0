import os

import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from Envs.terrain_heightmap import CSVTerrainMap, TerrainHeightMap

'''
直接输出路点的路径规划环境
'''

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
    def __init__(self, x, y, radius, height, z=0.0):
        self.kind = "hemisphere"
        self.x = x
        self.y = y
        self.z = float(z)
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
    lidar_num_samples = 10

    def __init__(
        self,
        space_dim=(100.0, 100.0, 100.0),
        n_agents=3,
        max_steps=2500,
        dt=0.1,
        num_obstacles=10,
        hemisphere_fraction=0.5,
        obstacle_rule="primitives",

        obs_radius_range=(2.0, 8.0),
        obs_height_range=(10.0, 40.0),
        hemisphere_radius_range=(4.0, 10.0),
        hemisphere_height_range=(10.0, 10.0),
        terrain_enabled=False,
        terrain_source="procedural",
        terrain_csv_path=None,
        terrain_base_height=0.0,
        terrain_num_hemispheres=0,
        terrain_radius_range=(250.0, 400.0),
        terrain_height_range=(10.0, 10.0),
        terrain_resolution=(128, 128),
        terrain_grid_resolution=1.0,
        terrain_origin=(0.0, 0.0),
        terrain_min_clearance=0.0,
        terrain_max_clearance=None,
        terrain_match_space_dim=True,
        terrain_spawn_clearance=1.0,
        spawn_max_height=30.0,
        ground_target_clearance=0.001,
        terrain_csv_target_x=None,
        terrain_csv_target_y=None,
        terrain_csv_target_z=None,
        terrain_csv_z_scale=1.0,
        terrain_csv_simplify_stride=1,
        terrain_csv_smooth_passes=0,

        uav_speed = 0.06, # km/s
        decision_period = 1,

        #lidar模拟
        lidar_rays=24,
        lidar_range=0.2,
        lidar_step_size=None,

        #奖励函数相关
        goal_threshold=1.0,
        step_penalty=-0.001,
        progress_reward_scale=1.0,
        obstacle_collision_penalty=3.0,
        agent_collision_penalty=3.0,
        goal_reward=30.0,
        timeout_penalty_scale=2.0,
        collision_radius=0.02,
        near_goal_collision_free_radius=0.0,
        hemisphere_exclusion_buffer=5.0,
        avoid_mountain_basins=True,
        basin_window_radius=4,
        basin_highland_threshold=20.0,
        basin_relief_threshold=3.0,
        basin_high_neighbor_ratio=0.35,
        log_collisions=False,
        progress_normalizer=None,
        render_mode=None,
    ):
        super().__init__()
        # Core workspace and episode configuration. All geometric quantities below
        # live in the same scene unit system that the caller chooses for the task.
        self.X, self.Y, self.Z = space_dim
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.dt = dt
        self.obstacle_rule = obstacle_rule
        self.num_obstacles = num_obstacles
        self.hemisphere_fraction = float(np.clip(hemisphere_fraction, 0.0, 1.0))
        self.obs_radius_range = obs_radius_range
        self.obs_height_range = obs_height_range
        self.hemisphere_radius_range = (
            obs_radius_range if hemisphere_radius_range is None else hemisphere_radius_range
        )
        waypoint_step_size= uav_speed * decision_period #步长, 3km // 0.6km /s = 5s
        self.hemisphere_height_range = hemisphere_height_range
        self.terrain_source = terrain_source
        self.terrain_num_hemispheres = terrain_num_hemispheres
        self.terrain_radius_range = terrain_radius_range
        self.terrain_height_range = terrain_height_range
        self.terrain_spawn_clearance = terrain_spawn_clearance
        self.spawn_max_height = float(min(max(spawn_max_height, 0.0), self.Z))
        self.ground_target_clearance = float(ground_target_clearance)
        self.terrain_match_space_dim = terrain_match_space_dim
        self.waypoint_step_size = waypoint_step_size
        self.lidar_rays = lidar_rays
        self.lidar_range = lidar_range
        self.lidar_step_size = (
            self._default_lidar_step_size(terrain_enabled, terrain_grid_resolution)
            if lidar_step_size is None
            else max(float(lidar_step_size), 1e-3)
        )
        self.goal_threshold = goal_threshold
        self.step_penalty = step_penalty
        self.progress_reward_scale = progress_reward_scale
        self.obstacle_collision_penalty = obstacle_collision_penalty
        self.agent_collision_penalty = agent_collision_penalty
        self.goal_reward = goal_reward
        self.timeout_penalty_scale = timeout_penalty_scale
        self.collision_radius = collision_radius
        self.near_goal_collision_free_radius = max(float(near_goal_collision_free_radius), 0.0)
        self.hemisphere_exclusion_buffer = max(float(hemisphere_exclusion_buffer), 0.0)
        self.avoid_mountain_basins = bool(avoid_mountain_basins)
        self.basin_window_radius = max(int(basin_window_radius), 1)
        self.basin_highland_threshold = float(basin_highland_threshold)
        self.basin_relief_threshold = float(basin_relief_threshold)
        self.basin_high_neighbor_ratio = float(
            np.clip(basin_high_neighbor_ratio, 0.0, 1.0)
        )
        self.log_collisions = bool(log_collisions)
        self.render_mode = render_mode

        # Progress rewards are normalized by a characteristic single-step travel
        # distance so reward magnitudes remain comparable when speed settings vary.
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

        # 动作为当前位置三个维度应该前进的比重乘以最大步长
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
        self.terrain = self._build_terrain_backend(
            terrain_enabled=terrain_enabled,
            terrain_source=terrain_source,
            terrain_csv_path=terrain_csv_path,
            terrain_base_height=terrain_base_height,
            terrain_num_hemispheres=terrain_num_hemispheres,
            terrain_radius_range=terrain_radius_range,
            terrain_height_range=terrain_height_range,
            terrain_resolution=terrain_resolution,
            terrain_grid_resolution=terrain_grid_resolution,
            terrain_origin=terrain_origin,
            terrain_min_clearance=terrain_min_clearance,
            terrain_max_clearance=terrain_max_clearance,
            terrain_spawn_clearance=terrain_spawn_clearance,
            terrain_csv_target_x=terrain_csv_target_x,
            terrain_csv_target_y=terrain_csv_target_y,
            terrain_csv_target_z=terrain_csv_target_z,
            terrain_csv_z_scale=terrain_csv_z_scale,
            terrain_csv_simplify_stride=terrain_csv_simplify_stride,
            terrain_csv_smooth_passes=terrain_csv_smooth_passes,
        )
        self.start_target_min_distance = 10.0
        self.spawn_min_separation = 6.0

        self.agent_positions = {}
        self.agent_last_motion = {}
        self.agent_targets = {}
        self.agent_initial_distances = {}
        self.agent_reached_targets = {}
        self.agent_dones = {}
        self.agent_timeouts = {}
        self.agent_obstacle_collisions = {}
        self.agent_inter_agent_collisions = {}
        self.agent_collision_reasons = {}

        valid_obstacle_rules = {"primitives", "terrain", "hybrid"}
        if self.obstacle_rule not in valid_obstacle_rules:
            raise ValueError(
                f"Unsupported obstacle_rule={self.obstacle_rule!r}. "
                f"Expected one of {sorted(valid_obstacle_rules)}."
            )
        valid_terrain_sources = {"procedural", "csv"}
        if self.terrain_source not in valid_terrain_sources:
            raise ValueError(
                f"Unsupported terrain_source={self.terrain_source!r}. "
                f"Expected one of {sorted(valid_terrain_sources)}."
            )
        if terrain_enabled and self.terrain_source == "csv" and not terrain_csv_path:
            raise ValueError("terrain_csv_path is required when terrain_source='csv'.")

    def _build_terrain_backend(
        self,
        terrain_enabled,
        terrain_source,
        terrain_csv_path,
        terrain_base_height,
        terrain_num_hemispheres,
        terrain_radius_range,
        terrain_height_range,
        terrain_resolution,
        terrain_grid_resolution,
        terrain_origin,
        terrain_min_clearance,
        terrain_max_clearance,
        terrain_spawn_clearance,
        terrain_csv_target_x,
        terrain_csv_target_y,
        terrain_csv_target_z,
        terrain_csv_z_scale,
        terrain_csv_simplify_stride,
        terrain_csv_smooth_passes,
    ):
        # Wrap CSV/procedural terrain behind a common interface so collision,
        # spawning and height queries do not need to branch elsewhere.
        if terrain_source == "csv":
            return CSVTerrainMap(
                csv_path=terrain_csv_path,
                enabled=terrain_enabled,
                resolution=terrain_grid_resolution,
                origin=terrain_origin,
                min_clearance=terrain_min_clearance,
                max_clearance=terrain_max_clearance,
                edge_margin=max(terrain_spawn_clearance, 0.0),
                target_extent=(
                    terrain_csv_target_x,
                    terrain_csv_target_y,
                )
                if terrain_csv_target_x is not None and terrain_csv_target_y is not None
                else None,
                target_height=terrain_csv_target_z,
                z_scale=terrain_csv_z_scale,
                simplify_stride=terrain_csv_simplify_stride,
                smooth_passes=terrain_csv_smooth_passes,
            )

        return TerrainHeightMap(
            x_extent=self.X,
            y_extent=self.Y,
            enabled=terrain_enabled,
            base_height=terrain_base_height,
            num_hemispheres=terrain_num_hemispheres,
            hemisphere_radius_range=terrain_radius_range,
            hemisphere_height_range=terrain_height_range,
            edge_margin=max(terrain_spawn_clearance, 0.0),
            resolution=terrain_resolution,
        )

    def _sync_space_with_terrain(self):
        # CSV terrains can redefine the practical workspace extent. This keeps the
        # simulated airspace consistent with the loaded map after resampling/scaling.
        if (
            not self._use_terrain_obstacles()
            or self.terrain_source != "csv"
            or not self.terrain_match_space_dim
        ):
            return
        if getattr(self.terrain, "height_map", None) is None:
            return

        self.X = max(float(self.terrain.x_extent), 1e-6)
        self.Y = max(float(self.terrain.y_extent), 1e-6)
        terrain_peak = float(np.max(self.terrain.height_map))
        terrain_ceiling = terrain_peak + self.terrain.min_clearance + self.terrain_spawn_clearance
        self.Z = max(float(self.Z), terrain_ceiling + self.waypoint_step_size)
        self.workspace_diagonal = max(
            np.linalg.norm(np.array([self.X, self.Y, self.Z], dtype=np.float32)),
            1e-6,
        )

    def _default_lidar_step_size(self, terrain_enabled, terrain_grid_resolution):
        if terrain_enabled:
            return max(min(0.25, float(terrain_grid_resolution) * 0.5), 0.05)
        return 1.0

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

    def _is_near_hemisphere_obstacle(self, position):
        px = float(position[0])
        py = float(position[1])
        for obs in self.obstacles:
            if getattr(obs, "kind", None) != "hemisphere":
                continue
            min_dist = float(getattr(obs, "radius", 0.0)) + self.hemisphere_exclusion_buffer
            if (px - float(obs.x)) ** 2 + (py - float(obs.y)) ** 2 <= min_dist ** 2:
                return True
        return False

    def _is_mountain_basin_point(self, position):
        # Reject points that sit inside local depressions surrounded by high
        # terrain. This keeps spawn/target sampling away from "bowls" that are
        # technically free but hard to escape from.
        if (
            not self.avoid_mountain_basins
            or not isinstance(self.terrain, CSVTerrainMap)
            or self.terrain.height_map is None
        ):
            return False

        grid_idx = self.terrain.world_to_grid(position[0], position[1])
        if grid_idx is None:
            return False
        x_idx, y_idx = grid_idx
        r = self.basin_window_radius
        x0 = max(0, x_idx - r)
        x1 = min(self.terrain.grid_shape[0], x_idx + r + 1)
        y0 = max(0, y_idx - r)
        y1 = min(self.terrain.grid_shape[1], y_idx + r + 1)
        neighborhood = self.terrain.height_map[x0:x1, y0:y1]
        if neighborhood.size < 9:
            return False

        local_height = float(self.terrain.height_map[x_idx, y_idx])
        local_mean = float(np.mean(neighborhood))
        local_max = float(np.max(neighborhood))
        high_neighbor_ratio = float(
            np.mean(neighborhood >= self.basin_highland_threshold)
        )

        return (
            local_mean >= self.basin_highland_threshold
            and high_neighbor_ratio >= self.basin_high_neighbor_ratio
            and (local_max - local_height) >= self.basin_relief_threshold
        )

    def _violates_named_terrain_spawn_rule(self, position):
        if not isinstance(self.terrain, CSVTerrainMap) or not self.terrain.csv_path:
            return False

        terrain_name = os.path.splitext(os.path.basename(self.terrain.csv_path))[0].lower()
        if terrain_name != "jiangning_mountain_simplified_smoother":
            return False

        px = float(position[0])
        py = float(position[1])
        surface_height = self._terrain_surface_height(px, py)
        return py < 30.0 and surface_height < 20.0

    def _use_primitive_obstacles(self):
        return self.obstacle_rule in {"primitives", "hybrid"}

    def _use_terrain_obstacles(self):
        return self.obstacle_rule in {"terrain", "hybrid"} and self.terrain.enabled

    def _generate_surface_hemisphere_obstacles(self, num_hemispheres):
        max_attempts = max(100, num_hemispheres * 50)
        attempts = 0
        placed = 0
        while placed < num_hemispheres and attempts < max_attempts:
            attempts += 1
            radius = np.random.uniform(*self.terrain_radius_range)
            height = np.random.uniform(*self.terrain_height_range)
            x = np.random.uniform(radius, self.X - radius)
            y = np.random.uniform(radius, self.Y - radius)
            if self._is_obstacle_overlapping(x, y, radius):
                continue
            z = self._terrain_surface_height(x, y)
            self.obstacles.append(EllipsoidalHemisphereObstacle(x, y, radius, height, z=z))
            placed += 1

    def _generate_obstacles(self):
        # Terrain is generated first because terrain itself acts as a collision
        # object in terrain/hybrid modes; primitive obstacles are layered on top.
        if self._use_terrain_obstacles():
            self.terrain.generate()
            self._sync_space_with_terrain()
        else:
            self.terrain.clear()

        self.obstacles = []
        if self._use_terrain_obstacles():
            # Height-map modes use a single hemisphere rule set.
            hemisphere_count = self.num_obstacles if self.num_obstacles > 0 else self.terrain_num_hemispheres
            if hemisphere_count > 0:
                self._generate_surface_hemisphere_obstacles(hemisphere_count)
            return

        if self._use_primitive_obstacles():
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
                    self.obstacles.append(CylinderObstacle(x, y, radius, height))
                else:
                    radius = np.random.uniform(*self.hemisphere_radius_range)
                    height = np.random.uniform(*self.hemisphere_height_range)
                    x = np.random.uniform(radius, self.X - radius)
                    y = np.random.uniform(radius, self.Y - radius)
                    if self._is_obstacle_overlapping(x, y, radius):
                        continue
                    self.obstacles.append(EllipsoidalHemisphereObstacle(x, y, radius, height, z=0.0))

                plan_index += 1

    def _is_collision(self, px, py, pz):
        if px < 0 or px > self.X or py < 0 or py > self.Y or pz < 0 or pz > self.Z:
            return True
        if self._use_terrain_obstacles() and self.terrain.collides(px, py, pz):
            return True
        if self._use_primitive_obstacles():
            for obs in self.obstacles:
                if obs.contains(px, py, pz):
                    return True
        return False

    def _collision_reason(self, px, py, pz):
        if px < 0 or px > self.X or py < 0 or py > self.Y or pz < 0 or pz > self.Z:
            return "workspace_boundary"

        if self._use_terrain_obstacles():
            if isinstance(self.terrain, CSVTerrainMap):
                surface_height = self.terrain.surface_height_at(px, py)
                if pz <= surface_height + self.terrain.min_clearance:
                    return "terrain_surface"
            elif self.terrain.collides(px, py, pz):
                return "terrain_surface"

        if self._use_primitive_obstacles():
            for idx, obs in enumerate(self.obstacles):
                if obs.contains(px, py, pz):
                    return f"{obs.kind}_obstacle_{idx}"

        return "unknown_collision"

    def _terrain_surface_height(self, px, py):
        if not self._use_terrain_obstacles():
            return 0.0
        return self.terrain.surface_height_at(px, py)

    def _log_collision(self, agent, reason):
        if self.log_collisions:
            print(f"[Collision] {agent}: {reason}")

    def _find_safe_motion_endpoint(self, start_pos, target_pos, resolution=0.5):
        # March along the requested segment and stop at the last safe sample so
        # callers can treat collisions as clipped motion rather than tunneling.
        start_pos = np.asarray(start_pos, dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)
        motion = target_pos - start_pos
        distance = float(np.linalg.norm(motion))

        if distance < 1e-8:
            return start_pos.copy(), False, None

        num_samples = max(int(np.ceil(distance / max(resolution, 1e-6))), 1)
        last_safe = start_pos.copy()

        for alpha in np.linspace(1.0 / num_samples, 1.0, num_samples):
            candidate = start_pos + alpha * motion
            if self._is_collision(candidate[0], candidate[1], candidate[2]):
                return (
                    last_safe,
                    True,
                    self._collision_reason(candidate[0], candidate[1], candidate[2]),
                )
            last_safe = candidate.astype(np.float32)

        return target_pos.copy(), False, None

    def _get_random_free_pos(self):
        while True:
            x = np.random.uniform(0, self.X)
            y = np.random.uniform(0, self.Y)
            z_min, z_max = self.terrain.sample_height_bounds(
                x,
                y,
                clearance=self.terrain_spawn_clearance,
            )
            z_min = max(z_min, 0.0)
            z_upper = self.Z if z_max is None else min(self.Z, z_max)
            z_upper = min(z_upper, self.spawn_max_height)
            if z_min >= z_upper:
                continue
            z = np.random.uniform(z_min, z_upper)
            if not self._is_collision(x, y, z):
                return np.array([x, y, z], dtype=np.float32)

    def _sample_free_position(self, min_distance=0.0, existing_positions=None, max_attempts=500):
        existing_positions = existing_positions or []
        for _ in range(max_attempts):
            candidate = self._get_random_free_pos()
            if self._is_near_hemisphere_obstacle(candidate):
                continue
            if self._is_mountain_basin_point(candidate):
                continue
            if self._violates_named_terrain_spawn_rule(candidate):
                continue
            if all(np.linalg.norm(candidate - pos) >= min_distance for pos in existing_positions):
                return candidate
        raise RuntimeError("Unable to sample a free start position outside hemisphere buffers.")

    def _ground_target_height(self, x, y):
        if self._use_terrain_obstacles():
            surface_height = self._terrain_surface_height(x, y)
            if isinstance(self.terrain, CSVTerrainMap):
                return float(surface_height + self.terrain.min_clearance + self.ground_target_clearance)
            return float(surface_height + self.ground_target_clearance)
        return 0.0

    def _get_random_ground_target_pos(self):
        while True:
            x = np.random.uniform(0, self.X)
            y = np.random.uniform(0, self.Y)
            z = self._ground_target_height(x, y)
            if z > self.Z:
                continue
            if not self._is_collision(x, y, z):
                return np.array([x, y, z], dtype=np.float32)

    def _sample_ground_target_position(
        self,
        min_distance=0.0,
        existing_positions=None,
        max_attempts=500,
    ):
        existing_positions = existing_positions or []
        for _ in range(max_attempts):
            candidate = self._get_random_ground_target_pos()
            if self._is_near_hemisphere_obstacle(candidate):
                continue
            if self._is_mountain_basin_point(candidate):
                continue
            if self._violates_named_terrain_spawn_rule(candidate):
                continue
            if all(np.linalg.norm(candidate - pos) >= min_distance for pos in existing_positions):
                return candidate
        raise RuntimeError("Unable to sample a target position outside hemisphere buffers.")

    def _build_agent_info(self, agent):
        # This helper collects step-level diagnostics from persistent state so the
        # same values can be reused by reset/eval/training callbacks.
        remaining_distance = None
        initial_distance = None
        distance_reduction = None
        distance_reduction_ratio = None
        remaining_distance_normalized = None
        if agent in self.agent_positions and agent in self.agent_targets:
            remaining_distance = float(
                np.linalg.norm(self.agent_targets[agent] - self.agent_positions[agent])
            )
            initial_distance = float(self.agent_initial_distances.get(agent, remaining_distance))
            distance_reduction = float(initial_distance - remaining_distance)
            if initial_distance > 1e-8:
                distance_reduction_ratio = float(distance_reduction / initial_distance)
            remaining_distance_normalized = float(remaining_distance / self.workspace_diagonal)
        return {
            "reached_goal": self.agent_dones.get(agent, False),
            "timed_out": self.agent_timeouts.get(agent, False),
            "collision_with_obstacle": self.agent_obstacle_collisions.get(agent, 0),
            "collision_with_agent": self.agent_inter_agent_collisions.get(agent, 0),
            "last_collision_reason": self.agent_collision_reasons.get(agent),
            "steps": self.step_cnt,
            "remaining_distance": remaining_distance,
            "remaining_distance_normalized": remaining_distance_normalized,
            "initial_distance": initial_distance,
            "distance_reduction": distance_reduction,
            "distance_reduction_ratio": distance_reduction_ratio,
            "all_agents_reached": all(self.agent_dones.values()) if self.agent_dones else False,
        }

    def _build_episode_metrics(self, agent):
        # Terminal metrics are separated from transient step info because vector
        # wrappers may auto-reset environments immediately after done=True.
        info = self._build_agent_info(agent)
        collision_with_obstacle = float(info.get("collision_with_obstacle", 0))
        collision_with_agent = float(info.get("collision_with_agent", 0))
        reached_goal = bool(info.get("reached_goal", False))
        collision_free_success = (
            reached_goal
            and collision_with_obstacle <= 0.0
            and collision_with_agent <= 0.0
        )
        return {
            "reached_goal": reached_goal,
            "timed_out": bool(info.get("timed_out", False)),
            "collision_with_obstacle": collision_with_obstacle,
            "collision_with_agent": collision_with_agent,
            "collision_free_success": bool(collision_free_success),
            "last_collision_reason": info.get("last_collision_reason"),
            "steps": int(info.get("steps", 0)),
            "remaining_distance": info.get("remaining_distance"),
            "remaining_distance_normalized": info.get("remaining_distance_normalized"),
            "initial_distance": info.get("initial_distance"),
            "distance_reduction": info.get("distance_reduction"),
            "distance_reduction_ratio": info.get("distance_reduction_ratio"),
            "all_agents_reached": bool(info.get("all_agents_reached", False)),
        }

    def _has_reached_target(self, agent):
        target_delta = self.agent_targets[agent] - self.agent_positions[agent]
        distance_3d = float(np.linalg.norm(target_delta))
        distance_xy = float(np.linalg.norm(target_delta[:2]))
        distance_z = float(abs(target_delta[2]))

        # relaxed_xy_threshold = max(self.goal_threshold * 1.8, self.waypoint_step_size * 1.5)
        # relaxed_z_threshold = max(self.goal_threshold * 1.2, self.waypoint_step_size)

        relaxed_xy_threshold = max(self.goal_threshold, self.waypoint_step_size)
        relaxed_z_threshold = max(self.goal_threshold, self.waypoint_step_size)

        return (
            distance_3d < self.goal_threshold
            or (distance_xy < relaxed_xy_threshold and distance_z < relaxed_z_threshold)
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # A full reset rebuilds terrain/obstacles first, then resamples starts
        # and targets, then clears every per-agent statistic used downstream.
        self.agents = self.possible_agents.copy()
        self.step_cnt = 0
        self._generate_obstacles()

        self.agent_positions = {}
        self.agent_last_motion = {}
        self.agent_targets = {}
        self.agent_initial_distances = {}
        self.agent_reached_targets = {}
        self.agent_dones = {}
        self.agent_timeouts = {}
        self.agent_obstacle_collisions = {}
        self.agent_inter_agent_collisions = {}
        self.agent_collision_reasons = {}

        existing_starts = []
        existing_targets = []
        for agent in self.agents:
            pos = self._sample_free_position(
                min_distance=self.spawn_min_separation,
                existing_positions=existing_starts,
            )
            target = self._sample_ground_target_position(
                min_distance=self.spawn_min_separation,
                existing_positions=existing_targets,
            )
            while self._is_collision(pos[0], pos[1], pos[2]):
                pos = self._sample_free_position(
                    min_distance=self.spawn_min_separation,
                    existing_positions=existing_starts,
                )
            while self._is_collision(target[0], target[1], target[2]):
                target = self._sample_ground_target_position(
                    min_distance=self.spawn_min_separation,
                    existing_positions=existing_targets,
                )
            while np.linalg.norm(pos - target) < self.start_target_min_distance:
                target = self._sample_ground_target_position(
                    min_distance=self.spawn_min_separation,
                    existing_positions=existing_targets,
                )
                while self._is_collision(target[0], target[1], target[2]):
                    target = self._sample_ground_target_position(
                        min_distance=self.spawn_min_separation,
                        existing_positions=existing_targets,
                    )

            self.agent_positions[agent] = pos
            self.agent_last_motion[agent] = np.zeros(3, dtype=np.float32)
            self.agent_targets[agent] = target
            self.agent_initial_distances[agent] = float(np.linalg.norm(pos - target))
            self.agent_reached_targets[agent] = False
            self.agent_dones[agent] = False
            self.agent_timeouts[agent] = False
            self.agent_obstacle_collisions[agent] = 0
            self.agent_inter_agent_collisions[agent] = 0
            self.agent_collision_reasons[agent] = None
            existing_starts.append(pos)
            existing_targets.append(target)

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: self._build_agent_info(agent) for agent in self.agents}
        return observations, infos

    def _cast_rays(self, agent_pos):
        # Ray returns are normalized into [0, 1]. A value near 1.0 means no hit
        # inside lidar_range, while smaller values mean earlier contact.
        distances = np.full(self.lidar_rays, self.lidar_range, dtype=np.float32)
        num_steps = max(int(self.lidar_num_samples), 1)
        step_size = self.lidar_range / num_steps if self.lidar_range > 0.0 else 0.0

        for i, direction in enumerate(self.ray_dirs):
            for step in range(1, num_steps + 1):
                travel = min(step * step_size, self.lidar_range)
                test_pos = agent_pos + direction * travel
                if self._is_collision(test_pos[0], test_pos[1], test_pos[2]):
                    distances[i] = travel
                    break

        return distances / self.lidar_range

    def observe(self, agent):
        # Observation layout: local LiDAR, ego pose/motion, relative target and
        # coarse teammate positions. The policy therefore sees both local geometry
        # and enough global context to keep making progress.
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
        # Step order:
        # 1) clear transient collision reasons,
        # 2) apply clipped motion and obstacle checks,
        # 3) resolve inter-agent collisions,
        # 4) assign rewards / done flags / terminal metrics.
        # 存储各种信息
        for agent in self.agents:
            self.agent_collision_reasons[agent] = None

        rewards = {
            agent: 0.0 if self.agent_reached_targets.get(agent, False) else self.step_penalty
            for agent in self.agents
        }
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        collision_happened = {agent: False for agent in self.agents}

        # 获取相对目标位置的距离
        self.step_cnt += 1
        dist_before = {
            agent: np.linalg.norm(self.agent_targets[agent] - self.agent_positions[agent])
            for agent in self.agents
        }

        # 更新每个agent的状态
        for agent in self.agents:

            if self.agent_reached_targets.get(agent, False):    #到达目标
                self.agent_last_motion[agent] = np.zeros(3, dtype=np.float32)
                continue
            
            #输出路点
            action = np.asarray(actions[agent], dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)     # 三轴幅度([-1, 1]) * 步长
            motion = action * self.waypoint_step_size
            target_waypoint = self.agent_positions[agent] + motion
            target_waypoint = np.clip(
                target_waypoint,
                [0.0, 0.0, 0.0],
                [self.X, self.Y, self.Z],
            )
            safe_pos, hit_obstacle, collision_reason = self._find_safe_motion_endpoint(
                self.agent_positions[agent],
                target_waypoint,
                resolution=0.5,
            )
            actual_motion = safe_pos - self.agent_positions[agent]
            new_pos = self.agent_positions[agent] + actual_motion

            if hit_obstacle:
                rewards[agent] -= self.obstacle_collision_penalty
                self.agent_obstacle_collisions[agent] += 1
                collision_happened[agent] = True
                self.agent_collision_reasons[agent] = collision_reason
                self._log_collision(agent, collision_reason)

            self.agent_positions[agent] = new_pos.astype(np.float32)
            self.agent_last_motion[agent] = actual_motion.astype(np.float32)

        agent_list = list(self.agents)
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a1 = agent_list[i]
                a2 = agent_list[j]
                if self.near_goal_collision_free_radius > 0.0:
                    dist_to_goal_a1 = np.linalg.norm(self.agent_targets[a1] - self.agent_positions[a1])
                    dist_to_goal_a2 = np.linalg.norm(self.agent_targets[a2] - self.agent_positions[a2])
                    if (
                        dist_to_goal_a1 <= self.near_goal_collision_free_radius
                        and dist_to_goal_a2 <= self.near_goal_collision_free_radius
                    ):
                        continue
                dist = np.linalg.norm(self.agent_positions[a1] - self.agent_positions[a2])
                if dist < self.collision_radius:
                    rewards[a1] -= self.agent_collision_penalty
                    rewards[a2] -= self.agent_collision_penalty
                    self.agent_inter_agent_collisions[a1] += 1
                    self.agent_inter_agent_collisions[a2] += 1
                    collision_happened[a1] = True
                    collision_happened[a2] = True
                    self.agent_collision_reasons[a1] = f"agent_collision_with_{a2}"
                    self.agent_collision_reasons[a2] = f"agent_collision_with_{a1}"
                    self._log_collision(a1, f"agent_collision_with_{a2}")
                    self._log_collision(a2, f"agent_collision_with_{a1}")
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
                and self._has_reached_target(agent)
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
                    self.agent_timeouts[agent] = True
                truncations[agent] = True

        self.agents = [
            agent for agent in self.agents if not terminations[agent] and not truncations[agent]
        ]

        for agent in infos:
            infos[agent] = self._build_agent_info(agent)
            if terminations.get(agent, False) or truncations.get(agent, False):
                episode_metrics = self._build_episode_metrics(agent)
                infos[agent]["episode_metrics"] = episode_metrics
                infos[agent]["is_success"] = bool(
                    episode_metrics.get("collision_free_success", False)
                )
                infos[agent]["goal_success"] = bool(
                    episode_metrics.get("reached_goal", False)
                )

        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations, rewards, terminations, truncations, infos


# Backward-compatible alias so existing training scripts can switch files
# without needing to rename the imported class immediately.
UAVContinuous3DMAPFEnv = UAVWaypoint3DMAPFEnv
