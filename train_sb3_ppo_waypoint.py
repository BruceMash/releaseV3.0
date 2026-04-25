import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import supersuit as ss
import datetime
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from functools import lru_cache

plugin_dir = r"D:\RL_env\Lib\site-packages\PyQt6\Qt6\plugins\platforms"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir

from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import Circle
from matplotlib.patches import PathPatch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Affine2D
from PIL import Image

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")



from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor

# 导入本地定义的环境
from Envs.continuous_3d_waypoint_env import (
    UAVWaypoint3DMAPFEnv,
    CylinderObstacle,
    EllipsoidalHemisphereObstacle,
)
from Envs.terrain_heightmap import CSVTerrainMap


DEFAULT_TERRAIN_CSV = os.path.join("Envs", "jiangning_mountain_simplified_smoother.csv")
ICON_DIR = "icons"
UAV_ICON_PATH = os.path.join(ICON_DIR, "固定翼无人机.svg")
TASK_ICON_PATH = os.path.join(ICON_DIR, "旗子.svg")


def _parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _uses_raw_csv_heightmap_z(env):
    terrain = getattr(env, "terrain", None)
    return (
        isinstance(terrain, CSVTerrainMap)
        and getattr(terrain, "target_height", None) is None
        and abs(float(getattr(terrain, "z_scale", 1.0)) - 1.0) <= 1e-12
    )


def _smooth_history_path(path, smooth_window=5):
    path = np.asarray(path, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] <= 2:
        return path

    window = max(int(smooth_window), 1)
    if window <= 1:
        return path
    if window % 2 == 0:
        window += 1
    window = min(window, path.shape[0] if path.shape[0] % 2 == 1 else path.shape[0] - 1)
    if window <= 1:
        return path

    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.empty_like(path, dtype=np.float32)
    for axis in range(path.shape[1]):
        padded = np.pad(path[:, axis], (pad, pad), mode="edge")
        smoothed[:, axis] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _binary_segment_collision_check(env, start, end, coarse_step=None, binary_iters=12):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    delta = end - start
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-8:
        return False

    coarse_step = max(
        float(coarse_step if coarse_step is not None else getattr(env, "waypoint_step_size", 1.0) * 0.5),
        1e-3,
    )
    num_samples = max(int(np.ceil(distance / coarse_step)), 2)
    start_collided = bool(env._is_collision(float(start[0]), float(start[1]), float(start[2])))
    if start_collided:
        return True

    previous_ratio = 0.0
    for idx in range(1, num_samples + 1):
        ratio = float(idx) / float(num_samples)
        point = start + ratio * delta
        collided = bool(env._is_collision(float(point[0]), float(point[1]), float(point[2])))
        if collided:
            left_ratio = previous_ratio
            right_ratio = ratio
            for _ in range(max(int(binary_iters), 1)):
                mid_ratio = 0.5 * (left_ratio + right_ratio)
                mid_point = start + mid_ratio * delta
                mid_collided = bool(
                    env._is_collision(float(mid_point[0]), float(mid_point[1]), float(mid_point[2]))
                )
                if mid_collided:
                    right_ratio = mid_ratio
                else:
                    left_ratio = mid_ratio
            return True
        previous_ratio = ratio
    return False


def _is_render_segment_safe(env, start, end):
    return not _binary_segment_collision_check(
        env,
        start,
        end,
        coarse_step=max(float(getattr(env, "waypoint_step_size", 1.0)) * 0.5, 0.25),
        binary_iters=12,
    )


def _build_render_polyline(env, points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return points
    if points.shape[0] == 1:
        return points

    polyline = [points[0].copy()]
    for idx in range(1, points.shape[0]):
        current = points[idx].copy()
        previous = points[idx - 1].copy()
        if not _is_render_segment_safe(env, previous, current):
            polyline.append(np.full(points.shape[1], np.nan, dtype=np.float32))
        polyline.append(current)
    return np.asarray(polyline, dtype=np.float32)


def _extract_explicit_cli_dests(argv, parser):
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit_dests = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit_dests.add(dest)
    return explicit_dests


def _merge_args_from_config_json(args, parser, argv):
    config_path = getattr(args, "config_json", "")
    if not config_path:
        return args

    with open(config_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"config_json must contain a JSON object, got {type(loaded).__name__}")
    setattr(args, "_loaded_config_json", loaded)

    loaded_args = loaded.get("args")
    if isinstance(loaded_args, dict):
        merged = dict(loaded_args)
        for key, value in loaded.items():
            if key == "args":
                continue
            if key not in merged:
                merged[key] = value
        loaded = merged

    explicit_dests = _extract_explicit_cli_dests(argv, parser)
    explicit_dests.add("config_json")

    for key, value in loaded.items():
        if not hasattr(args, key):
            continue
        if key in explicit_dests:
            continue
        setattr(args, key, value)

    return args


class StepRewardCallback(BaseCallback):
    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        actions = self.locals.get("actions", None)
        infos = self.locals.get("infos", None)
        if rewards is not None:
            rewards = np.asarray(rewards, dtype=np.float32)
            self.logger.record("step/reward_mean", float(np.mean(rewards)))
            self.logger.record("step/reward_min", float(np.min(rewards)))
            self.logger.record("step/reward_max", float(np.max(rewards)))
            self.logger.record("step/reward_std", float(np.std(rewards)))
        if dones is not None:
            dones = np.asarray(dones, dtype=np.float32)
            self.logger.record("step/done_rate", float(np.mean(dones)))
        if actions is not None:
            actions = np.asarray(actions, dtype=np.float32)
            self.logger.record("step/action_mean_abs", float(np.mean(np.abs(actions))))
            self.logger.record("step/action_std", float(np.std(actions)))
            if actions.ndim >= 2 and actions.shape[-1] >= 3:
                self.logger.record("step/action_x_mean_abs", float(np.mean(np.abs(actions[..., 0]))))
                self.logger.record("step/action_y_mean_abs", float(np.mean(np.abs(actions[..., 1]))))
                self.logger.record("step/action_z_mean_abs", float(np.mean(np.abs(actions[..., 2]))))
        if infos:
            obstacle_hits = [
                float(
                    isinstance(info.get("last_collision_reason"), str)
                    and "agent_collision_with_" not in info.get("last_collision_reason", "")
                )
                for info in infos
                if info is not None
            ]
            agent_hits = [
                float(
                    isinstance(info.get("last_collision_reason"), str)
                    and "agent_collision_with_" in info.get("last_collision_reason", "")
                )
                for info in infos
                if info is not None
            ]
            if obstacle_hits:
                self.logger.record("step/obstacle_collision_rate", float(np.mean(obstacle_hits)))
            if agent_hits:
                self.logger.record("step/agent_collision_rate", float(np.mean(agent_hits)))
        return True


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_returns = None
        self.episode_lengths = None
        self.num_envs = 0

    def _on_training_start(self) -> None:
        self.num_envs = self.training_env.num_envs
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if rewards is None or dones is None:
            return True

        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        infos = infos or [{} for _ in range(len(rewards))]

        self.episode_returns[: len(rewards)] += rewards
        self.episode_lengths[: len(rewards)] += 1

        finished_returns = []
        finished_lengths = []
        finished_goal_success = []
        finished_collision_free_success = []
        finished_timeouts = []
        finished_obstacle_collisions = []
        finished_agent_collisions = []
        finished_initial_distance = []
        finished_remaining_distance = []
        finished_remaining_distance_normalized = []
        finished_distance_reduction = []
        finished_distance_reduction_ratio = []

        for idx, done in enumerate(dones):
            if not done:
                continue

            info = infos[idx] if idx < len(infos) else {}
            episode_info = info.get("episode", {})
            terminal_metrics = info.get("episode_metrics", {})

            finished_returns.append(
                float(episode_info.get("r", self.episode_returns[idx]))
            )
            finished_lengths.append(
                float(episode_info.get("l", self.episode_lengths[idx]))
            )
            goal_success = float(bool(terminal_metrics.get("reached_goal", False)))
            collision_free_success = float(
                bool(terminal_metrics.get("collision_free_success", False))
            )
            finished_goal_success.append(goal_success)
            finished_collision_free_success.append(collision_free_success)
            finished_timeouts.append(float(bool(terminal_metrics.get("timed_out", False))))
            finished_obstacle_collisions.append(float(terminal_metrics.get("collision_with_obstacle", 0)))
            finished_agent_collisions.append(float(terminal_metrics.get("collision_with_agent", 0)))

            initial_distance = terminal_metrics.get("initial_distance", None)
            if initial_distance is not None:
                finished_initial_distance.append(float(initial_distance))
            remaining_distance = terminal_metrics.get("remaining_distance", None)
            if remaining_distance is not None:
                finished_remaining_distance.append(float(remaining_distance))
            remaining_distance_normalized = terminal_metrics.get("remaining_distance_normalized", None)
            if remaining_distance_normalized is not None:
                finished_remaining_distance_normalized.append(float(remaining_distance_normalized))
            distance_reduction = terminal_metrics.get("distance_reduction", None)
            if distance_reduction is not None:
                finished_distance_reduction.append(float(distance_reduction))
            distance_reduction_ratio = terminal_metrics.get("distance_reduction_ratio", None)
            if distance_reduction_ratio is not None:
                finished_distance_reduction_ratio.append(float(distance_reduction_ratio))

            self.episode_returns[idx] = 0.0
            self.episode_lengths[idx] = 0

        if finished_returns:
            self.logger.record("episode/return", float(np.mean(finished_returns)))
            self.logger.record("episode/return_std", float(np.std(finished_returns)))
            self.logger.record("episode/length", float(np.mean(finished_lengths)))
            self.logger.record(
                "episode/success_rate",
                float(np.mean(finished_collision_free_success)),
            )
            self.logger.record(
                "episode/collision_free_success_rate",
                float(np.mean(finished_collision_free_success)),
            )
            self.logger.record(
                "episode/goal_success_rate",
                float(np.mean(finished_goal_success)),
            )
            self.logger.record("episode/timeout_rate", float(np.mean(finished_timeouts)))
            self.logger.record(
                "episode/obstacle_collisions",
                float(np.mean(finished_obstacle_collisions)),
            )
            self.logger.record(
                "episode/agent_collisions",
                float(np.mean(finished_agent_collisions)),
            )
            self.logger.record(
                "episode/obstacle_collisions_per_step",
                float(
                    np.mean(
                        [
                            obs / max(length, 1.0)
                            for obs, length in zip(finished_obstacle_collisions, finished_lengths)
                        ]
                    )
                ),
            )
            self.logger.record(
                "episode/agent_collisions_per_step",
                float(
                    np.mean(
                        [
                            coll / max(length, 1.0)
                            for coll, length in zip(finished_agent_collisions, finished_lengths)
                        ]
                    )
                ),
            )
            if finished_initial_distance:
                self.logger.record(
                    "episode/initial_distance",
                    float(np.mean(finished_initial_distance)),
                )
            if finished_remaining_distance:
                self.logger.record(
                    "episode/final_remaining_distance",
                    float(np.mean(finished_remaining_distance)),
                )
                self.logger.record(
                    "episode/final_remaining_distance_p90",
                    float(np.percentile(finished_remaining_distance, 90)),
                )
            if finished_remaining_distance_normalized:
                self.logger.record(
                    "episode/final_remaining_distance_normalized",
                    float(np.mean(finished_remaining_distance_normalized)),
                )
            if finished_distance_reduction:
                self.logger.record(
                    "episode/distance_reduction",
                    float(np.mean(finished_distance_reduction)),
                )
            if finished_distance_reduction_ratio:
                self.logger.record(
                    "episode/distance_reduction_ratio",
                    float(np.mean(finished_distance_reduction_ratio)),
                )
                self.logger.record(
                    "episode/distance_reduction_ratio_p90",
                    float(np.percentile(finished_distance_reduction_ratio, 90)),
                )

        return True


def _build_waypoint_env_kwargs(args_or_none=None, n_agents=5, max_steps=200):
    args = args_or_none
    terrain_csv_path = (
        getattr(args, "terrain_csv_path", DEFAULT_TERRAIN_CSV)
        if args is not None
        else DEFAULT_TERRAIN_CSV
    )
    return {
        "space_dim": (
            getattr(args, "space_x", 100.0),
            getattr(args, "space_y", 100.0),
            getattr(args, "space_z", 15.0),
        )
        if args is not None
        else (100.0, 100.0, 15.0),
        "n_agents": n_agents,
        "max_steps": max_steps,
        "render_mode": None,
        "obstacle_rule": getattr(args, "obstacle_rule", "terrain") if args is not None else "terrain",
        "num_obstacles": getattr(args, "num_obstacles", 0) if args is not None else 0,
        "hemisphere_fraction": getattr(args, "hemisphere_fraction", 0.5) if args is not None else 0.5,
        "obs_radius_range": (
            getattr(args, "obs_radius_min", 2.0),
            getattr(args, "obs_radius_max", 8.0),
        )
        if args is not None
        else (2.0, 8.0),
        "obs_height_range": (
            getattr(args, "obs_height_min", 10.0),
            getattr(args, "obs_height_max", 40.0),
        )
        if args is not None
        else (10.0, 40.0),
        "hemisphere_radius_range": (
            getattr(args, "hemisphere_radius_min", 4.0),
            getattr(args, "hemisphere_radius_max", 10.0),
        )
        if args is not None
        else (4.0, 10.0),
        "hemisphere_height_range": (
            getattr(args, "hemisphere_height_min", 10.0),
            getattr(args, "hemisphere_height_max", 10.0),
        )
        if args is not None
        else (10.0, 10.0),
        "terrain_enabled": getattr(args, "terrain_enabled", True) if args is not None else True,
        "terrain_source": getattr(args, "terrain_source", "csv") if args is not None else "csv",
        "terrain_csv_path": terrain_csv_path,
        "terrain_origin": (
            getattr(args, "terrain_origin_x", 0.0),
            getattr(args, "terrain_origin_y", 0.0),
        )
        if args is not None
        else (0.0, 0.0),
        "terrain_grid_resolution": getattr(args, "terrain_grid_resolution", 1.0)
        if args is not None
        else 1.0,
        "terrain_csv_target_x": getattr(args, "terrain_csv_target_x", None)
        if args is not None
        else None,
        "terrain_csv_target_y": getattr(args, "terrain_csv_target_y", None)
        if args is not None
        else None,
        "terrain_csv_target_z": getattr(args, "terrain_csv_target_z", None)
        if args is not None
        else None,
        "terrain_csv_z_scale": getattr(args, "terrain_csv_z_scale", 0.001)
        if args is not None
        else 0.001,
        "terrain_min_clearance": getattr(args, "terrain_min_clearance", 10.0)
        if args is not None
        else 10.0,
        "terrain_max_clearance": getattr(args, "terrain_max_clearance", 100.0)
        if args is not None
        else 100.0,
        "terrain_match_space_dim": getattr(args, "terrain_match_space_dim", True)
        if args is not None
        else True,
        "terrain_spawn_clearance": getattr(args, "terrain_spawn_clearance", 1.0)
        if args is not None
        else 1.0,
        "spawn_max_height": getattr(args, "spawn_max_height", 30.0)
        if args is not None
        else 30.0,
        "ground_target_clearance": getattr(args, "ground_target_clearance", 0.5)
        if args is not None
        else 0.5,
        "terrain_num_hemispheres": getattr(args, "terrain_num_hemispheres", 0)
        if args is not None
        else 0,
        "terrain_radius_range": (
            getattr(args, "terrain_radius_min", 250.0),
            getattr(args, "terrain_radius_max", 400.0),
        )
        if args is not None
        else (25.0, 40.0),
        "terrain_height_range": (
            getattr(args, "terrain_height_min", 10.0),
            getattr(args, "terrain_height_max", 10.0),
        )
        if args is not None
        else (10.0, 10.0),
        "uav_speed": getattr(args, "uav_speed", 0.06) if args is not None else 0.06,
        "decision_period": getattr(args, "decision_period", 1.0) if args is not None else 1.0,
        "lidar_rays": getattr(args, "lidar_rays", 24) if args is not None else 24,
        "lidar_range": getattr(args, "lidar_range", 0.2) if args is not None else 0.2,
        "lidar_step_size": getattr(args, "lidar_step_size", None) if args is not None else None,
        "goal_threshold": getattr(args, "goal_threshold", 1.0) if args is not None else 1.0,
        "step_penalty": getattr(args, "step_penalty", -0.001) if args is not None else -0.001,
        "progress_reward_scale": getattr(args, "progress_reward_scale", 1.0)
        if args is not None
        else 1.0,
        "obstacle_collision_penalty": getattr(args, "obstacle_collision_penalty", 3.0)
        if args is not None
        else 3.0,
        "agent_collision_penalty": getattr(args, "agent_collision_penalty", 3.0)
        if args is not None
        else 3.0,
        "goal_reward": getattr(args, "goal_reward", 30.0) if args is not None else 30.0,
        "timeout_penalty_scale": getattr(args, "timeout_penalty_scale", 2.0)
        if args is not None
        else 2.0,
        "collision_radius": getattr(args, "collision_radius", 2.0) if args is not None else 2.0,
        "near_goal_collision_free_radius": getattr(args, "near_goal_collision_free_radius", 1.0)
        if args is not None
        else 1.0,
        "hemisphere_exclusion_buffer": getattr(args, "hemisphere_exclusion_buffer", 5.0)
        if args is not None
        else 5.0,
        "avoid_mountain_basins": getattr(args, "avoid_mountain_basins", True)
        if args is not None
        else True,
        "basin_window_radius": getattr(args, "basin_window_radius", 4)
        if args is not None
        else 4,
        "basin_highland_threshold": getattr(args, "basin_highland_threshold", 20.0)
        if args is not None
        else 20.0,
        "basin_relief_threshold": getattr(args, "basin_relief_threshold", 3.0)
        if args is not None
        else 3.0,
        "basin_high_neighbor_ratio": getattr(args, "basin_high_neighbor_ratio", 0.35)
        if args is not None
        else 0.35,
    }


def make_env(n_agents=5, max_steps=200, args=None):
    env = UAVWaypoint3DMAPFEnv(
        **_build_waypoint_env_kwargs(args_or_none=args, n_agents=n_agents, max_steps=max_steps)
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    return env


def _compute_goal_directed_action(env, agent):
    target_vec = env.agent_targets[agent] - env.agent_positions[agent]
    scaled_action = target_vec / max(env.waypoint_step_size, 1e-8)
    return np.clip(scaled_action, -1.0, 1.0).astype(np.float32)


def _get_hidden_eval_policy_config():
    # Keep these eval-only helper behaviors internal so the test/train
    # interfaces expose only core scenario and model parameters.
    return {
        "smoothing": 0.65,
        "goal_blend": 0.2,
        "emergency_avoidance": True,
        "emergency_lidar_threshold": 0.5,
        "emergency_max_blend": 0.95,
        "emergency_repulsion_power": 2.0,
        "emergency_speed_scale_min": 0.3,
        "emergency_speed_scale_max": 1.0,
        "min_safe_agl": 0.2,
    }


def _smooth_eval_action(env, agent, raw_action, previous_action, smoothing_factor, target_blend):
    raw_action = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
    previous_action = np.asarray(previous_action, dtype=np.float32)

    smoothed_action = (
        smoothing_factor * previous_action
        + (1.0 - smoothing_factor) * raw_action
    )

    if target_blend > 0.0:
        goal_action = _compute_goal_directed_action(env, agent)
        smoothed_action = (
            (1.0 - target_blend) * smoothed_action
            + target_blend * goal_action
        )

    return np.clip(smoothed_action, -1.0, 1.0).astype(np.float32)


def _format_lidar_observation(env, agent):
    if agent not in env.agent_positions:
        return "[]"
    lidar_values = env._cast_rays(env.agent_positions[agent]).astype(np.float32)
    return np.array2string(
        lidar_values,
        precision=3,
        separator=", ",
        suppress_small=False,
        max_line_width=200,
    )


def _compute_emergency_avoid_action(env, agent, base_action, trigger_threshold, repulsion_power):
    if agent not in env.agent_positions:
        return np.clip(np.asarray(base_action, dtype=np.float32), -1.0, 1.0).astype(np.float32), 1.0

    lidar_values = env._cast_rays(env.agent_positions[agent]).astype(np.float32)
    min_lidar = float(np.min(lidar_values)) if lidar_values.size > 0 else 1.0
    if min_lidar >= trigger_threshold:
        return np.clip(np.asarray(base_action, dtype=np.float32), -1.0, 1.0).astype(np.float32), min_lidar

    base_action = np.clip(np.asarray(base_action, dtype=np.float32), -1.0, 1.0)
    closeness = np.clip(trigger_threshold - lidar_values, 0.0, trigger_threshold)
    if trigger_threshold > 1e-8:
        closeness = closeness / trigger_threshold
    weights = np.power(closeness, max(float(repulsion_power), 1.0)).astype(np.float32)

    repulsion = -np.sum(env.ray_dirs * weights[:, None], axis=0, dtype=np.float32)
    repulsion_norm = float(np.linalg.norm(repulsion))
    if repulsion_norm <= 1e-8:
        return base_action.astype(np.float32), min_lidar

    avoid_action = (repulsion / repulsion_norm).astype(np.float32)
    goal_action = _compute_goal_directed_action(env, agent)
    if float(np.linalg.norm(goal_action)) > 1e-8:
        tangent_goal = goal_action - np.dot(goal_action, avoid_action) * avoid_action
        tangent_norm = float(np.linalg.norm(tangent_goal))
        if tangent_norm > 1e-8:
            avoid_action = 0.8 * avoid_action + 0.2 * (tangent_goal / tangent_norm).astype(np.float32)

    avoid_norm = float(np.linalg.norm(avoid_action))
    if avoid_norm > 1e-8:
        avoid_action = (avoid_action / avoid_norm).astype(np.float32)

    return np.clip(avoid_action, -1.0, 1.0).astype(np.float32), min_lidar


def _apply_eval_emergency_avoidance(env, agent, action, args):
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    eval_cfg = _get_hidden_eval_policy_config()
    if not bool(eval_cfg["emergency_avoidance"]):
        return action.astype(np.float32)

    threshold = float(np.clip(eval_cfg["emergency_lidar_threshold"], 0.0, 1.0))
    if threshold <= 0.0:
        return action.astype(np.float32)

    avoid_action, min_lidar = _compute_emergency_avoid_action(
        env,
        agent,
        action,
        trigger_threshold=threshold,
        repulsion_power=max(float(eval_cfg["emergency_repulsion_power"]), 1.0),
    )
    if min_lidar >= threshold:
        return action.astype(np.float32)

    severity = (threshold - min_lidar) / max(threshold, 1e-8)
    action_dot_avoid = float(np.dot(action, avoid_action))
    if action_dot_avoid < 0.0:
        tangential_action = action - action_dot_avoid * avoid_action
    else:
        tangential_action = action.copy()

    base_speed = max(float(np.linalg.norm(action)), 0.2)
    push_strength = float(np.clip(eval_cfg["emergency_max_blend"], 0.0, 1.0)) * severity * base_speed
    mixed_action = tangential_action + push_strength * avoid_action
    mixed_norm = float(np.linalg.norm(mixed_action))
    if mixed_norm > 1.0:
        mixed_action = mixed_action / mixed_norm
    mixed_action = np.clip(mixed_action, -1.0, 1.0).astype(np.float32)

    max_scale = float(np.clip(eval_cfg["emergency_speed_scale_max"], 0.0, 1.0))
    min_scale = float(np.clip(eval_cfg["emergency_speed_scale_min"], 0.0, max_scale))
    scale = max_scale - severity * (max_scale - min_scale)

    action_norm = float(np.linalg.norm(mixed_action))
    if action_norm <= 1e-8:
        return mixed_action

    scaled_action = mixed_action * scale
    scaled_norm = float(np.linalg.norm(scaled_action))
    if scaled_norm > 1.0:
        scaled_action = scaled_action / scaled_norm
    return scaled_action.astype(np.float32)


def _enforce_eval_min_agl(env, agent, action, min_agl):
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    if agent not in env.agent_positions or min_agl <= 0.0:
        return action.astype(np.float32)

    current_pos = np.asarray(env.agent_positions[agent], dtype=np.float32)
    predicted_pos = current_pos + action * float(env.waypoint_step_size)
    surface_height = float(env._terrain_surface_height(predicted_pos[0], predicted_pos[1]))
    min_safe_z = surface_height + float(min_agl)

    if predicted_pos[2] >= min_safe_z:
        return action.astype(np.float32)

    required_delta_z = min_safe_z - current_pos[2]
    required_action_z = required_delta_z / max(float(env.waypoint_step_size), 1e-8)
    action[2] = np.clip(required_action_z, -1.0, 1.0)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _sample_eval_free_position_in_region(
    env,
    x_bounds,
    y_bounds,
    min_distance=0.0,
    existing_positions=None,
    max_attempts=500,
):
    existing_positions = existing_positions or []
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    for _ in range(max_attempts):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z_min, z_max = env.terrain.sample_height_bounds(
            x,
            y,
            clearance=env.terrain_spawn_clearance,
        )
        z_min = max(z_min, 0.0)
        z_upper = env.Z if z_max is None else min(env.Z, z_max)
        if z_min >= z_upper:
            continue
        z = np.random.uniform(z_min, z_upper)
        candidate = np.array([x, y, z], dtype=np.float32)
        if env._is_collision(x, y, z):
            continue
        if all(np.linalg.norm(candidate - pos) >= min_distance for pos in existing_positions):
            return candidate
    return env._sample_free_position(
        min_distance=min_distance,
        existing_positions=existing_positions,
        max_attempts=max_attempts,
    )


def _sample_eval_ground_target_in_region(
    env,
    x_bounds,
    y_bounds,
    min_distance=0.0,
    existing_positions=None,
    max_attempts=500,
):
    existing_positions = existing_positions or []
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    for _ in range(max_attempts):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = env._ground_target_height(x, y)
        if z > env.Z or env._is_collision(x, y, z):
            continue
        candidate = np.array([x, y, z], dtype=np.float32)
        if all(np.linalg.norm(candidate - pos) >= min_distance for pos in existing_positions):
            return candidate
    return env._sample_ground_target_position(
        min_distance=min_distance,
        existing_positions=existing_positions,
        max_attempts=max_attempts,
    )


def _configure_eval_start_target_regions(env):
    if not isinstance(getattr(env, "terrain", None), CSVTerrainMap):
        return

    start_region = ((0.0, 0.28 * env.X), (0.0, 0.28 * env.Y))
    target_region = ((0.72 * env.X, env.X), (0.72 * env.Y, env.Y))

    existing_starts = []
    existing_targets = []
    for agent in env.possible_agents:
        pos = _sample_eval_free_position_in_region(
            env,
            start_region[0],
            start_region[1],
            min_distance=env.spawn_min_separation,
            existing_positions=existing_starts,
        )
        target = _sample_eval_ground_target_in_region(
            env,
            target_region[0],
            target_region[1],
            min_distance=env.spawn_min_separation,
            existing_positions=existing_targets,
        )
        while np.linalg.norm(pos - target) < env.start_target_min_distance:
            target = _sample_eval_ground_target_in_region(
                env,
                target_region[0],
                target_region[1],
                min_distance=env.spawn_min_separation,
                existing_positions=existing_targets,
            )

        env.agent_positions[agent] = pos
        env.agent_last_motion[agent] = np.zeros(3, dtype=np.float32)
        env.agent_targets[agent] = target
        env.agent_initial_distances[agent] = float(np.linalg.norm(pos - target))
        env.agent_reached_targets[agent] = False
        env.agent_dones[agent] = False
        env.agent_timeouts[agent] = False
        env.agent_obstacle_collisions[agent] = 0
        env.agent_inter_agent_collisions[agent] = 0
        env.agent_collision_reasons[agent] = None
        existing_starts.append(pos)
        existing_targets.append(target)

    env.agents = env.possible_agents.copy()
    env.step_cnt = 0


def _capture_figure_frame(fig):
    # QtAgg/QTAgg canvases create the renderer lazily, so force a draw
    # before reading the RGBA buffer.
    fig.canvas.draw()
    # Some Matplotlib backends expose a buffer whose pixel size does not
    # match get_width_height() exactly (for example under DPI scaling).
    # Convert the RGBA memoryview directly to an array to preserve the
    # renderer-reported shape.
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return Image.fromarray(frame[..., :3].copy())


def _save_eval_artifacts(frames, output_dir, run_name, gif_fps):
    os.makedirs(output_dir, exist_ok=True)
    final_image_path = os.path.join(output_dir, f"{run_name}_final.png")
    gif_path = os.path.join(output_dir, f"{run_name}_trajectory.gif")

    if frames:
        frames[-1].save(final_image_path)
        frame_duration_ms = max(int(1000 / max(gif_fps, 1)), 1)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )

    return gif_path, final_image_path


def _svg_local_name(tag):
    return str(tag).split("}", 1)[-1]


def _tokenize_svg_path(path_data):
    token_pattern = r"[AaCcHhLlMmZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
    return re.findall(token_pattern, str(path_data or ""))


def _is_svg_command(token):
    return len(token) == 1 and token.isalpha()


def _parse_svg_path_data(path_data):
    tokens = _tokenize_svg_path(path_data)
    if not tokens:
        return None

    vertices = []
    codes = []
    index = 0
    command = None
    current = np.zeros(2, dtype=np.float64)
    start_point = np.zeros(2, dtype=np.float64)

    def read_float():
        nonlocal index
        if index >= len(tokens) or _is_svg_command(tokens[index]):
            return None
        value = float(tokens[index])
        index += 1
        return value

    def append_point(point, code):
        vertices.append((float(point[0]), float(point[1])))
        codes.append(code)

    while index < len(tokens):
        if _is_svg_command(tokens[index]):
            command = tokens[index]
            index += 1
        if command is None:
            break

        relative = command.islower()
        upper_command = command.upper()

        if upper_command == "M":
            first_point = True
            while index < len(tokens) and not _is_svg_command(tokens[index]):
                x_value = read_float()
                y_value = read_float()
                if x_value is None or y_value is None:
                    break
                point = np.array([x_value, y_value], dtype=np.float64)
                if relative:
                    point += current
                current = point
                if first_point:
                    start_point = current.copy()
                    append_point(current, Path.MOVETO)
                    first_point = False
                else:
                    append_point(current, Path.LINETO)
            command = "l" if relative else "L"
        elif upper_command == "L":
            while index < len(tokens) and not _is_svg_command(tokens[index]):
                x_value = read_float()
                y_value = read_float()
                if x_value is None or y_value is None:
                    break
                point = np.array([x_value, y_value], dtype=np.float64)
                if relative:
                    point += current
                current = point
                append_point(current, Path.LINETO)
        elif upper_command == "H":
            while index < len(tokens) and not _is_svg_command(tokens[index]):
                x_value = read_float()
                if x_value is None:
                    break
                current = current + np.array([x_value, 0.0], dtype=np.float64) if relative else np.array([x_value, current[1]], dtype=np.float64)
                append_point(current, Path.LINETO)
        elif upper_command == "V":
            while index < len(tokens) and not _is_svg_command(tokens[index]):
                y_value = read_float()
                if y_value is None:
                    break
                current = current + np.array([0.0, y_value], dtype=np.float64) if relative else np.array([current[0], y_value], dtype=np.float64)
                append_point(current, Path.LINETO)
        elif upper_command == "C":
            while index < len(tokens) and not _is_svg_command(tokens[index]):
                values = [read_float() for _ in range(6)]
                if any(value is None for value in values):
                    break
                points = [
                    np.array(values[0:2], dtype=np.float64),
                    np.array(values[2:4], dtype=np.float64),
                    np.array(values[4:6], dtype=np.float64),
                ]
                if relative:
                    points = [point + current for point in points]
                vertices.extend((tuple(points[0]), tuple(points[1]), tuple(points[2])))
                codes.extend((Path.CURVE4, Path.CURVE4, Path.CURVE4))
                current = points[2]
        elif upper_command == "A":
            while index < len(tokens) and not _is_svg_command(tokens[index]):
                values = [read_float() for _ in range(7)]
                if any(value is None for value in values):
                    break
                point = np.array(values[5:7], dtype=np.float64)
                if relative:
                    point += current
                current = point
                append_point(current, Path.LINETO)
        elif upper_command == "Z":
            current = start_point.copy()
            append_point(current, Path.CLOSEPOLY)
            command = None
        else:
            while index < len(tokens) and not _is_svg_command(tokens[index]):
                index += 1

    if not vertices or not codes:
        return None
    return vertices, codes


@lru_cache(maxsize=16)
def _load_svg_icon_path(svg_path):
    if not svg_path or not os.path.exists(svg_path):
        return None
    try:
        root = ET.parse(svg_path).getroot()
    except Exception:
        return None

    all_vertices = []
    all_codes = []
    for element in root.iter():
        if _svg_local_name(element.tag) != "path":
            continue
        parsed = _parse_svg_path_data(element.attrib.get("d", ""))
        if parsed is None:
            continue
        vertices, codes = parsed
        all_vertices.extend(vertices)
        all_codes.extend(codes)

    if not all_vertices:
        return None

    points = np.asarray(all_vertices, dtype=np.float64)
    finite_mask = np.isfinite(points).all(axis=1)
    if not finite_mask.any():
        return None
    bounds_points = points[finite_mask]
    min_xy = np.min(bounds_points, axis=0)
    max_xy = np.max(bounds_points, axis=0)
    center_xy = (min_xy + max_xy) * 0.5
    scale = float(max(np.max(max_xy - min_xy), 1e-6))
    normalized = (points - center_xy) / scale
    return Path(normalized, all_codes)


def _draw_svg_icon(
    ax,
    svg_path,
    x_value,
    y_value,
    size,
    color,
    angle=0.0,
    alpha=1.0,
    zorder=20,
    edgecolor="white",
):
    icon_path = _load_svg_icon_path(svg_path)
    if icon_path is None:
        return False
    transform = (
        Affine2D()
        .scale(float(size), -float(size))
        .rotate(float(angle))
        .translate(float(x_value), float(y_value))
        + ax.transData
    )
    patch = PathPatch(
        icon_path,
        transform=transform,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=max(float(size) * 0.08, 0.25),
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return True


def _clone_eval_history(history):
    return {
        agent: [np.asarray(point, dtype=np.float32).copy() for point in points]
        for agent, points in history.items()
    }


def _clone_eval_render_meta(render_meta):
    if not isinstance(render_meta, dict):
        return {}

    cloned = {}
    geo_origin = render_meta.get("geo_origin")
    if isinstance(geo_origin, dict):
        cloned["geo_origin"] = {
            "origin_lat": float(geo_origin.get("origin_lat", 0.0)),
            "origin_lon": float(geo_origin.get("origin_lon", 0.0)),
        }

    task_paths = render_meta.get("task_paths")
    if isinstance(task_paths, dict):
        cloned["task_paths"] = {
            str(agent): [np.asarray(point, dtype=np.float32).copy() for point in points]
            for agent, points in task_paths.items()
            if isinstance(points, (list, tuple))
        }

    visible_tasks = render_meta.get("visible_tasks")
    if isinstance(visible_tasks, (list, tuple)):
        cloned["visible_tasks"] = []
        for task in visible_tasks:
            if not isinstance(task, dict):
                continue
            cloned["visible_tasks"].append(
                {
                    "task_id": str(task.get("task_id", "")),
                    "label": str(task.get("label", task.get("task_id", ""))),
                    "position": np.asarray(task.get("position", []), dtype=np.float32).copy(),
                    "position_geo": dict(task.get("position_geo", {}))
                    if isinstance(task.get("position_geo"), dict)
                    else {},
                    "completed": bool(task.get("completed", False)),
                }
            )

    global_subgoals = render_meta.get("global_subgoals")
    if isinstance(global_subgoals, dict):
        cloned["global_subgoals"] = {
            str(agent): np.asarray(point, dtype=np.float32).copy()
            for agent, point in global_subgoals.items()
            if point is not None
        }

    global_routes = render_meta.get("global_routes")
    if isinstance(global_routes, dict):
        cloned["global_routes"] = {
            str(agent): [np.asarray(point, dtype=np.float32).copy() for point in points]
            for agent, points in global_routes.items()
            if isinstance(points, (list, tuple))
        }

    damaged_agents = render_meta.get("damaged_agents")
    if isinstance(damaged_agents, (list, tuple, set)):
        cloned["damaged_agents"] = [str(agent) for agent in damaged_agents]

    active_agents = render_meta.get("active_agents")
    if isinstance(active_agents, (list, tuple, set)):
        cloned["active_agents"] = [str(agent) for agent in active_agents]

    inactive_agents = render_meta.get("inactive_agents")
    if isinstance(inactive_agents, (list, tuple, set)):
        cloned["inactive_agents"] = [str(agent) for agent in inactive_agents]

    task_status_panel = render_meta.get("task_status_panel")
    if isinstance(task_status_panel, (list, tuple)):
        cloned["task_status_panel"] = [
            {
                "agent": str(row.get("agent", "")),
                "tasks": [str(item) for item in row.get("tasks", [])],
                "text": str(row.get("text", "")),
            }
            for row in task_status_panel
            if isinstance(row, dict)
        ]

    reassignment_panel = render_meta.get("reassignment_panel")
    if isinstance(reassignment_panel, dict):
        cloned["reassignment_panel"] = json.loads(json.dumps(reassignment_panel))

    discovered_hemisphere_indices = render_meta.get("discovered_hemisphere_indices")
    if isinstance(discovered_hemisphere_indices, (list, tuple, set)):
        cloned["discovered_hemisphere_indices"] = [int(idx) for idx in discovered_hemisphere_indices]

    return cloned


def _format_geo_label(geo_payload):
    if not isinstance(geo_payload, dict):
        return ""
    latitude = geo_payload.get("latitude")
    longitude = geo_payload.get("longitude")
    altitude = geo_payload.get("alt_km")
    if latitude is None or longitude is None:
        return ""
    if altitude is None:
        return f"{float(latitude):.3f}, {float(longitude):.3f}"
    return f"{float(latitude):.3f}, {float(longitude):.3f}, {float(altitude):.3f}km"


def _format_queue_for_panel(queue, max_items=3):
    queue = [str(item) for item in (queue or [])]
    shown = queue[:max_items]
    if len(queue) > max_items:
        shown.append("...")
    return " -> ".join(shown) if shown else "idle"


def _draw_task_status_panels(ax2d, render_meta):
    if not isinstance(render_meta, dict):
        return

    rows = render_meta.get("task_status_panel", [])
    if isinstance(rows, (list, tuple)) and rows:
        lines = ["UAV task queues"]
        lines.extend(str(row.get("text", "")) for row in rows[:10] if isinstance(row, dict))
        ax2d.text(
            1.04,
            0.98,
            "\n".join(lines),
            transform=ax2d.transAxes,
            fontsize=7,
            color="black",
            ha="left",
            va="top",
            clip_on=False,
            zorder=100,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.75"},
        )

    reassignment = render_meta.get("reassignment_panel")
    if isinstance(reassignment, dict):
        before = reassignment.get("before", {})
        after = reassignment.get("after", {})
        active_agents = [
            str(row.get("agent", ""))
            for row in rows
            if isinstance(row, dict) and row.get("agent", "")
        ]
        if not active_agents:
            active_agents = sorted(str(agent) for agent in set(before.keys()) | set(after.keys()))
        lines = [f"Reassignment @ step {int(reassignment.get('step', 0))}"]
        for agent in active_agents[:6]:
            before_text = _format_queue_for_panel(before.get(agent, []), max_items=3)
            after_text = _format_queue_for_panel(after.get(agent, []), max_items=3)
            lines.append(f"{agent}: {before_text} => {after_text}")
        ax2d.text(
            1.04,
            0.52,
            "\n".join(lines),
            transform=ax2d.transAxes,
            fontsize=6.5,
            color="darkslategray",
            ha="left",
            va="top",
            clip_on=False,
            zorder=100,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "aliceblue", "alpha": 0.82, "edgecolor": "0.7"},
        )


def _configure_real_world_axes(ax3d, ax2d, geo_origin):
    if not isinstance(geo_origin, dict):
        return
    origin_lat = float(geo_origin.get("origin_lat", 0.0))
    origin_lon = float(geo_origin.get("origin_lon", 0.0))
    lat_rad = np.radians(origin_lat)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = max(111320.0 * np.cos(lat_rad), 1e-6)

    def lon_formatter(x_value, _pos):
        longitude = origin_lon + (float(x_value) * 1000.0) / meters_per_deg_lon
        return f"{longitude:.3f}"

    def lat_formatter(y_value, _pos):
        latitude = origin_lat + (float(y_value) * 1000.0) / meters_per_deg_lat
        return f"{latitude:.3f}"

    def alt_formatter(z_value, _pos):
        return f"{float(z_value) * 1000.0:.0f}"

    ax3d.set_xlabel("Longitude (deg)")
    ax3d.set_ylabel("Latitude (deg)")
    ax3d.set_zlabel("Altitude (m)")
    ax3d.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax3d.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax3d.zaxis.set_major_formatter(FuncFormatter(alt_formatter))

    ax2d.set_xlabel("Longitude (deg)")
    ax2d.set_ylabel("Latitude (deg)")
    ax2d.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax2d.yaxis.set_major_formatter(FuncFormatter(lat_formatter))


def _make_eval_render_snapshot(env, step, history, render_meta=None):
    return {
        "step": int(step),
        "agents": list(env.agents),
        "positions": {
            agent: np.asarray(pos, dtype=np.float32).copy()
            for agent, pos in env.agent_positions.items()
        },
        "targets": {
            agent: np.asarray(target, dtype=np.float32).copy()
            for agent, target in env.agent_targets.items()
        },
        "last_motion": {
            agent: np.asarray(motion, dtype=np.float32).copy()
            for agent, motion in env.agent_last_motion.items()
        },
        "initial_distances": dict(env.agent_initial_distances),
        "reached": dict(env.agent_reached_targets),
        "dones": dict(env.agent_dones),
        "timeouts": dict(env.agent_timeouts),
        "obstacle_collisions": dict(env.agent_obstacle_collisions),
        "agent_collisions": dict(env.agent_inter_agent_collisions),
        "collision_reasons": dict(env.agent_collision_reasons),
        "history": _clone_eval_history(history),
        "render_meta": _clone_eval_render_meta(
            getattr(env, "_render_meta", None) if render_meta is None else render_meta
        ),
    }


def _apply_eval_render_snapshot(env, snapshot):
    env.agents = list(snapshot["agents"])
    env.agent_positions = {
        agent: np.asarray(pos, dtype=np.float32).copy()
        for agent, pos in snapshot["positions"].items()
    }
    env.agent_targets = {
        agent: np.asarray(target, dtype=np.float32).copy()
        for agent, target in snapshot["targets"].items()
    }
    env.agent_last_motion = {
        agent: np.asarray(motion, dtype=np.float32).copy()
        for agent, motion in snapshot["last_motion"].items()
    }
    env.agent_initial_distances = dict(snapshot["initial_distances"])
    env.agent_reached_targets = dict(snapshot["reached"])
    env.agent_dones = dict(snapshot["dones"])
    env.agent_timeouts = dict(snapshot["timeouts"])
    env.agent_obstacle_collisions = dict(snapshot["obstacle_collisions"])
    env.agent_inter_agent_collisions = dict(snapshot["agent_collisions"])
    env.agent_collision_reasons = dict(snapshot["collision_reasons"])
    env._render_meta = _clone_eval_render_meta(snapshot.get("render_meta", {}))
    return _clone_eval_history(snapshot["history"])


def _serialize_obstacle(obstacle):
    payload = {
        "kind": getattr(obstacle, "kind", obstacle.__class__.__name__),
        "type": obstacle.__class__.__name__,
    }
    for attr in ("x", "y", "z", "radius", "height"):
        if hasattr(obstacle, attr):
            payload[attr] = float(getattr(obstacle, attr))
    return payload


def _build_obstacle_from_setup(payload):
    kind = str(payload.get("kind", "")).lower()
    if kind == "cylinder":
        return CylinderObstacle(
            float(payload["x"]),
            float(payload["y"]),
            float(payload["radius"]),
            float(payload["height"]),
        )
    if kind == "hemisphere":
        return EllipsoidalHemisphereObstacle(
            float(payload["x"]),
            float(payload["y"]),
            float(payload["radius"]),
            float(payload["height"]),
            z=float(payload.get("z", 0.0)),
        )
    raise ValueError(f"Unsupported obstacle kind in setup: {payload.get('kind')}")


def _restore_eval_scene_from_setup(env, setup_payload):
    obstacle_payloads = setup_payload.get("obstacles")
    agent_payloads = setup_payload.get("agents")
    if not isinstance(obstacle_payloads, list) or not isinstance(agent_payloads, dict):
        return None, None

    env.obstacles = [_build_obstacle_from_setup(obstacle) for obstacle in obstacle_payloads]
    env.agent_positions = {}
    env.agent_targets = {}
    env.agent_last_motion = {}
    env.agent_initial_distances = {}
    env.agent_reached_targets = {}
    env.agent_dones = {}
    env.agent_timeouts = {}
    env.agent_obstacle_collisions = {}
    env.agent_inter_agent_collisions = {}
    env.agent_collision_reasons = {}

    for agent in env.possible_agents:
        payload = agent_payloads.get(agent)
        if payload is None:
            continue
        start = np.asarray(payload["start"], dtype=np.float32)
        target = np.asarray(payload["target"], dtype=np.float32)
        env.agent_positions[agent] = start
        env.agent_targets[agent] = target
        env.agent_last_motion[agent] = np.zeros(3, dtype=np.float32)
        env.agent_initial_distances[agent] = float(
            payload.get("initial_distance", np.linalg.norm(target - start))
        )
        env.agent_reached_targets[agent] = False
        env.agent_dones[agent] = False
        env.agent_timeouts[agent] = False
        env.agent_obstacle_collisions[agent] = 0
        env.agent_inter_agent_collisions[agent] = 0
        env.agent_collision_reasons[agent] = None

    env.agents = [agent for agent in env.possible_agents if agent in env.agent_positions]
    env.step_cnt = 0
    observations = {agent: env.observe(agent) for agent in env.agents}
    infos = {agent: env._build_agent_info(agent) for agent in env.agents}
    return observations, infos


def _save_eval_setup(raw_env, args, model_dir, run_name):
    eval_artifact_dir = os.path.join(model_dir, "eval_artifacts")
    os.makedirs(eval_artifact_dir, exist_ok=True)

    terrain = getattr(raw_env, "terrain", None)
    setup = {
        "run_name": run_name,
        "terrain": {
            "type": terrain.__class__.__name__ if terrain is not None else None,
            "csv_path": getattr(terrain, "csv_path", None),
            "target_extent": list(getattr(terrain, "target_extent", []) or []),
            "target_height": getattr(terrain, "target_height", None),
            "resolution": getattr(terrain, "resolution", None),
            "origin": (
                np.asarray(getattr(terrain, "origin", []), dtype=np.float32).tolist()
                if hasattr(terrain, "origin")
                else []
            ),
        },
        "space_dim": [float(raw_env.X), float(raw_env.Y), float(raw_env.Z)],
        "obstacle_rule": raw_env.obstacle_rule,
        "obstacles": [_serialize_obstacle(obstacle) for obstacle in raw_env.obstacles],
        "agents": {},
        "args": vars(args).copy(),
    }

    for agent in raw_env.possible_agents:
        setup["agents"][agent] = {
            "start": np.asarray(raw_env.agent_positions[agent], dtype=np.float32).tolist(),
            "target": np.asarray(raw_env.agent_targets[agent], dtype=np.float32).tolist(),
            "initial_distance": float(raw_env.agent_initial_distances.get(agent, 0.0)),
        }

    setup_path = os.path.join(eval_artifact_dir, f"{run_name}_setup.json")
    with open(setup_path, "w", encoding="utf-8") as f:
        json.dump(setup, f, indent=2, ensure_ascii=False)
    return setup_path


@lru_cache(maxsize=128)
def _cached_cylinder_mesh(radius, height, z_steps=50, theta_steps=50):
    z = np.linspace(0, height, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    return x_grid, y_grid, z_grid


def _draw_cylinder(ax, x, y, radius, height, color="gray", alpha=0.5):
    x_grid, y_grid, z_grid = _cached_cylinder_mesh(float(radius), float(height))
    ax.plot_surface(x_grid + x, y_grid + y, z_grid, color=color, alpha=alpha, shade=True)


@lru_cache(maxsize=128)
def _cached_ground_disc_mesh(radius, theta_steps=60, radial_steps=20):
    theta = np.linspace(0, 2 * np.pi, 60)
    radial = np.linspace(0, 1.0, 20)
    theta_grid, radial_grid = np.meshgrid(theta, radial)
    x_grid = radius * radial_grid * np.cos(theta_grid)
    y_grid = radius * radial_grid * np.sin(theta_grid)
    z_grid = np.zeros_like(x_grid)
    return x_grid, y_grid, z_grid


def _draw_ground_disc(ax, x, y, radius, color="gray", alpha=0.18):
    x_grid, y_grid, z_grid = _cached_ground_disc_mesh(float(radius))
    ax.plot_surface(x_grid + x, y_grid + y, z_grid, color=color, alpha=alpha, shade=False)


def _get_terrain_render_cache(env, terrain_stride=1, alpha_3d=0.35):
    terrain = getattr(env, "terrain", None)
    if terrain is None or terrain.height_map is None:
        return None

    cache = getattr(env, "_eval_terrain_render_cache", None)
    cache_key = (max(int(terrain_stride), 1), float(alpha_3d))
    if cache is not None and cache.get("key") == cache_key:
        return cache

    stride = max(int(terrain_stride), 1)
    grid_x = terrain.grid_x[::stride, ::stride]
    grid_y = terrain.grid_y[::stride, ::stride]
    height_map = terrain.height_map[::stride, ::stride]

    vmin = float(np.min(height_map))
    vmax = float(np.max(height_map))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    facecolors = cm.terrain(norm(height_map))
    facecolors[..., -1] = alpha_3d

    cache = {
        "key": cache_key,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "height_map": height_map,
        "facecolors": facecolors,
    }
    env._eval_terrain_render_cache = cache
    return cache


def _draw_terrain_surface(ax, env, alpha=0.35, terrain_stride=1):
    terrain = getattr(env, "terrain", None)
    if terrain is None or terrain.height_map is None:
        return

    cache = _get_terrain_render_cache(env, terrain_stride=terrain_stride, alpha_3d=alpha)
    if cache is None:
        return

    ax.plot_surface(
        cache["grid_x"],
        cache["grid_y"],
        cache["height_map"],
        facecolors=cache["facecolors"],
        linewidth=0,
        antialiased=True,
        shade=False,
    )


def _draw_terrain_heatmap(ax, env, cmap="terrain", alpha=0.55, terrain_stride=1):
    terrain = getattr(env, "terrain", None)
    if terrain is None or terrain.height_map is None:
        return

    cache = _get_terrain_render_cache(env, terrain_stride=terrain_stride, alpha_3d=0.35)
    if cache is None:
        return

    ax.pcolormesh(
        cache["grid_x"],
        cache["grid_y"],
        cache["height_map"],
        cmap=cmap,
        shading="auto",
        alpha=alpha,
        zorder=0,
    )


@lru_cache(maxsize=128)
def _cached_hemisphere_mesh(radius, height, theta_steps=72, phi_steps=36):
    theta = np.linspace(0.0, 2.0 * np.pi, 72)
    phi = np.linspace(0.0, np.pi / 2.0, 36)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x_grid = radius * np.cos(theta_grid) * np.sin(phi_grid)
    y_grid = radius * np.sin(theta_grid) * np.sin(phi_grid)
    z_grid = height * np.cos(phi_grid)
    strength = 0.35 + 0.65 * (np.cos(phi_grid) ** 0.7)
    rim_theta = np.linspace(0.0, 2.0 * np.pi, 120)
    ring_lines = []
    for frac in (0.3, 0.6, 0.9):
        phi_line = np.full_like(theta, frac * np.pi / 2.0)
        ring_lines.append(
            (
                radius * np.cos(theta) * np.sin(phi_line),
                radius * np.sin(theta) * np.sin(phi_line),
                height * np.cos(phi_line),
            )
        )
    return (
        x_grid,
        y_grid,
        z_grid,
        strength,
        tuple(ring_lines),
        radius * np.cos(rim_theta),
        radius * np.sin(rim_theta),
        np.zeros_like(rim_theta),
    )


def _draw_hemisphere(ax, x, y, z, radius, height, color="darkorange", alpha=0.34, z_offset=0.0):
    (
        x_grid,
        y_grid,
        z_grid,
        strength,
        ring_lines,
        rim_x,
        rim_y,
        rim_z,
    ) = _cached_hemisphere_mesh(float(radius), float(height))
    base_rgb = np.array(mcolors.to_rgb(color), dtype=np.float32)
    facecolors = np.ones((*strength.shape, 4), dtype=np.float32)
    facecolors[..., :3] = np.clip(base_rgb * strength[..., None] + 0.08, 0.0, 1.0)
    facecolors[..., 3] = alpha

    surface = ax.plot_surface(
        x_grid + x,
        y_grid + y,
        z_grid + z + z_offset,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    if hasattr(surface, "set_zsort"):
        surface.set_zsort("max")

    for x_line, y_line, z_line in ring_lines:
        ax.plot(
            x + x_line,
            y + y_line,
            z + z_offset + z_line,
            color="goldenrod",
            linewidth=0.9,
            alpha=0.55,
        )

    ax.plot(
        x + rim_x,
        y + rim_y,
        z + z_offset + rim_z,
        color="saddlebrown",
        linewidth=1.2,
        alpha=0.8,
    )

    return surface


def _plot_waypoint_env(
    ax3d,
    ax2d,
    env,
    step,
    history=None,
    history_stride=1,
    history_smooth_window=5,
    terrain_stride=1,
):
    ax3d.clear()
    render_meta = getattr(env, "_render_meta", {})
    geo_origin = render_meta.get("geo_origin") if isinstance(render_meta, dict) else None

    ax3d.set_xlim(0, env.X)
    ax3d.set_ylim(0, env.Y)
    ax3d.set_zlim(0, max(float(getattr(env, "Z", 0.0)), 15.0))
    ax3d.set_xlabel("X (km)")
    ax3d.set_ylabel("Y (km)")
    ax3d.set_zlabel("Z (m)" if _uses_raw_csv_heightmap_z(env) else "Z (km)")
    ax3d.set_title(f"Waypoint 3D View - Step: {step}", fontsize=14)
    ax3d.xaxis.pane.set_facecolor((0.94, 0.97, 1.0, 0.18))
    ax3d.yaxis.pane.set_facecolor((0.94, 0.97, 1.0, 0.18))
    ax3d.zaxis.pane.set_facecolor((0.96, 0.98, 1.0, 0.12))
    ax3d.grid(True, alpha=0.18)

    ax2d.clear()
    ax2d.set_xlim(0, env.X)
    ax2d.set_ylim(0, env.Y)
    ax2d.set_xlabel("X (km)")
    ax2d.set_ylabel("Y (km)")
    ax2d.set_title(f"Waypoint Top View - Step: {step}", fontsize=14)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(True, alpha=0.2)
    _configure_real_world_axes(ax3d, ax2d, geo_origin)

    _draw_terrain_surface(ax3d, env, terrain_stride=terrain_stride)
    _draw_terrain_heatmap(ax2d, env, terrain_stride=terrain_stride)

    task_paths = render_meta.get("task_paths", {}) if isinstance(render_meta, dict) else {}
    visible_tasks = render_meta.get("visible_tasks", []) if isinstance(render_meta, dict) else []
    global_subgoals = render_meta.get("global_subgoals", {}) if isinstance(render_meta, dict) else {}
    global_routes = render_meta.get("global_routes", {}) if isinstance(render_meta, dict) else {}
    if isinstance(geo_origin, dict):
        geo_text = (
            f"ENU origin -> lat {float(geo_origin.get('origin_lat', 0.0)):.3f}, "
            f"lon {float(geo_origin.get('origin_lon', 0.0)):.3f}"
        )
        ax2d.text(
            0.01,
            0.99,
            geo_text,
            transform=ax2d.transAxes,
            fontsize=8,
            color="dimgray",
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )
    damaged_agents = {
        str(agent)
        for agent in (render_meta.get("damaged_agents", []) if isinstance(render_meta, dict) else [])
    }
    inactive_agents = {
        str(agent)
        for agent in (render_meta.get("inactive_agents", []) if isinstance(render_meta, dict) else [])
    }
    discovered_hemisphere_indices = None
    if isinstance(render_meta, dict) and "discovered_hemisphere_indices" in render_meta:
        discovered_hemisphere_indices = {
            int(idx) for idx in render_meta.get("discovered_hemisphere_indices", [])
        }

    _draw_task_status_panels(ax2d, render_meta)

    for obs_idx, obs in enumerate(env.obstacles):
        if getattr(obs, "kind", "cylinder") == "hemisphere":
            if discovered_hemisphere_indices is not None and obs_idx not in discovered_hemisphere_indices:
                continue
            _draw_hemisphere(
                ax3d,
                obs.x,
                obs.y,
                obs.z,
                obs.radius,
                obs.height,
                color="darkorange",
                alpha=0.30,
                z_offset=0.0,
            )
            _draw_ground_disc(ax3d, obs.x, obs.y, obs.radius, color="darkorange", alpha=0.10)
            ax2d.add_patch(
                Circle((obs.x, obs.y), obs.radius, color="darkorange", alpha=0.22)
            )
            ax2d.add_patch(
                Circle(
                    (obs.x, obs.y),
                    obs.radius,
                    fill=False,
                    color="saddlebrown",
                    linewidth=1.5,
                    alpha=0.95,
                )
            )
        else:
            _draw_cylinder(ax3d, obs.x, obs.y, obs.radius, obs.height, color="steelblue", alpha=0.32)
            ax2d.add_patch(Circle((obs.x, obs.y), obs.radius, color="steelblue", alpha=0.22))
            ax2d.add_patch(Circle((obs.x, obs.y), obs.radius, fill=False, color="navy", linewidth=1.2, alpha=0.85))

    colors = plt.cm.rainbow(np.linspace(0, 1, env.n_agents))
    color_map = {agent: colors[index] for index, agent in enumerate(env.possible_agents)}
    collided_agents = {
        agent
        for agent in env.possible_agents
        if env.agent_obstacle_collisions.get(agent, 0) > 0
        or env.agent_inter_agent_collisions.get(agent, 0) > 0
    }

    for task in visible_tasks:
        if not isinstance(task, dict):
            continue
        task_position = np.asarray(task.get("position", []), dtype=np.float32)
        if task_position.shape[0] < 3:
            continue
        task_completed = bool(task.get("completed", False))
        task_color = "seagreen" if task_completed else "dimgray"
        task_marker = "X" if task_completed else "P"
        task_alpha = 0.72 if task_completed else 0.62
        task_size_3d = 58 if task_completed else 54
        task_size_2d = 44 if task_completed else 38
        ax3d.scatter(
            task_position[0],
            task_position[1],
            task_position[2],
            color=task_color,
            marker=task_marker,
            s=task_size_3d,
            alpha=task_alpha,
            edgecolors="white",
            linewidths=0.6,
        )
        task_icon_drawn = _draw_svg_icon(
            ax2d,
            TASK_ICON_PATH,
            task_position[0],
            task_position[1],
            size=(2.5 if task_completed else 2.2),
            color=task_color,
            alpha=task_alpha,
            zorder=9,
            edgecolor="white",
        )
        if not task_icon_drawn:
            ax2d.scatter(
                task_position[0],
                task_position[1],
                color=task_color,
                marker=task_marker,
                s=task_size_2d,
                alpha=task_alpha,
                edgecolors="white",
                linewidths=0.6,
            )

    for agent in env.possible_agents:
        if agent in inactive_agents:
            continue
        if agent not in env.agent_positions or agent not in env.agent_targets:
            continue

        color = color_map[agent]
        is_active = agent in env.agents
        is_done = getattr(env, "agent_dones", {}).get(agent, False)
        is_collided = agent in collided_agents
        is_damaged = agent in damaged_agents
        position = env.agent_positions[agent]
        target = env.agent_targets[agent]
        last_motion = env.agent_last_motion.get(agent, np.zeros(3, dtype=np.float32))
        remaining_task_points = [
            np.asarray(point, dtype=np.float32).copy()
            for point in task_paths.get(agent, [])
        ]
        current_subgoal = global_subgoals.get(agent)
        route_points = [
            np.asarray(point, dtype=np.float32).copy()
            for point in global_routes.get(agent, [])
        ]

        marker = "X" if is_collided else ("^" if is_done else "o")
        alpha = 0.95 if (is_active or is_done) else 0.45
        point_color = "crimson" if (is_collided or is_damaged) else color
        if is_damaged:
            ax3d.scatter(
                position[0],
                position[1],
                position[2],
                color=point_color,
                marker="o",
                s=35,
                edgecolors="none",
                alpha=0.18,
                label=agent,
            )
            ax3d.text(
                position[0],
                position[1],
                position[2],
                "❌",
                color="crimson",
                fontsize=16,
                ha="center",
                va="center",
            )
        else:
            ax3d.scatter(
                position[0],
                position[1],
                position[2],
                color=point_color,
                marker=marker,
                s=(150 if is_collided else 100),
                edgecolors=("white" if is_collided else "black"),
                linewidths=(1.6 if is_collided else 0.8),
                alpha=alpha,
                label=agent,
            )

        if is_active and np.linalg.norm(last_motion) > 0.05:
            ax3d.plot(
                [position[0], position[0] + last_motion[0]],
                [position[1], position[1] + last_motion[1]],
                [position[2], position[2] + last_motion[2]],
                color=color,
                linewidth=2.5,
            )

        ax3d.scatter(target[0], target[1], target[2], color=color, marker="*", s=150, alpha=0.9)
        ax3d.plot(
            [position[0], target[0]],
            [position[1], target[1]],
            [position[2], target[2]],
            color=color,
            linestyle="--",
            alpha=0.3,
        )
        if route_points:
            ordered_route = np.asarray(route_points, dtype=np.float32)
            render_route = _build_render_polyline(env, ordered_route)
            ax3d.plot(
                render_route[:, 0],
                render_route[:, 1],
                render_route[:, 2],
                color=color,
                linestyle=(0, (3, 2)),
                linewidth=2.0,
                alpha=0.8,
            )
            ax3d.scatter(
                ordered_route[:, 0],
                ordered_route[:, 1],
                ordered_route[:, 2],
                color=color,
                marker=".",
                s=16,
                alpha=0.45,
            )
        if current_subgoal is not None:
            current_subgoal = np.asarray(current_subgoal, dtype=np.float32)
            ax3d.scatter(
                current_subgoal[0],
                current_subgoal[1],
                current_subgoal[2],
                color=color,
                marker="D",
                s=64,
                edgecolors="white",
                linewidths=0.8,
                alpha=0.92,
            )
            ax3d.plot(
                [position[0], current_subgoal[0]],
                [position[1], current_subgoal[1]],
                [position[2], current_subgoal[2]],
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.5,
            )
        if len(remaining_task_points) > 1:
            ordered_targets = np.asarray(remaining_task_points, dtype=np.float32)
            ax3d.plot(
                ordered_targets[:, 0],
                ordered_targets[:, 1],
                ordered_targets[:, 2],
                color=color,
                linestyle=(0, (4, 2)),
                linewidth=1.6,
                alpha=0.65,
            )

        heading_vector = np.asarray(last_motion[:2], dtype=np.float32)
        if np.linalg.norm(heading_vector) <= 1e-6:
            heading_vector = np.asarray(target[:2] - position[:2], dtype=np.float32)
        heading_angle = float(np.arctan2(heading_vector[1], heading_vector[0])) if np.linalg.norm(heading_vector) > 1e-6 else 0.0

        if is_damaged:
            damaged_icon_drawn = _draw_svg_icon(
                ax2d,
                UAV_ICON_PATH,
                position[0],
                position[1],
                size=2.8,
                color=point_color,
                angle=heading_angle,
                alpha=0.14,
                zorder=7,
                edgecolor="none",
            )
            if not damaged_icon_drawn:
                ax2d.scatter(
                    position[0],
                    position[1],
                    color=point_color,
                    marker="o",
                    s=20,
                    edgecolors="none",
                    alpha=0.12,
                    label=agent,
                )
            ax2d.text(
                position[0],
                position[1],
                "❌",
                color="crimson",
                fontsize=18,
                ha="center",
                va="center",
                zorder=8,
            )
        else:
            uav_icon_drawn = _draw_svg_icon(
                ax2d,
                UAV_ICON_PATH,
                position[0],
                position[1],
                size=(3.2 if is_collided else 2.8),
                color=point_color,
                angle=heading_angle,
                alpha=alpha,
                zorder=12,
                edgecolor=("white" if is_collided else "black"),
            )
            if not uav_icon_drawn:
                ax2d.scatter(
                    position[0],
                    position[1],
                    color=point_color,
                    marker=marker,
                    s=(110 if is_collided else 80),
                    edgecolors=("white" if is_collided else "black"),
                    linewidths=(1.4 if is_collided else 0.8),
                    alpha=alpha,
                    label=agent,
                )
        ax2d.scatter(target[0], target[1], color=color, marker="*", s=120, alpha=0.9)
        ax2d.plot([position[0], target[0]], [position[1], target[1]], color=color, linestyle="--", alpha=0.3)
        if route_points:
            ordered_route = np.asarray(route_points, dtype=np.float32)
            render_route = _build_render_polyline(env, ordered_route)
            ax2d.plot(
                render_route[:, 0],
                render_route[:, 1],
                color=color,
                linestyle=(0, (3, 2)),
                linewidth=2.0,
                alpha=0.8,
            )
            ax2d.scatter(
                ordered_route[:, 0],
                ordered_route[:, 1],
                color=color,
                marker=".",
                s=12,
                alpha=0.45,
            )
        if current_subgoal is not None:
            current_subgoal = np.asarray(current_subgoal, dtype=np.float32)
            ax2d.scatter(
                current_subgoal[0],
                current_subgoal[1],
                color=color,
                marker="D",
                s=52,
                edgecolors="white",
                linewidths=0.8,
                alpha=0.92,
            )
            ax2d.plot(
                [position[0], current_subgoal[0]],
                [position[1], current_subgoal[1]],
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.5,
            )
        if len(remaining_task_points) > 1:
            ordered_targets = np.asarray(remaining_task_points, dtype=np.float32)
            ax2d.plot(
                ordered_targets[:, 0],
                ordered_targets[:, 1],
                color=color,
                linestyle=(0, (4, 2)),
                linewidth=1.6,
                alpha=0.65,
            )

        if history is not None and agent in history:
            path = np.array(history[agent], dtype=np.float32)
            if len(path) > 1:
                path = path[:: max(int(history_stride), 1)]
                path = _smooth_history_path(path, smooth_window=history_smooth_window)
                ax3d.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=2.0, alpha=0.7)
                ax2d.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, alpha=0.7)

    handles, labels = ax3d.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax3d.legend(
            by_label.values(),
            by_label.keys(),
            loc="center right",
            bbox_to_anchor=(-0.14, 0.5),
            frameon=True,
            facecolor="white",
            framealpha=0.88,
        )


def _resolve_model_path_from_dir(model_dir):
    candidate_paths = [
        os.path.join(model_dir, "best_model", "best_model.zip"),
        os.path.join(model_dir, "ppo_uav_waypoint_final.zip"),
    ]

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    if os.path.isdir(checkpoints_dir):
        checkpoint_files = [
            os.path.join(checkpoints_dir, name)
            for name in os.listdir(checkpoints_dir)
            if name.endswith(".zip")
        ]
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        candidate_paths.extend(checkpoint_files)

    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def train(args):
    is_resuming = bool(args.resume_run_id or args.resume_model_path)
    run_id = args.resume_run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./sb3_logs_waypoint/{run_id}/"
    model_dir = f"./sb3_models_waypoint/{run_id}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "algo": "PPO",
            "env": "UAVWaypoint3DMAPFEnv",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 256,
            "gamma": 0.99,
        }
    )
    config.update(_build_waypoint_env_kwargs(args_or_none=args, n_agents=args.n_agents, max_steps=args.max_steps))
    config["is_resuming"] = is_resuming
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"正在初始化包含 {args.n_agents} 架无人机的 waypoint 训练环境...")
    print(f"当前训练 Run ID: {run_id}")
    env = make_env(n_agents=args.n_agents, max_steps=args.max_steps, args=args)
    eval_env = make_env(n_agents=args.n_agents, max_steps=args.max_steps, args=args)
    env = VecMonitor(env, filename=os.path.join(log_dir, "train_monitor.csv"))
    eval_env = VecMonitor(eval_env, filename=os.path.join(log_dir, "eval_monitor.csv"))

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50000 // args.n_agents),
        save_path=os.path.join(model_dir, "checkpoints/"),
        name_prefix="uav_waypoint_ppo",
    )
    eval_freq = args.eval_freq
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model/"),
        log_path=os.path.join(log_dir, "eval/"),
        eval_freq=max(1, eval_freq // args.n_agents),
        deterministic=True,
        render=False,
    )

    #PPO模型参数
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    if is_resuming:
        model_path = args.resume_model_path or _resolve_model_path_from_dir(model_dir)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"未找到可用于恢复训练的模型文件，run_id={run_id!r}, model_dir={model_dir!r}"
            )
        print(f"正在恢复训练，加载模型: {model_path}")
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir, device=args.device)
    else:
        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device=args.device,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95, #GAE平滑系数，表示看多少步差分来估计优势
            clip_range=0.2, #clip，当优势估计过于乐观或悲观时截断L_clip的梯度
            ent_coef=0.01,  #熵正则项系数
        )

    print(f"\n开始训练，计划总步数: {args.total_timesteps}...")
    print("TensorBoard 可通过 `tensorboard --logdir sb3_logs_waypoint` 查看。")

    step_reward_callback = StepRewardCallback()
    episode_metrics_callback = EpisodeMetricsCallback()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[
            checkpoint_callback,
            eval_callback,
            step_reward_callback,
            episode_metrics_callback,
        ],
        progress_bar=True,
        reset_num_timesteps=not is_resuming,
    )

    model.save(os.path.join(model_dir, "ppo_uav_waypoint_final"))
    print("\n训练完成，最终模型已保存。")


def evaluate(args):
    explicit_eval_model_path = str(getattr(args, "eval_model_path", "") or "").strip()
    if explicit_eval_model_path:
        model_path = explicit_eval_model_path
        if not os.path.exists(model_path):
            print(f"未找到指定的评估模型文件: {model_path}")
            return
        model_dir = os.path.dirname(os.path.dirname(model_path))
    elif args.run_id:
        model_dir = f"./sb3_models_waypoint/{args.run_id}/"
    else:
        if not os.path.exists("./sb3_models_waypoint/"):
            print("未找到 ./sb3_models_waypoint/ 目录，请先训练。")
            return
        runs = [
            d
            for d in os.listdir("./sb3_models_waypoint/")
            if os.path.isdir(os.path.join("./sb3_models_waypoint/", d))
        ]
        if not runs:
            print("./sb3_models_waypoint/ 下没有模型目录。")
            return
        runs.sort(reverse=True)
        model_dir = f"./sb3_models_waypoint/{runs[0]}/"
        print(f"未指定 run_id，自动使用最新训练记录 {runs[0]}")

    if not explicit_eval_model_path:
        model_path = os.path.join(model_dir, "best_model/best_model.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "ppo_uav_waypoint_final.zip")
            if not os.path.exists(model_path):
                print(f"未在 {model_dir} 找到模型文件。")
                return

    print(f"正在加载模型用于 waypoint 环境测试: {model_path}")
    model = PPO.load(model_path, device=args.device)

    raw_env = UAVWaypoint3DMAPFEnv(
        **_build_waypoint_env_kwargs(
            args_or_none=args,
            n_agents=args.n_agents,
            max_steps=args.max_steps,
        ),
        log_collisions=True,
    )

    obs, infos = raw_env.reset()
    loaded_setup = getattr(args, "_loaded_config_json", None)
    if isinstance(loaded_setup, dict) and isinstance(loaded_setup.get("agents"), dict):
        restored_obs, restored_infos = _restore_eval_scene_from_setup(raw_env, loaded_setup)
        if restored_obs is not None and restored_infos is not None:
            obs, infos = restored_obs, restored_infos

    plt.ion()
    fig = plt.figure(figsize=(20, 8))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    fig.subplots_adjust(right=0.78, wspace=0.18)
    ax3d.view_init(elev=30, azim=45)
    eval_artifact_dir = os.path.join(model_dir, "eval_artifacts")
    eval_run_name = datetime.datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    captured_frames = []
    setup_path = _save_eval_setup(raw_env, args, model_dir, eval_run_name)
    print(f"已保存评估场景配置: {setup_path}")

    total_rewards = {agent: 0.0 for agent in raw_env.possible_agents}
    history = {
        agent: [raw_env.agent_positions[agent].copy()]
        for agent in raw_env.possible_agents
        if agent in raw_env.agent_positions
    }
    previous_eval_actions = {
        agent: np.zeros(3, dtype=np.float32) for agent in raw_env.possible_agents
    }
    collided_eval_agents = set()
    render_every = max(int(args.eval_render_every), 1)
    capture_every = max(int(args.eval_capture_every), 1)
    history_stride = max(int(args.eval_history_stride), 1)
    terrain_stride = max(int(args.eval_terrain_stride), 1)
    pause_time = max(float(args.eval_pause), 0.0)
    print("开始渲染 waypoint 策略表现...")

    #主验证循环
    for step in range(args.max_steps):
        for agent in raw_env.possible_agents:
            if agent in raw_env.agent_positions:
                history[agent].append(raw_env.agent_positions[agent].copy())

        should_render = (step % render_every == 0)
        should_capture = (step % capture_every == 0)
        if should_render or should_capture:
            _plot_waypoint_env(
                ax3d,
                ax2d,
                raw_env,
                step,
                history=history,
                history_stride=history_stride,
                terrain_stride=terrain_stride,
            )
            fig.canvas.draw()
            if should_render and pause_time > 0.0:
                plt.pause(pause_time)
            if should_capture:
                captured_frames.append(_capture_figure_frame(fig))

        actions = {}
        for agent in raw_env.agents:
            if agent in collided_eval_agents:
                actions[agent] = np.zeros(3, dtype=np.float32)
                previous_eval_actions[agent] = actions[agent]
                continue
            raw_action, _states = model.predict(obs[agent], deterministic=True)
            eval_cfg = _get_hidden_eval_policy_config()
            action = _smooth_eval_action(
                raw_env,
                agent,
                raw_action,
                previous_eval_actions[agent],
                smoothing_factor=np.clip(eval_cfg["smoothing"], 0.0, 0.95),
                target_blend=np.clip(eval_cfg["goal_blend"], 0.0, 1.0),
            )
            action = _apply_eval_emergency_avoidance(raw_env, agent, action, args)
            action = _enforce_eval_min_agl(
                raw_env,
                agent,
                action,
                min_agl=max(float(eval_cfg["min_safe_agl"]), 0.0),
            )
            actions[agent] = action
            previous_eval_actions[agent] = action

        obs, rewards, terminations, truncations, infos = raw_env.step(actions)

        for agent, reward in rewards.items():
            total_rewards[agent] += reward

        for agent, info in infos.items():
            if info.get("last_collision_reason"):
                lidar_obs_str = _format_lidar_observation(raw_env, agent)
                print(
                    f"[Eval Collision][step={step}] {agent}: "
                    f"reason={info.get('last_collision_reason')} "
                    f"lidar={lidar_obs_str}"
                )
                collided_eval_agents.add(agent)

        raw_env.agents = [
            agent for agent in raw_env.agents
            if agent not in collided_eval_agents and not raw_env.agent_dones.get(agent, False)
        ]

        all_agents_finished = all(
            raw_env.agent_dones.get(agent, False) or agent in collided_eval_agents
            for agent in raw_env.possible_agents
        )

        if all_agents_finished:
            _plot_waypoint_env(
                ax3d,
                ax2d,
                raw_env,
                step + 1,
                history=history,
                history_stride=history_stride,
                terrain_stride=terrain_stride,
            )
            fig.canvas.draw()
            captured_frames.append(_capture_figure_frame(fig))
            print(f"评估在第 {step} 步提前结束：所有无人机均已到达目标或发生碰撞。")
            break

        if all(truncations.values()) or all(terminations.values()):
            _plot_waypoint_env(
                ax3d,
                ax2d,
                raw_env,
                step + 1,
                history=history,
                history_stride=history_stride,
                terrain_stride=terrain_stride,
            )
            fig.canvas.draw()
            captured_frames.append(_capture_figure_frame(fig))
            print(f"环境在第 {step} 步结束。")
            break

    gif_path, final_image_path = _save_eval_artifacts(
        captured_frames,
        eval_artifact_dir,
        eval_run_name,
        gif_fps=args.eval_gif_fps,
    )

    plt.ioff()
    plt.show()

    print(f"评估过程 GIF 已保存到: {gif_path}")
    print(f"评估最后一步图片已保存到: {final_image_path}")
    print("\n测试回合结束，各无人机累计回报如下:")
    for agent, reward in total_rewards.items():
        print(f"  {agent}: {reward:.2f}")


def evaluate(args):
    explicit_eval_model_path = str(getattr(args, "eval_model_path", "") or "").strip()
    if explicit_eval_model_path:
        model_path = explicit_eval_model_path
        if not os.path.exists(model_path):
            print(f"未找到指定的评估模型文件: {model_path}")
            return
        model_dir = os.path.dirname(os.path.dirname(model_path))
    elif args.run_id:
        model_dir = f"./sb3_models_waypoint/{args.run_id}/"
    else:
        if not os.path.exists("./sb3_models_waypoint/"):
            print("未找到 ./sb3_models_waypoint/ 目录，请先训练。")
            return
        runs = [
            d
            for d in os.listdir("./sb3_models_waypoint/")
            if os.path.isdir(os.path.join("./sb3_models_waypoint/", d))
        ]
        if not runs:
            print("./sb3_models_waypoint/ 下没有模型目录。")
            return
        runs.sort(reverse=True)
        model_dir = f"./sb3_models_waypoint/{runs[0]}/"
        print(f"未指定 run_id，自动使用最新训练记录 {runs[0]}")

    if not explicit_eval_model_path:
        model_path = os.path.join(model_dir, "best_model/best_model.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "ppo_uav_waypoint_final.zip")
            if not os.path.exists(model_path):
                print(f"未在 {model_dir} 找到模型文件。")
                return

    print(f"正在加载模型用于 waypoint 环境测试: {model_path}")
    model = PPO.load(model_path, device=args.device)

    raw_env = UAVWaypoint3DMAPFEnv(
        **_build_waypoint_env_kwargs(
            args_or_none=args,
            n_agents=args.n_agents,
            max_steps=args.max_steps,
        ),
        log_collisions=True,
    )

    obs, infos = raw_env.reset()
    loaded_setup = getattr(args, "_loaded_config_json", None)
    if isinstance(loaded_setup, dict) and isinstance(loaded_setup.get("agents"), dict):
        restored_obs, restored_infos = _restore_eval_scene_from_setup(raw_env, loaded_setup)
        if restored_obs is not None and restored_infos is not None:
            obs, infos = restored_obs, restored_infos

    eval_artifact_dir = os.path.join(model_dir, "eval_artifacts")
    eval_run_name = datetime.datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    captured_frames = []
    render_snapshots = []
    setup_path = _save_eval_setup(raw_env, args, model_dir, eval_run_name)
    print(f"已保存评估场景配置: {setup_path}")

    total_rewards = {agent: 0.0 for agent in raw_env.possible_agents}
    history = {
        agent: [raw_env.agent_positions[agent].copy()]
        for agent in raw_env.possible_agents
        if agent in raw_env.agent_positions
    }
    previous_eval_actions = {
        agent: np.zeros(3, dtype=np.float32) for agent in raw_env.possible_agents
    }
    collided_eval_agents = set()
    render_every = max(int(args.eval_render_every), 1)
    capture_every = max(int(args.eval_capture_every), 1)
    history_stride = max(int(args.eval_history_stride), 1)
    terrain_stride = max(int(args.eval_terrain_stride), 1)
    pause_time = max(float(args.eval_pause), 0.0)
    render_mode_enabled = bool(getattr(args, "eval_render_mode", False))
    deadlock_window = max(int(args.eval_deadlock_window), 1)
    deadlock_motion_epsilon = max(float(args.eval_deadlock_motion_epsilon), 0.0)
    deadlock_distance_epsilon = max(float(args.eval_deadlock_distance_epsilon), 0.0)
    deadlock_motion_history = {
        agent: deque(maxlen=deadlock_window) for agent in raw_env.possible_agents
    }
    deadlock_distance_history = {
        agent: deque(maxlen=deadlock_window) for agent in raw_env.possible_agents
    }
    total_decision_time = 0.0
    decision_steps = 0
    print("开始采样 waypoint 策略表现...")

    render_snapshots.append(_make_eval_render_snapshot(raw_env, 0, history))

    for step in range(args.max_steps):
        for agent in raw_env.possible_agents:
            if agent in raw_env.agent_positions:
                history[agent].append(raw_env.agent_positions[agent].copy())

        actions = {}
        decision_start = time.perf_counter()
        for agent in raw_env.agents:
            if agent in collided_eval_agents:
                actions[agent] = np.zeros(3, dtype=np.float32)
                previous_eval_actions[agent] = actions[agent]
                continue
            raw_action, _states = model.predict(obs[agent], deterministic=True)
            eval_cfg = _get_hidden_eval_policy_config()
            action = _smooth_eval_action(
                raw_env,
                agent,
                raw_action,
                previous_eval_actions[agent],
                smoothing_factor=np.clip(eval_cfg["smoothing"], 0.0, 0.95),
                target_blend=np.clip(eval_cfg["goal_blend"], 0.0, 1.0),
            )
            action = _apply_eval_emergency_avoidance(raw_env, agent, action, args)
            action = _enforce_eval_min_agl(
                raw_env,
                agent,
                action,
                min_agl=max(float(eval_cfg["min_safe_agl"]), 0.0),
            )
            actions[agent] = action.astype(np.float32)
            previous_eval_actions[agent] = actions[agent]
        total_decision_time += time.perf_counter() - decision_start
        decision_steps += 1

        obs, rewards, terminations, truncations, infos = raw_env.step(actions)

        for agent, reward in rewards.items():
            total_rewards[agent] += reward

        for agent, info in infos.items():
            if info.get("last_collision_reason"):
                lidar_obs_str = _format_lidar_observation(raw_env, agent)
                print(
                    f"[Eval Collision][step={step}] {agent}: "
                    f"reason={info.get('last_collision_reason')} "
                    f"lidar={lidar_obs_str}"
                )
                collided_eval_agents.add(agent)

        deadlock_detected = False
        deadlocked_agent = None
        for agent in raw_env.possible_agents:
            if (
                agent in collided_eval_agents
                or raw_env.agent_dones.get(agent, False)
                or raw_env.agent_timeouts.get(agent, False)
                or agent not in raw_env.agent_positions
            ):
                deadlock_motion_history[agent].clear()
                deadlock_distance_history[agent].clear()
                continue

            motion_norm = float(
                np.linalg.norm(raw_env.agent_last_motion.get(agent, np.zeros(3, dtype=np.float32)))
            )
            remaining_distance = infos.get(agent, {}).get("remaining_distance")
            if remaining_distance is None:
                remaining_distance = float(
                    np.linalg.norm(raw_env.agent_targets[agent] - raw_env.agent_positions[agent])
                )
            deadlock_motion_history[agent].append(motion_norm)
            deadlock_distance_history[agent].append(float(remaining_distance))

            if (
                len(deadlock_motion_history[agent]) >= deadlock_window
                and max(deadlock_motion_history[agent]) <= deadlock_motion_epsilon
                and (
                    max(deadlock_distance_history[agent]) - min(deadlock_distance_history[agent])
                ) <= deadlock_distance_epsilon
            ):
                deadlock_detected = True
                deadlocked_agent = agent
                break

        raw_env.agents = [
            agent for agent in raw_env.agents
            if agent not in collided_eval_agents and not raw_env.agent_dones.get(agent, False)
        ]

        if ((step + 1) % render_every == 0) or ((step + 1) % capture_every == 0):
            render_snapshots.append(_make_eval_render_snapshot(raw_env, step + 1, history))

        all_agents_finished = all(
            raw_env.agent_dones.get(agent, False) or agent in collided_eval_agents
            for agent in raw_env.possible_agents
        )

        if all_agents_finished:
            render_snapshots.append(_make_eval_render_snapshot(raw_env, step + 1, history))
            print(f"评估在第 {step} 步提前结束：所有无人机均已到达目标或发生碰撞。")
            break

        if deadlock_detected:
            render_snapshots.append(_make_eval_render_snapshot(raw_env, step + 1, history))
            print(f"评估在第 {step} 步提前结束：{deadlocked_agent} 发生死锁。")
            break

        if all(truncations.values()) or all(terminations.values()):
            render_snapshots.append(_make_eval_render_snapshot(raw_env, step + 1, history))
            print(f"环境在第 {step} 步结束。")
            break

    print("开始回放渲染 waypoint 策略表现...")
    fig = plt.figure(figsize=(20, 8))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    fig.subplots_adjust(right=0.78, wspace=0.18)
    ax3d.view_init(elev=30, azim=45)
    if render_mode_enabled:
        plt.ion()
    else:
        plt.ioff()

    if render_snapshots:
        for snapshot_idx, snapshot in enumerate(render_snapshots):
            should_capture = (
                snapshot_idx == len(render_snapshots) - 1
                or (snapshot_idx % capture_every == 0)
            )
            should_draw_frame = bool(render_mode_enabled) or bool(should_capture)
            if not should_draw_frame:
                continue
            replay_history = _apply_eval_render_snapshot(raw_env, snapshot)
            _plot_waypoint_env(
                ax3d,
                ax2d,
                raw_env,
                snapshot["step"],
                history=replay_history,
                history_stride=history_stride,
                terrain_stride=terrain_stride,
            )
            fig.canvas.draw()
            if should_capture:
                captured_frames.append(_capture_figure_frame(fig))
            if render_mode_enabled and pause_time > 0.0:
                plt.pause(pause_time)

    gif_path, final_image_path = _save_eval_artifacts(
        captured_frames,
        eval_artifact_dir,
        eval_run_name,
        gif_fps=args.eval_gif_fps,
    )

    if render_mode_enabled:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)

    print(f"评估过程 GIF 已保存到: {gif_path}")
    print(f"评估最后一步图片已保存到: {final_image_path}")
    if decision_steps > 0:
        print(f"平均决策时间: {1000.0 * total_decision_time / decision_steps:.3f} ms/step")
    print("\n测试回合结束，各无人机累计回报如下:")
    for agent, reward in total_rewards.items():
        print(f"  {agent}: {reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waypoint 3D UAV PPO 训练与评估脚本")
    parser.add_argument(
        "--config_json",
        type=str,
        default="",
        help="从保存的 config.json 恢复参数；命令行显式传入的参数会覆盖 JSON 中的值",
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--n_agents", type=int, default=5, help="训练/测试的无人机数量")
    parser.add_argument("--max_steps", type=int, default=3000, help="每回合最大步数")
    parser.add_argument("--model_dir", type=str, default=None, help="给定模型位置")
    parser.add_argument(
        "--eval_model_path",
        type=str,
        default="",
        help="评估模式下显式指定要加载的模型文件路径（例如某个 checkpoint zip）",
    )
    parser.add_argument("--space_x", type=float, default=100.0, help="环境 X 尺寸")
    parser.add_argument("--space_y", type=float, default=100.0, help="环境 Y 尺寸")
    parser.add_argument("--space_z", type=float, default=100.0, help="环境 Z 尺寸")
    parser.add_argument(
        "--obstacle_rule",
        type=str,
        default="terrain",
        choices=["primitives", "terrain", "hybrid"],
        help="障碍物规则",
    )
    parser.add_argument(
        "--num_obstacles",
        type=int,
        default=0,
        help="原始几何障碍数量，仅 primitives/hybrid 生效",
    )
    parser.add_argument("--eval_freq", type=float, default=10000, help="训练过程的验证频率")
    parser.add_argument("--hemisphere_fraction", type=float, default=0.5, help="半球障碍占比")
    parser.add_argument("--obs_radius_min", type=float, default=2.0, help="圆柱障碍最小半径")
    parser.add_argument("--obs_radius_max", type=float, default=8.0, help="圆柱障碍最大半径")
    parser.add_argument("--obs_height_min", type=float, default=10.0, help="圆柱障碍最小高度")
    parser.add_argument("--obs_height_max", type=float, default=40.0, help="圆柱障碍最大高度")
    parser.add_argument("--hemisphere_radius_min", type=float, default=4.0, help="半球障碍最小半径")
    parser.add_argument("--hemisphere_radius_max", type=float, default=10.0, help="半球障碍最大半径")
    parser.add_argument("--hemisphere_height_min", type=float, default=10.0, help="半球障碍最小高度")
    parser.add_argument("--hemisphere_height_max", type=float, default=10.0, help="半球障碍最大高度")
    parser.add_argument(
        "--terrain_enabled",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="是否启用地形障碍", 
    )
    parser.add_argument(
        "--terrain_source",
        type=str,
        default="csv",
        choices=["procedural", "csv"],
        help="地形来源",
    )
    parser.add_argument(
        "--terrain_csv_path",
        type=str,
        default=DEFAULT_TERRAIN_CSV,
        help="CSV 高度图路径",
    )
    parser.add_argument(
        "--terrain_grid_resolution",
        type=float,
        default=1.0,
        help="CSV 栅格分辨率",
    )
    parser.add_argument(
        "--terrain_csv_target_x",
        type=float,
        default=None,
        help="将 CSV 地形重采样到目标 X 尺寸；不填则保持原始尺寸",
    )
    parser.add_argument(
        "--terrain_csv_target_y",
        type=float,
        default=None,
        help="将 CSV 地形重采样到目标 Y 尺寸；不填则保持原始尺寸",
    )
    parser.add_argument(
        "--terrain_csv_target_z",
        type=float,
        default=None,
        help="将 CSV 高度线性压缩到 0~目标 Z；不填则保持原始高度范围",
    )
    parser.add_argument(
        "--terrain_csv_z_scale",
        type=float,
        default=0.001,
        help="CSV 高度图 z 缩放因子；默认 0.001，表示将米换算为千米",
    )
    parser.add_argument("--terrain_origin_x", type=float, default=0.0, help="CSV 地形原点 X")
    parser.add_argument("--terrain_origin_y", type=float, default=0.0, help="CSV 地形原点 Y")
    parser.add_argument(
        "--terrain_min_clearance",
        type=float,
        default=0.01,
        help="最低离地高度",
    )
    parser.add_argument(
        "--terrain_max_clearance",
        type=float,
        default=1.0,
        help="最高离地高度",
    )
    parser.add_argument(
        "--terrain_match_space_dim",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="是否使用 CSV 地形尺寸覆盖空间尺寸",
    )
    parser.add_argument(
        "--terrain_spawn_clearance",
        type=float,
        default=0.1,
        help="起点采样离地余量",
    )
    parser.add_argument(
        "--spawn_max_height",
        type=float,
        default=30.0,
        help="起点采样高度上限（km）",
    )
    parser.add_argument(
        "--ground_target_clearance",
        type=float,
        default=0.5,
        help="目标点离地高度，默认 0.5 km（500 m）",
    )
    parser.add_argument(
        "--lidar_step_size",
        type=float,
        default=None,
        help="LiDAR 采样步长，不填则自动根据地形模式设置",
    )
    parser.add_argument("--lidar_rays", type=int, default=24, help="LiDAR 射线数量")
    parser.add_argument("--lidar_range", type=float, default=0.2, help="LiDAR 探测范围")
    parser.add_argument(
        "--terrain_num_hemispheres",
        type=int,
        default=0,
        help="叠加到地形上的随机半球数量",
    )
    parser.add_argument("--terrain_radius_min", type=float, default=5.0, help="随机半球最小半径")
    parser.add_argument("--terrain_radius_max", type=float, default=10.0, help="随机半球最大半径")
    parser.add_argument("--terrain_height_min", type=float, default=10.0, help="随机半球最小高度")
    parser.add_argument("--terrain_height_max", type=float, default=10.0, help="随机半球最大高度")
    parser.add_argument("--uav_speed", type=float, default=0.06, help="路点模型标称速度")
    parser.add_argument("--decision_period", type=float, default=1.0, help="单次决策周期")
    parser.add_argument("--goal_threshold", type=float, default=1.0, help="到达目标阈值")

    # 奖励函数
    parser.add_argument("--step_penalty", type=float, default=-0.02, help="每步惩罚")
    parser.add_argument("--progress_reward_scale", type=float, default=2.0, help="进度奖励权重")
    parser.add_argument("--obstacle_collision_penalty", type=float, default=80, help="障碍碰撞惩罚")
    parser.add_argument("--agent_collision_penalty", type=float, default=160, help="机间碰撞惩罚")
    parser.add_argument("--goal_reward", type=float, default=1500, help="到达目标奖励")
    parser.add_argument("--timeout_penalty_scale", type=float, default=10, help="超时惩罚系数")
    
    parser.add_argument("--collision_radius", type=float, default=0.02, help="机间碰撞半径")
    parser.add_argument(
        "--near_goal_collision_free_radius",
        type=float,
        default=1.0,
        help="双方都在各自目标点附近时豁免机间碰撞的半径",
    )
    parser.add_argument(
        "--hemisphere_exclusion_buffer",
        type=float,
        default=5.0,
        help="起点和目标点避开半球障碍物边缘的最小水平缓冲距离",
    )
    parser.add_argument(
        "--avoid_mountain_basins",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="是否避免在高海拔山地区域的低洼地带生成起点和目标点",
    )
    parser.add_argument(
        "--basin_window_radius",
        type=int,
        default=4,
        help="判定山区低洼地带时使用的局部窗口半径（栅格）",
    )
    parser.add_argument(
        "--basin_highland_threshold",
        type=float,
        default=20.0,
        help="局部平均高度超过该阈值时，区域被视为高海拔山地",
    )
    parser.add_argument(
        "--basin_relief_threshold",
        type=float,
        default=3.0,
        help="若局部最高点与当前位置高差超过该阈值，则视为低洼候选",
    )
    parser.add_argument(
        "--basin_high_neighbor_ratio",
        type=float,
        default=0.35,
        help="局部窗口内高于高海拔阈值的格点比例要求",
    )

    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=4_000_000,
        help="训练总步数",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="评估模式下指定训练 run_id，不填则自动选择最新模型",
    )
    parser.add_argument(
        "--resume_run_id",
        type=str,
        default="",
        help="训练模式下恢复指定 run_id 的训练，日志和模型会继续写入该目录",
    )
    parser.add_argument(
        "--resume_model_path",
        type=str,
        default="",
        help="训练模式下从指定模型文件恢复训练；不填则在 resume_run_id 目录下自动查找",
    )
    parser.add_argument(
        "--eval_gif_fps",
        type=int,
        default=10,
        help="eval 导出 GIF 的帧率",
    )
    parser.add_argument(
        "--eval_render_every",
        type=int,
        default=5,
        help="eval 可视化每隔多少步刷新一次图像",
    )
    parser.add_argument(
        "--eval_capture_every",
        type=int,
        default=5,
        help="eval 每隔多少步抓取一帧写入 GIF",
    )
    parser.add_argument(
        "--eval_history_stride",
        type=int,
        default=3,
        help="eval 绘制轨迹时的下采样步长",
    )
    parser.add_argument(
        "--eval_terrain_stride",
        type=int,
        default=2,
        help="eval 绘制地形时的网格下采样步长",
    )
    parser.add_argument(
        "--eval_pause",
        type=float,
        default=0.0,
        help="eval 每次渲染后的暂停时长，0 表示尽快刷新",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PPO 训练/评估设备，例如 cpu、cuda、auto",
    )
    parser.add_argument(
        "--eval_render_mode",
        type=_parse_bool_arg,
        default=False,
        help="是否在 eval 回放阶段实时显示 matplotlib 窗口",
    )
    parser.add_argument(
        "--eval_deadlock_window",
        type=int,
        default=10,
        help="eval 死锁检测窗口长度",
    )
    parser.add_argument(
        "--eval_deadlock_motion_epsilon",
        type=float,
        default=0.01,
        help="eval 死锁检测的运动量阈值",
    )
    parser.add_argument(
        "--eval_deadlock_distance_epsilon",
        type=float,
        default=0.001,
        help="eval 死锁检测的剩余距离变化阈值",
    )
    args = parser.parse_args()
    args = _merge_args_from_config_json(args, parser, sys.argv[1:])

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
