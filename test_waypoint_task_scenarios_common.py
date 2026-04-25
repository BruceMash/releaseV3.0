import argparse
import copy
import datetime
import importlib
import importlib.util
import json
import math
import os
import sys
import time

plugin_dir = r"D:\RL_env\Lib\site-packages\PyQt6\Qt6\plugins\platforms"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from Envs.continuous_3d_waypoint_env import (
    UAVWaypoint3DMAPFEnv,
    CylinderObstacle,
    EllipsoidalHemisphereObstacle,
)
from Envs.continuous_3d_waypoint_env_global_planner import (
    UAVWaypoint3DMAPFEnv as GlobalPlannerWaypointEnv,
)
from train_sb3_ppo_waypoint import (
    DEFAULT_TERRAIN_CSV,
    _apply_eval_emergency_avoidance,
    _build_waypoint_env_kwargs,
    _capture_figure_frame,
    _enforce_eval_min_agl,
    _get_hidden_eval_policy_config,
    _make_eval_render_snapshot,
    _apply_eval_render_snapshot,
    _plot_waypoint_env,
    _save_eval_artifacts,
    _smooth_eval_action,
    _resolve_model_path_from_dir,
    _parse_bool_arg,
    _merge_args_from_config_json,
)

plugin_dir = r"D:\RL_env\Lib\site-packages\PyQt6\Qt6\plugins\platforms"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir

DEFAULT_MODEL_ROOT = "./sb3_models_waypoint"
DEFAULT_GLOBAL_PLANNER_MODEL_ROOT = "./sb3_models_waypoint"


def _parse_bool(value): # 处理parser类的布尔值
    return _parse_bool_arg(value)


def _resolve_dotted_attr(obj, dotted_name): # 
    target = obj
    for part in str(dotted_name).split("."):
        part = part.strip()
        if not part:
            continue
        if not hasattr(target, part):
            raise AttributeError(f"{target!r} 不存在属性 {part!r}")
        target = getattr(target, part)
    return target


def _load_python_module(module_ref):    # 加载外部python脚本
    module_ref = str(module_ref or "").strip()
    if not module_ref:
        return None

    if module_ref.endswith(".py") or os.path.isfile(module_ref):
        module_path = os.path.abspath(module_ref)
        module_name = f"external_assignment_provider_{abs(hash(module_path))}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法从 {module_path} 加载外部任务分配模块。")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return importlib.import_module(module_ref)


def load_assignment_provider(module_ref, callable_name):    # 
    module_ref = str(module_ref or "").strip()
    callable_name = str(callable_name or "").strip()
    if not module_ref and not callable_name:
        return None
    if not module_ref or not callable_name:
        raise ValueError("assignment_provider_module 和 assignment_provider_callable 需要同时提供。")

    module = _load_python_module(module_ref)
    provider = _resolve_dotted_attr(module, callable_name)
    if not callable(provider):
        raise TypeError("外部任务分配接口必须是可调用对象。")
    return provider


def get_assignment_provider(args):
    if getattr(args, "_assignment_provider_loaded", False):
        return getattr(args, "_assignment_provider", None)

    provider = load_assignment_provider(
        getattr(args, "assignment_provider_module", ""),
        getattr(args, "assignment_provider_callable", ""),
    )
    setattr(args, "_assignment_provider", provider)
    setattr(args, "_assignment_provider_loaded", True)
    return provider

def _as_provider_position(value):
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 2:
        return None
    if arr.size == 2:
        arr = np.concatenate([arr, np.zeros(1, dtype=np.float32)])
    return arr[:3].copy()


def _task_id_from_index(index):
    return f"task_{int(index) + 1:02d}"


def _collect_task_positions_from_scenario(scenario_payload, task_count):
    task_positions = {}
    if not isinstance(scenario_payload, dict):
        return task_positions

    raw_positions = scenario_payload.get("task_positions") or scenario_payload.get("tasks")
    if isinstance(raw_positions, dict):
        for task_id, value in raw_positions.items():
            if isinstance(value, dict):
                value = value.get("position") or value.get("pos") or value.get("xyz")
            pos = _as_provider_position(value)
            if pos is not None:
                task_positions[str(task_id)] = pos
    elif isinstance(raw_positions, list):
        for index, item in enumerate(raw_positions[: max(int(task_count), 0)]):
            task_id = _task_id_from_index(index)
            value = item
            if isinstance(item, dict):
                task_id = str(item.get("task_id") or item.get("id") or task_id)
                value = item.get("position") or item.get("pos") or item.get("xyz")
            pos = _as_provider_position(value)
            if pos is not None:
                task_positions[task_id] = pos
    return task_positions


def _collect_provider_available_agents(agent_names, provider_context):
    agent_names = [str(agent) for agent in agent_names]
    provider_context = provider_context or {}
    env = provider_context.get("env")
    args = provider_context.get("args")
    scenario_payload = provider_context.get("scenario_payload")
    if not agent_names and env is not None:
        env_possible_agents = [str(agent) for agent in getattr(env, "possible_agents", [])]
        if env_possible_agents:
            agent_names = env_possible_agents
        elif isinstance(getattr(env, "agent_positions", None), dict):
            agent_names = [str(agent) for agent in env.agent_positions.keys()]
    if not agent_names and args is not None:
        try:
            agent_names = [f"agent_{idx}" for idx in range(max(int(getattr(args, "n_agents", 0)), 0))]
        except (TypeError, ValueError):
            agent_names = []
    requested = None
    if isinstance(scenario_payload, dict):
        requested = scenario_payload.get("active_n_agents", scenario_payload.get("available_n_agents"))
    if requested is None and args is not None:
        requested = getattr(args, "active_n_agents", None)
    if requested is not None and int(requested) > 0:
        try:
            active_count = int(np.clip(int(requested), 0, len(agent_names)))
            agent_names = agent_names[:active_count]
        except (TypeError, ValueError):
            pass

    state = provider_context.get("state")
    inactive_agents = set()
    if isinstance(state, dict):
        inactive_agents.update(str(agent) for agent in state.get("inactive_agents", set()))
        active_agents = state.get("active_agents")
        if active_agents:
            active_set = {str(agent) for agent in active_agents}
            agent_names = [agent for agent in agent_names if agent in active_set]
    return [agent for agent in agent_names if agent not in inactive_agents]


def _collect_provider_positions(agent_names, provider_context):
    provider_context = provider_context or {}
    env = provider_context.get("env")
    positions = {}
    raw_positions = getattr(env, "agent_positions", {}) if env is not None else {}
    if isinstance(raw_positions, dict):
        for agent in agent_names:
            pos = _as_provider_position(raw_positions.get(agent))
            if pos is not None:
                positions[str(agent)] = pos
    return positions


def _collect_provider_task_positions(task_count, provider_context):
    provider_context = provider_context or {}
    state = provider_context.get("state")
    task_positions = {}

    raw_context_positions = provider_context.get("task_positions")
    if isinstance(raw_context_positions, dict):
        for task_id, value in raw_context_positions.items():
            pos = _as_provider_position(value)
            if pos is not None:
                task_positions[str(task_id)] = pos
    elif isinstance(raw_context_positions, list):
        for index, value in enumerate(raw_context_positions[: max(int(task_count), 0)]):
            pos = _as_provider_position(value)
            if pos is not None:
                task_positions[_task_id_from_index(index)] = pos

    if isinstance(state, dict):
        for task_id, task in state.get("tasks", {}).items():
            if not isinstance(task, dict):
                continue
            pos = _as_provider_position(task.get("position"))
            if pos is not None:
                task_positions[str(task_id)] = pos

    if not task_positions:
        task_positions.update(
            _collect_task_positions_from_scenario(provider_context.get("scenario_payload"), task_count)
        )

    env = provider_context.get("env")
    if not task_positions and env is not None and isinstance(getattr(env, "task_positions", None), dict):
        for task_id, value in env.task_positions.items():
            pos = _as_provider_position(value)
            if pos is not None:
                task_positions[str(task_id)] = pos

    return task_positions


def _collect_pending_task_ids(task_count, provider_context, task_positions):
    provider_context = provider_context or {}
    state = provider_context.get("state")
    if isinstance(state, dict) and state.get("tasks"):
        return [
            str(task_id)
            for task_id, task in state.get("tasks", {}).items()
            if isinstance(task, dict)
            and task.get("active", True)
            and not task.get("completed", False)
        ]
    if task_positions:
        return list(task_positions.keys())
    return [_task_id_from_index(index) for index in range(max(int(task_count), 0))]


def _enrich_assignment_provider_context(agent_names, task_count, provider_context):
    provider_kwargs = dict(provider_context or {})
    available_agent_names = _collect_provider_available_agents(agent_names, provider_kwargs)
    task_positions = _collect_provider_task_positions(task_count, provider_kwargs)
    pending_task_ids = _collect_pending_task_ids(task_count, provider_kwargs, task_positions)
    pending_task_positions = {
        task_id: task_positions[task_id]
        for task_id in pending_task_ids
        if task_id in task_positions
    }
    provider_kwargs["available_agent_names"] = list(available_agent_names)
    provider_kwargs["available_agent_positions"] = _collect_provider_positions(available_agent_names, provider_kwargs)
    provider_kwargs["task_positions"] = task_positions
    provider_kwargs["pending_task_ids"] = list(pending_task_ids)
    provider_kwargs["pending_task_positions"] = pending_task_positions
    return provider_kwargs


def call_assignment_provider(
    assignment_provider,
    agent_names,
    task_count,
    assignment_override=None,
    provider_context=None,
):      # 唤起任务分配
    if not callable(assignment_provider):
        return {}

    provider_kwargs = dict(provider_context or {})
    provider_kwargs.setdefault("phase", "initial")
    provider_kwargs.setdefault("event", None)
    provider_kwargs.setdefault("state", None)
    provider_kwargs = _enrich_assignment_provider_context(agent_names, task_count, provider_kwargs)

    provider_payload = assignment_provider(
        agent_names=[str(agent) for agent in provider_kwargs["available_agent_names"]],
        task_count=max(int(task_count), 0),
        assignment_override=assignment_override or {},
        **provider_kwargs,
    )
    if provider_payload is None:
        return {}
    if not isinstance(provider_payload, dict):
        raise ValueError("assignment_provider 必须返回 dict 或 None。")
    return provider_payload


def _resolve_latest_run_id(model_root): # 获取最新模型
    if not os.path.isdir(model_root):
        return None
    candidates = [
        run_id
        for run_id in os.listdir(model_root)
        if os.path.isdir(os.path.join(model_root, run_id))
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]


def resolve_model_path(args):   # 解算模型
    if args.eval_model_path:
        return args.eval_model_path
    run_id = args.run_id
    if not run_id:
        run_id = _resolve_latest_run_id(args.model_root)
        if run_id is None:
            raise FileNotFoundError(f"未在 {args.model_root} 下找到可用训练记录。")
        print(f"未指定 run_id，自动使用最新训练记录 {run_id}")
    model_dir = os.path.join(args.model_root, run_id)
    return _resolve_model_path_from_dir(model_dir)


def _resolve_model_dir_from_model_path(model_path):
    model_dir = os.path.dirname(model_path)
    if os.path.basename(model_dir).lower() == "best_model":
        model_dir = os.path.dirname(model_dir)
    return model_dir


def _load_json_file(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def resolve_global_planner_model_path(args):
    explicit_model_path = str(getattr(args, "global_planner_model_path", "") or "").strip()
    if explicit_model_path:
        if not os.path.exists(explicit_model_path):
            raise FileNotFoundError(f"未找到全局规划模型文件: {explicit_model_path}")
        return explicit_model_path

    planner_root = str(getattr(args, "global_planner_model_root", DEFAULT_GLOBAL_PLANNER_MODEL_ROOT) or "").strip()
    planner_run_id = str(getattr(args, "global_planner_run_id", "") or "").strip()
    if planner_run_id:
        planner_dir = os.path.join(planner_root, planner_run_id)
        return _resolve_model_path_from_dir(planner_dir)

    latest_run_id = _resolve_latest_run_id(planner_root)
    if latest_run_id is None:
        raise FileNotFoundError(f"未在 {planner_root} 下找到可用的全局规划模型。")
    planner_dir = os.path.join(planner_root, latest_run_id)
    print(f"未指定全局规划 run_id，自动使用最新记录 {latest_run_id}")
    return _resolve_model_path_from_dir(planner_dir)


def _load_global_planner_config(model_path):
    model_dir = _resolve_model_dir_from_model_path(model_path)
    config_path = os.path.join(model_dir, "config.json")
    return _load_json_file(config_path)


def _build_global_planner_env_kwargs(common_env, args, planner_config):
    planner_agent_count = int(planner_config.get("n_agents", getattr(args, "n_agents", 1)))
    if planner_agent_count != int(getattr(args, "n_agents", planner_agent_count)):
        raise ValueError(
            f"全局规划模型训练时 n_agents={planner_agent_count}，"
            f"当前测试场景 n_agents={getattr(args, 'n_agents', planner_agent_count)}，两者不一致。"
        )

    return {
        "space_dim": (
            float(getattr(common_env, "X", planner_config.get("space_dim", [100.0, 100.0, 50.0])[0])),
            float(getattr(common_env, "Y", planner_config.get("space_dim", [100.0, 100.0, 50.0])[1])),
            float(getattr(common_env, "Z", planner_config.get("space_dim", [100.0, 100.0, 50.0])[2])),
        ),
        "n_agents": planner_agent_count,
        "max_steps": int(getattr(args, "global_planner_max_steps", planner_config.get("max_steps", 30))),
        "num_obstacles": max(int(len(getattr(common_env, "obstacles", []))), int(planner_config.get("num_obstacles", 3))),
        "waypoint_step_size": float(planner_config.get("waypoint_step_size", getattr(common_env, "waypoint_step_size", 3.0))),
        "lidar_rays": int(planner_config.get("lidar_rays", getattr(args, "lidar_rays", 24))),
        "lidar_range": float(planner_config.get("lidar_range", getattr(args, "lidar_range", 0.6) * 100.0)),
        "goal_threshold": float(planner_config.get("goal_threshold", getattr(common_env, "goal_threshold", 1.0))),
        "step_penalty": float(planner_config.get("step_penalty", -0.01)),
        "progress_reward_scale": float(planner_config.get("progress_reward_scale", 1.0)),
        "obstacle_collision_penalty": float(planner_config.get("obstacle_collision_penalty", 3.0)),
        "agent_collision_penalty": 0.0,
        "goal_reward": float(planner_config.get("goal_reward", 30.0)),
        "timeout_penalty_scale": float(planner_config.get("timeout_penalty_scale", 2.0)),
        "collision_radius": 0.0,
        "render_mode": None,
        "scenario_json": "",
        "terrain_enabled": bool(getattr(args, "terrain_enabled", planner_config.get("terrain_enabled", True))),
        "terrain_csv_path": str(getattr(args, "terrain_csv_path", planner_config.get("terrain_csv_path", DEFAULT_TERRAIN_CSV))),
        "terrain_grid_resolution": float(getattr(args, "terrain_grid_resolution", planner_config.get("terrain_grid_resolution", 1.0))),
        "terrain_origin": (0.0, 0.0),
        "terrain_min_clearance": float(planner_config.get("terrain_min_clearance", 0.0)),
        "terrain_z_scale": float(getattr(args, "terrain_csv_z_scale", planner_config.get("terrain_z_scale", 0.001))),
    }


def _append_global_nav_event(state, message):
    entry = str(message)
    state.setdefault("event_log", []).append(entry)
    print(f"[GlobalPlan] {entry}")


def _build_global_planner_context(args, common_env, state):
    if not bool(getattr(args, "use_global_planner", False)):
        return None
    existing_ctx = state.get("_global_planner_ctx")
    if existing_ctx is not None:
        return existing_ctx

    model_path = resolve_global_planner_model_path(args)
    planner_config = _load_global_planner_config(model_path)
    planner_env_kwargs = _build_global_planner_env_kwargs(common_env, args, planner_config)
    planner_env = GlobalPlannerWaypointEnv(**planner_env_kwargs)
    planner_model = PPO.load(model_path, device=getattr(args, "device", "cpu"))
    ctx = {
        "model_path": model_path,
        "config": planner_config,
        "env": planner_env,
        "model": planner_model,
        "waypoint_stride": max(int(getattr(args, "global_planner_waypoint_stride", 2)), 1),
        "fallback_limit": max(int(getattr(args, "global_fallback_max_steps", 5)), 1),
    }
    state["_global_planner_ctx"] = ctx
    _append_global_nav_event(state, f"加载全局规划模型: {model_path}")
    return ctx


def _reset_global_nav_state_for_agent(state, agent):
    state.setdefault("global_plan_routes", {})[agent] = []
    state.setdefault("global_plan_indices", {})[agent] = 0
    state.setdefault("global_plan_targets", {})[agent] = None
    state.setdefault("global_plan_status", {})[agent] = "idle"
    state.setdefault("global_plan_subgoals", {})[agent] = None
    state.setdefault("global_plan_smoothed_subgoals", {})[agent] = None
    state.setdefault("global_plan_completed", {})[agent] = False
    state.setdefault("global_detour_modes", {})[agent] = "track"
    state.setdefault("global_detour_targets", {})[agent] = None
    state.setdefault("global_detour_directions", {})[agent] = 0
    state.setdefault("global_detour_lock_steps", {})[agent] = 0
    state.setdefault("global_detour_eval_cache", {})[agent] = {
        "step": -1,
        "segment_blocked": False,
        "min_lidar": 1.0,
        "blocking_obstacle": None,
        "near_edge": False,
    }
    state.setdefault("global_plan_debug", {})[agent] = {
        "last_index": 0,
        "last_distance": None,
        "stagnation_steps": 0,
        "threshold_hold_steps": 0,
        "last_switch_step": None,
        "cooldown_until_step": None,
        "last_summary_step": None,
    }
    state.setdefault("apf_override_active", {})[agent] = False
    state.setdefault("apf_override_steps", {})[agent] = 0
    state.setdefault("apf_override_clear_steps", {})[agent] = 0
    state.setdefault("apf_override_last_obstacle", {})[agent] = None


def _ensure_global_nav_state(state, agent_names):
    for agent in [str(name) for name in agent_names]:
        _reset_global_nav_state_for_agent(state, agent)
    state.setdefault("direct_task_targets", {})
    state.setdefault("global_plan_last_targets", {})


def _get_active_agent_names(args, all_agent_names):
    all_agent_names = [str(agent) for agent in all_agent_names]
    requested = getattr(args, "active_n_agents", None)
    if requested is None or int(requested) <= 0:
        requested = len(all_agent_names)
    active_count = int(np.clip(int(requested), 0, len(all_agent_names)))
    return all_agent_names[:active_count]


def apply_active_agent_queue_mask(agent_names, state):
    active_agent_list = [str(agent) for agent in state.get("active_agents", [])]
    if not active_agent_list:
        active_agent_list = [str(agent) for agent in agent_names]
    active_agents = set(active_agent_list)
    inactive_agents = {str(agent) for agent in state.get("inactive_agents", set())}
    inactive_agents.update(str(agent) for agent in agent_names if str(agent) not in active_agents)
    state["inactive_agents"] = inactive_agents
    state["active_agents"] = list(active_agent_list)

    reassigned_tasks = []
    for agent in [str(agent) for agent in agent_names]:
        if agent in inactive_agents:
            reassigned_tasks.extend(state.setdefault("agent_queues", {}).get(agent, []))
            state.setdefault("agent_queues", {})[agent] = []

    if active_agent_list:
        existing_tasks = {
            str(task_id)
            for agent in active_agent_list
            for task_id in state.setdefault("agent_queues", {}).get(agent, [])
        }
        insert_index = 0
        for task_id in reassigned_tasks:
            task_id = str(task_id)
            if task_id in existing_tasks:
                continue
            target_agent = active_agent_list[insert_index % len(active_agent_list)]
            state.setdefault("agent_queues", {}).setdefault(target_agent, []).append(task_id)
            existing_tasks.add(task_id)
            insert_index += 1
    return state["agent_queues"]


def _apply_active_agent_mask(env, state, args):
    active_agent_list = _get_active_agent_names(args, env.possible_agents)
    state["active_agents"] = list(active_agent_list)
    state["inactive_agents"] = {
        agent for agent in env.possible_agents if agent not in set(active_agent_list)
    } | set(state.get("inactive_agents", set()))
    apply_active_agent_queue_mask(env.possible_agents, state)

    for agent in env.possible_agents:
        if agent not in state["inactive_agents"]:
            continue
        if agent in env.agent_positions:
            mark_idle_agent(env, agent)
            env.agent_targets[agent] = np.asarray(env.agent_positions[agent], dtype=np.float32).copy()
            env.agent_last_motion[agent] = np.zeros(3, dtype=np.float32)
            env.agent_reached_targets[agent] = True
            env.agent_dones[agent] = True

    sync_task_assignments_from_queues(state, agent_names=env.possible_agents)


def print_agent_task_sequences(state, title="当前无人机任务序列", agent_names=None, include_inactive=True):
    agent_queues = state.get("agent_queues", {})
    inactive_agents = {str(agent) for agent in state.get("inactive_agents", set())}
    if agent_names is None:
        agent_names = list(agent_queues.keys())
    print(title)
    for agent in [str(agent) for agent in agent_names]:
        if agent in inactive_agents:
            if include_inactive:
                print(f"  {agent}: inactive")
            continue
        queue = [str(task_id) for task_id in agent_queues.get(agent, [])]
        print(f"  {agent}: {' -> '.join(queue) if queue else 'idle'}")


def format_sim_time(step_completed, args_or_state):
    args = args_or_state.get("_runtime_args") if isinstance(args_or_state, dict) else args_or_state
    decision_period = float(getattr(args, "decision_period", 1.0)) if args is not None else 1.0
    return float(step_completed) * decision_period


def _get_cached_detour_environment_eval(env, state, agent, base_target, edge_clearance_threshold, force_refresh=False):
    cache = state.setdefault("global_detour_eval_cache", {}).setdefault(
        agent,
        {
            "step": -1,
            "segment_blocked": False,
            "min_lidar": 1.0,
            "blocking_obstacle": None,
            "near_edge": False,
        },
    )
    current_step = int(getattr(env, "step_cnt", 0))
    refresh_interval = 6
    if (
        not force_refresh
        and int(cache.get("step", -1)) >= 0
        and (current_step - int(cache.get("step", -1))) < refresh_interval
    ):
        return cache

    current_pos = np.asarray(env.agent_positions[agent], dtype=np.float32)
    lidar_values = env._cast_rays(current_pos).astype(np.float32)
    min_lidar = float(np.min(lidar_values)) if lidar_values.size > 0 else 1.0
    segment_blocked = not _segment_is_valid(env, current_pos, base_target)
    blocking_obstacle = _find_detour_obstacle(env, current_pos, base_target)
    near_edge = False
    if blocking_obstacle is not None:
        edge_distance = _distance_to_obstacle_edge_xy(blocking_obstacle, current_pos[:2])
        near_edge = edge_distance <= edge_clearance_threshold

    cache["step"] = current_step
    cache["segment_blocked"] = bool(segment_blocked)
    cache["min_lidar"] = float(min_lidar)
    cache["blocking_obstacle"] = blocking_obstacle
    cache["near_edge"] = bool(near_edge)
    return cache


def _update_global_nav_debug(state, agent, route_index, distance_to_subgoal, step_completed=None):
    debug_state = state.setdefault("global_plan_debug", {}).setdefault(
        agent,
        {
            "last_index": int(route_index),
            "last_distance": None,
            "stagnation_steps": 0,
            "threshold_hold_steps": 0,
            "last_switch_step": None,
            "cooldown_until_step": None,
            "last_summary_step": None,
        },
    )
    previous_index = int(debug_state.get("last_index", route_index))
    previous_distance = debug_state.get("last_distance", None)

    switched = int(route_index) != previous_index
    if switched:
        debug_state["stagnation_steps"] = 0
        debug_state["threshold_hold_steps"] = 0
        debug_state["last_switch_step"] = None if step_completed is None else int(step_completed)
        state.setdefault("global_detour_modes", {})[agent] = "track"
        state.setdefault("global_detour_targets", {})[agent] = None
        state.setdefault("global_detour_directions", {})[agent] = 0
        state.setdefault("global_detour_lock_steps", {})[agent] = 0
    else:
        if previous_distance is not None and distance_to_subgoal is not None:
            args = state.get("_runtime_args")
            progress_reset_threshold = (
                float(getattr(args, "global_waypoint_progress_reset_threshold", 0.02))
                if args is not None
                else 0.02
            )
            jitter_tolerance = (
                float(getattr(args, "global_waypoint_jitter_tolerance", 0.01))
                if args is not None
                else 0.01
            )
            distance_delta = float(previous_distance) - float(distance_to_subgoal)
            if abs(distance_delta) <= jitter_tolerance:
                debug_state["stagnation_steps"] = int(debug_state.get("stagnation_steps", 0)) + 1
            elif distance_delta < progress_reset_threshold:
                debug_state["stagnation_steps"] = int(debug_state.get("stagnation_steps", 0)) + 1
            else:
                debug_state["stagnation_steps"] = 0
        else:
            debug_state["stagnation_steps"] = 0

    debug_state["last_index"] = int(route_index)
    debug_state["last_distance"] = (
        None if distance_to_subgoal is None else float(distance_to_subgoal)
    )
    return debug_state, switched, previous_index


def _get_waypoint_switch_threshold(env, state):
    args = state.get("_runtime_args")
    explicit_threshold = getattr(args, "global_waypoint_switch_threshold", None) if args is not None else None
    if explicit_threshold is not None and float(explicit_threshold) > 0.0:
        return float(explicit_threshold)
    return max(float(env.goal_threshold), float(env.waypoint_step_size) * 0.75)


def _should_force_advance_waypoint(env, state, agent, distance_to_subgoal, debug_state):
    args = state.get("_runtime_args")
    stagnation_steps = int(debug_state.get("stagnation_steps", 0))
    previous_distance = debug_state.get("last_distance", None)
    if previous_distance is None:
        previous_distance = float(distance_to_subgoal)

    relaxed_threshold = max(
        _get_waypoint_switch_threshold(env, state),
        float(getattr(args, "global_waypoint_relaxed_threshold", 1.2)) if args is not None else 1.2,
    )
    stagnation_trigger = int(getattr(args, "global_waypoint_stagnation_steps", 24)) if args is not None else 24
    skip_trigger = int(getattr(args, "global_waypoint_skip_steps", 120)) if args is not None else 120
    distance_delta = float(previous_distance) - float(distance_to_subgoal)

    if float(distance_to_subgoal) <= relaxed_threshold and stagnation_steps >= stagnation_trigger:
        return True, "relaxed_stagnation"
    if float(distance_to_subgoal) <= relaxed_threshold and distance_delta < -1e-3:
        return True, "distance_rebound"
    if stagnation_steps >= skip_trigger:
        return True, "hard_skip"
    return False, None


def _update_smoothed_subgoal(state, agent, target_point):
    target_point = np.asarray(target_point, dtype=np.float32).copy()
    args = state.get("_runtime_args")
    alpha = float(getattr(args, "global_waypoint_smoothing_alpha", 0.25)) if args is not None else 0.25
    alpha = float(np.clip(alpha, 0.0, 1.0))

    smoothed_map = state.setdefault("global_plan_smoothed_subgoals", {})
    previous = smoothed_map.get(agent)
    if previous is None:
        smoothed_map[agent] = target_point.copy()
        return smoothed_map[agent]

    previous = np.asarray(previous, dtype=np.float32)
    smoothed = ((1.0 - alpha) * previous + alpha * target_point).astype(np.float32)
    smoothed_map[agent] = smoothed
    return smoothed


def _compute_segment_lookahead_target(env, state, agent):
    route = state.get("global_plan_routes", {}).get(agent, [])
    route_index = int(state.get("global_plan_indices", {}).get(agent, 0))
    direct_target = state.get("direct_task_targets", {}).get(agent)
    if direct_target is None:
        return None

    direct_target = np.asarray(direct_target, dtype=np.float32).copy()
    current_pos = np.asarray(env.agent_positions[agent], dtype=np.float32).copy()
    current_waypoint = state.get("global_plan_subgoals", {}).get(agent)
    if current_waypoint is None:
        current_waypoint = direct_target
    current_waypoint = np.asarray(current_waypoint, dtype=np.float32).copy()

    if not route or route_index >= len(route):
        return direct_target

    segment_end = current_waypoint
    if route_index <= 0:
        segment_start = current_pos.copy()
    else:
        segment_start = np.asarray(route[route_index - 1], dtype=np.float32).copy()

    segment = segment_end - segment_start
    segment_length = float(np.linalg.norm(segment))
    if segment_length <= 1e-6:
        return segment_end

    segment_dir = segment / segment_length
    projection_ratio = float(
        np.dot(current_pos - segment_start, segment) / max(segment_length * segment_length, 1e-6)
    )

    args = state.get("_runtime_args")
    lookahead_distance = (
        float(getattr(args, "global_segment_lookahead_distance", 1.5))
        if args is not None
        else 1.5
    )
    min_progress_ratio = (
        float(getattr(args, "global_segment_min_progress_ratio", 0.2))
        if args is not None
        else 0.2
    )
    min_progress_ratio = float(np.clip(min_progress_ratio, 0.0, 1.0))

    lookahead_ratio = float(lookahead_distance / max(segment_length, 1e-6))
    target_ratio = max(projection_ratio + lookahead_ratio, min_progress_ratio)
    target_ratio = float(np.clip(target_ratio, 0.0, 1.0))
    lookahead_target = segment_start + target_ratio * segment
    return lookahead_target.astype(np.float32)


def _distance_point_to_segment_xy(point_xy, seg_start_xy, seg_end_xy):
    point_xy = np.asarray(point_xy, dtype=np.float32)
    seg_start_xy = np.asarray(seg_start_xy, dtype=np.float32)
    seg_end_xy = np.asarray(seg_end_xy, dtype=np.float32)
    seg_vec = seg_end_xy - seg_start_xy
    seg_len_sq = float(np.dot(seg_vec, seg_vec))
    if seg_len_sq <= 1e-8:
        return float(np.linalg.norm(point_xy - seg_start_xy)), 0.0
    ratio = float(np.dot(point_xy - seg_start_xy, seg_vec) / seg_len_sq)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    projection = seg_start_xy + ratio * seg_vec
    return float(np.linalg.norm(point_xy - projection)), ratio


def _find_detour_obstacle(env, start, target):
    start = np.asarray(start, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    start_xy = start[:2]
    target_xy = target[:2]
    best_obstacle = None
    best_score = float("inf")
    for obstacle in getattr(env, "obstacles", []):
        ox = float(getattr(obstacle, "x", 0.0))
        oy = float(getattr(obstacle, "y", 0.0))
        radius = float(getattr(obstacle, "radius", 0.0))
        distance_to_segment, ratio = _distance_point_to_segment_xy(
            np.array([ox, oy], dtype=np.float32),
            start_xy,
            target_xy,
        )
        if ratio <= 1e-4:
            continue
        if distance_to_segment > radius + max(float(getattr(env, "waypoint_step_size", 1.0)) * 2.0, 2.0):
            continue
        score = ratio * 1000.0 + distance_to_segment
        if score < best_score:
            best_score = score
            best_obstacle = obstacle
    return best_obstacle


def _compute_detour_candidate(env, current_pos, target_pos, obstacle, direction):
    current_pos = np.asarray(current_pos, dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    obstacle_xy = np.array(
        [float(getattr(obstacle, "x", 0.0)), float(getattr(obstacle, "y", 0.0))],
        dtype=np.float32,
    )
    radial = current_pos[:2] - obstacle_xy
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= 1e-6:
        base_vec = target_pos[:2] - current_pos[:2]
        if float(np.linalg.norm(base_vec)) <= 1e-6:
            base_vec = np.array([1.0, 0.0], dtype=np.float32)
        radial = np.array([-base_vec[1], base_vec[0]], dtype=np.float32)
        radial_norm = float(np.linalg.norm(radial))
    radial_unit = radial / max(radial_norm, 1e-6)
    tangent = np.array([-radial_unit[1], radial_unit[0]], dtype=np.float32) * float(direction)

    radius = float(getattr(obstacle, "radius", 0.0))
    base_clearance = max(float(getattr(env, "waypoint_step_size", 1.0)) * 1.5, 2.0)
    tangent_advance = max(float(getattr(env, "waypoint_step_size", 1.0)) * 3.0, 4.0)
    candidate_xy = obstacle_xy + radial_unit * (radius + base_clearance) + tangent * tangent_advance

    candidate = np.array(
        [
            float(np.clip(candidate_xy[0], 0.0, float(env.X))),
            float(np.clip(candidate_xy[1], 0.0, float(env.Y))),
            float(np.clip(current_pos[2], 0.0, float(env.Z))),
        ],
        dtype=np.float32,
    )
    if hasattr(env, "_lift_position_to_free_height"):
        try:
            candidate = np.asarray(env._lift_position_to_free_height(candidate), dtype=np.float32)
        except Exception:
            pass
    return candidate


def _compute_escape_edge_candidate(env, current_pos, target_pos, obstacle, direction):
    current_pos = np.asarray(current_pos, dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    obstacle_xy = np.array(
        [float(getattr(obstacle, "x", 0.0)), float(getattr(obstacle, "y", 0.0))],
        dtype=np.float32,
    )
    radial = current_pos[:2] - obstacle_xy
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= 1e-6:
        target_vec = target_pos[:2] - current_pos[:2]
        if float(np.linalg.norm(target_vec)) <= 1e-6:
            target_vec = np.array([1.0, 0.0], dtype=np.float32)
        radial = -target_vec
        radial_norm = float(np.linalg.norm(radial))
    radial_unit = radial / max(radial_norm, 1e-6)
    tangent = np.array([-radial_unit[1], radial_unit[0]], dtype=np.float32) * float(direction)

    radius = float(getattr(obstacle, "radius", 0.0))
    outward_clearance = max(float(getattr(env, "waypoint_step_size", 1.0)) * 2.5, 3.5)
    tangent_bias = max(float(getattr(env, "waypoint_step_size", 1.0)) * 1.2, 1.5)
    candidate_xy = obstacle_xy + radial_unit * (radius + outward_clearance) + tangent * tangent_bias

    candidate = np.array(
        [
            float(np.clip(candidate_xy[0], 0.0, float(env.X))),
            float(np.clip(candidate_xy[1], 0.0, float(env.Y))),
            float(np.clip(current_pos[2], 0.0, float(env.Z))),
        ],
        dtype=np.float32,
    )
    if hasattr(env, "_lift_position_to_free_height"):
        try:
            candidate = np.asarray(env._lift_position_to_free_height(candidate), dtype=np.float32)
        except Exception:
            pass
    return candidate


def _distance_to_obstacle_edge_xy(obstacle, point_xy):
    point_xy = np.asarray(point_xy, dtype=np.float32)
    ox = float(getattr(obstacle, "x", 0.0))
    oy = float(getattr(obstacle, "y", 0.0))
    radius = float(getattr(obstacle, "radius", 0.0))
    return float(np.hypot(point_xy[0] - ox, point_xy[1] - oy) - radius)


def _find_nearest_obstacle_for_apf(env, position):
    position = np.asarray(position, dtype=np.float32)
    best_obstacle = None
    best_edge_distance = float("inf")
    for obstacle in getattr(env, "obstacles", []):
        obstacle_top = float(getattr(obstacle, "height", getattr(obstacle, "z", 0.0)))
        if position[2] > obstacle_top + max(float(getattr(env, "waypoint_step_size", 1.0)), 1.0):
            continue
        edge_distance = _distance_to_obstacle_edge_xy(obstacle, position[:2])
        if edge_distance < best_edge_distance:
            best_edge_distance = edge_distance
            best_obstacle = obstacle
    return best_obstacle, float(best_edge_distance)


def _should_activate_apf_override(env, state, agent, guidance_target):
    active_map = state.setdefault("apf_override_active", {})
    if bool(active_map.get(agent, False)):
        return True

    debug_state = state.setdefault("global_plan_debug", {}).setdefault(agent, {})
    stagnation_steps = int(debug_state.get("stagnation_steps", 0))
    current_pos = np.asarray(env.agent_positions[agent], dtype=np.float32)
    nearest_obstacle, edge_distance = _find_nearest_obstacle_for_apf(env, current_pos)
    if nearest_obstacle is None:
        return False

    lidar_values = env._cast_rays(current_pos).astype(np.float32)
    min_lidar = float(np.min(lidar_values)) if lidar_values.size > 0 else 1.0
    segment_blocked = not _segment_is_valid(env, current_pos, guidance_target)

    trigger_stagnation = stagnation_steps >= 12
    trigger_near_obstacle = edge_distance <= max(float(getattr(env, "waypoint_step_size", 1.0)) * 0.9, 1.2)
    trigger_lidar = min_lidar <= 0.16

    if segment_blocked and trigger_near_obstacle and (trigger_stagnation or trigger_lidar):
        state.setdefault("apf_override_last_obstacle", {})[agent] = nearest_obstacle
        return True
    return False


def _compute_apf_override_action(env, state, agent, guidance_target, previous_action):
    current_pos = np.asarray(env.agent_positions[agent], dtype=np.float32)
    guidance_target = np.asarray(guidance_target, dtype=np.float32)
    goal_vec = guidance_target - current_pos

    attractive = np.zeros(3, dtype=np.float32)
    goal_norm = float(np.linalg.norm(goal_vec))
    if goal_norm > 1e-6:
        attractive = goal_vec / goal_norm

    lidar_range = max(float(getattr(env, "lidar_range", 0.0)), 1e-6)
    influence_distance = min(
        max(float(getattr(env, "waypoint_step_size", 1.0)) * 3.0, 6.0),
        lidar_range,
    )
    repulsive_xy = np.zeros(2, dtype=np.float32)
    closest_obstacle = None
    closest_edge_distance = float("inf")

    for obstacle in getattr(env, "obstacles", []):
        center_xy = np.array(
            [float(getattr(obstacle, "x", 0.0)), float(getattr(obstacle, "y", 0.0))],
            dtype=np.float32,
        )
        away_xy = current_pos[:2] - center_xy
        away_norm = float(np.linalg.norm(away_xy))
        if away_norm <= 1e-6:
            away_xy = np.array([1.0, 0.0], dtype=np.float32)
            away_norm = 1.0
        edge_distance = float(away_norm - float(getattr(obstacle, "radius", 0.0)))
        if edge_distance > influence_distance:
            continue
        if edge_distance < closest_edge_distance:
            closest_edge_distance = edge_distance
            closest_obstacle = obstacle
        away_unit = away_xy / away_norm
        safe_distance = max(edge_distance, 0.2)
        gain = (1.0 / safe_distance) - (1.0 / influence_distance)
        if gain <= 0.0:
            continue
        repulsive_xy += away_unit * float(gain / max(safe_distance, 0.2))

    tangential_xy = np.zeros(2, dtype=np.float32)
    if closest_obstacle is not None:
        center_xy = np.array(
            [float(getattr(closest_obstacle, "x", 0.0)), float(getattr(closest_obstacle, "y", 0.0))],
            dtype=np.float32,
        )
        away_xy = current_pos[:2] - center_xy
        away_norm = float(np.linalg.norm(away_xy))
        if away_norm <= 1e-6:
            away_xy = np.array([1.0, 0.0], dtype=np.float32)
            away_norm = 1.0
        away_unit = away_xy / away_norm
        tangent_left = np.array([-away_unit[1], away_unit[0]], dtype=np.float32)
        tangent_right = -tangent_left

        goal_xy = goal_vec[:2]
        prev_xy = np.asarray(previous_action, dtype=np.float32)[:2]
        if float(np.linalg.norm(prev_xy)) > 1e-6:
            tangent_sign = 1.0 if float(np.dot(prev_xy, tangent_left)) >= float(np.dot(prev_xy, tangent_right)) else -1.0
        else:
            tangent_sign = 1.0 if float(np.dot(goal_xy, tangent_left)) >= float(np.dot(goal_xy, tangent_right)) else -1.0
        tangential_xy = tangent_left if tangent_sign > 0.0 else tangent_right
        state.setdefault("apf_override_last_obstacle", {})[agent] = closest_obstacle

    apf_vec = np.zeros(3, dtype=np.float32)
    apf_vec[:2] = (
        0.55 * attractive[:2]
        + 1.35 * repulsive_xy
        + 0.75 * tangential_xy
    )
    apf_vec[2] = 0.35 * attractive[2]

    apf_norm = float(np.linalg.norm(apf_vec))
    if apf_norm <= 1e-6:
        return np.zeros(3, dtype=np.float32)
    return np.clip(apf_vec / max(apf_norm, 1e-6), -1.0, 1.0).astype(np.float32)


def _update_apf_override_state(env, state, agent, guidance_target):
    active_map = state.setdefault("apf_override_active", {})
    step_map = state.setdefault("apf_override_steps", {})
    clear_map = state.setdefault("apf_override_clear_steps", {})
    obstacle_map = state.setdefault("apf_override_last_obstacle", {})

    if not bool(active_map.get(agent, False)):
        return False

    current_pos = np.asarray(env.agent_positions[agent], dtype=np.float32)
    guidance_target = np.asarray(guidance_target, dtype=np.float32)
    nearest_obstacle = obstacle_map.get(agent)
    if nearest_obstacle is None:
        nearest_obstacle, edge_distance = _find_nearest_obstacle_for_apf(env, current_pos)
    else:
        edge_distance = _distance_to_obstacle_edge_xy(nearest_obstacle, current_pos[:2])

    lidar_values = env._cast_rays(current_pos).astype(np.float32)
    min_lidar = float(np.min(lidar_values)) if lidar_values.size > 0 else 1.0
    segment_clear = _segment_is_valid(env, current_pos, guidance_target)
    debug_state = state.setdefault("global_plan_debug", {}).setdefault(agent, {})
    stagnation_steps = int(debug_state.get("stagnation_steps", 0))

    if (
        segment_clear
        and min_lidar >= 0.28
        and edge_distance >= max(float(getattr(env, "waypoint_step_size", 1.0)) * 1.8, 2.5)
        and stagnation_steps <= 2
    ):
        clear_map[agent] = int(clear_map.get(agent, 0)) + 1
    else:
        clear_map[agent] = 0

    step_map[agent] = int(step_map.get(agent, 0)) + 1
    if int(clear_map.get(agent, 0)) >= 4:
        active_map[agent] = False
        step_map[agent] = 0
        clear_map[agent] = 0
        obstacle_map[agent] = None
        return False
    return True


def _select_detour_target(env, state, agent, base_target):
    current_pos = np.asarray(env.agent_positions[agent], dtype=np.float32)
    obstacle = _find_detour_obstacle(env, current_pos, base_target)
    if obstacle is None:
        return None, 0

    best_target = None
    best_direction = 0
    best_score = float("inf")
    for direction in (-1, 1):
        candidate = _compute_detour_candidate(env, current_pos, base_target, obstacle, direction)
        if not _point_is_valid(env, candidate):
            continue
        if not _segment_is_valid(env, current_pos, candidate):
            continue
        score = _score_fallback_candidate(env, agent, candidate, base_target)
        if score < best_score:
            best_score = score
            best_target = candidate.copy()
            best_direction = int(direction)
    return best_target, best_direction


def _select_escape_edge_target(env, state, agent, base_target, direction_hint=0):
    current_pos = np.asarray(env.agent_positions[agent], dtype=np.float32)
    obstacle = _find_detour_obstacle(env, current_pos, base_target)
    if obstacle is None:
        return None

    direction_candidates = []
    if int(direction_hint) != 0:
        direction_candidates.append(int(direction_hint))
    for direction in (-1, 1):
        if direction not in direction_candidates:
            direction_candidates.append(direction)

    best_target = None
    best_score = float("inf")
    for direction in direction_candidates:
        candidate = _compute_escape_edge_candidate(env, current_pos, base_target, obstacle, direction)
        if not _point_is_valid(env, candidate):
            continue
        if not _segment_is_valid(env, current_pos, candidate):
            continue
        score = _score_fallback_candidate(env, agent, candidate, base_target)
        if score < best_score:
            best_score = score
            best_target = candidate.copy()
    return best_target


def _get_detour_target(env, state, agent, base_target):
    base_target = np.asarray(base_target, dtype=np.float32).copy()
    current_pos = np.asarray(env.agent_positions[agent], dtype=np.float32).copy()
    debug_state = state.setdefault("global_plan_debug", {}).setdefault(agent, {})
    detour_modes = state.setdefault("global_detour_modes", {})
    detour_targets = state.setdefault("global_detour_targets", {})
    detour_directions = state.setdefault("global_detour_directions", {})
    detour_lock_steps = state.setdefault("global_detour_lock_steps", {})

    mode = str(detour_modes.get(agent, "track"))
    lock_steps = int(detour_lock_steps.get(agent, 0))
    args = state.get("_runtime_args")
    stagnation_trigger = int(getattr(args, "global_detour_stagnation_steps", 18)) if args is not None else 18
    lock_trigger = int(getattr(args, "global_detour_lock_steps", 24)) if args is not None else 24
    lidar_threshold = float(getattr(args, "global_detour_lidar_threshold", 0.22)) if args is not None else 0.22
    edge_stagnation_trigger = int(getattr(args, "global_detour_edge_stagnation_steps", 14)) if args is not None else 14
    edge_clearance_threshold = float(getattr(args, "global_detour_edge_clearance", 1.5)) if args is not None else 1.5

    stagnation_steps = int(debug_state.get("stagnation_steps", 0))

    if mode != "track":
        current_detour_target = detour_targets.get(agent)
        if current_detour_target is not None:
            current_detour_target = np.asarray(current_detour_target, dtype=np.float32)
        if (
            current_detour_target is not None
            and _point_is_valid(env, current_detour_target)
            and np.linalg.norm(current_detour_target - current_pos) > max(float(getattr(env, "waypoint_step_size", 1.0)), 1.0)
            and lock_steps > 0
        ):
            detour_lock_steps[agent] = max(lock_steps - 1, 0)
            return current_detour_target.copy()

        detour_eval = _get_cached_detour_environment_eval(
            env,
            state,
            agent,
            base_target,
            edge_clearance_threshold=edge_clearance_threshold,
            force_refresh=True,
        )
        segment_blocked = bool(detour_eval.get("segment_blocked", False))
        near_edge = bool(detour_eval.get("near_edge", False))
        if _segment_is_valid(env, current_pos, base_target) and stagnation_steps <= 2:
            detour_modes[agent] = "track"
            detour_targets[agent] = None
            detour_directions[agent] = 0
            detour_lock_steps[agent] = 0
            return base_target
        if mode in {"detour_left", "detour_right"} and near_edge and stagnation_steps >= edge_stagnation_trigger:
            escape_target = _select_escape_edge_target(
                env,
                state,
                agent,
                base_target,
                direction_hint=int(detour_directions.get(agent, 0)),
            )
            if escape_target is not None:
                detour_modes[agent] = "escape_edge"
                detour_targets[agent] = escape_target.copy()
                detour_lock_steps[agent] = max(lock_steps, 8)
                return escape_target.copy()
        if current_detour_target is not None and _point_is_valid(env, current_detour_target):
            if np.linalg.norm(current_detour_target - current_pos) <= max(float(getattr(env, "waypoint_step_size", 1.0)), 1.0):
                detour_modes[agent] = "track"
                detour_targets[agent] = None
                detour_directions[agent] = 0
                detour_lock_steps[agent] = 0
                return base_target
            if mode == "escape_edge" and not near_edge and _segment_is_valid(env, current_pos, base_target):
                detour_modes[agent] = "track"
                detour_targets[agent] = None
                detour_directions[agent] = 0
                detour_lock_steps[agent] = 0
                return base_target
            detour_lock_steps[agent] = max(lock_steps - 1, 0)
            return current_detour_target.copy()
        detour_modes[agent] = "track"
        detour_targets[agent] = None
        detour_directions[agent] = 0
        detour_lock_steps[agent] = 0

    if stagnation_steps < stagnation_trigger:
        return base_target

    detour_eval = _get_cached_detour_environment_eval(
        env,
        state,
        agent,
        base_target,
        edge_clearance_threshold=edge_clearance_threshold,
        force_refresh=False,
    )
    segment_blocked = bool(detour_eval.get("segment_blocked", False))
    min_lidar = float(detour_eval.get("min_lidar", 1.0))
    if not (segment_blocked and min_lidar < lidar_threshold):
        return base_target

    detour_target, detour_direction = _select_detour_target(env, state, agent, base_target)
    if detour_target is None:
        return base_target

    detour_modes[agent] = "detour_right" if detour_direction > 0 else "detour_left"
    detour_targets[agent] = detour_target.copy()
    detour_directions[agent] = int(detour_direction)
    detour_lock_steps[agent] = int(lock_trigger)
    return detour_target.copy()


def _maybe_log_global_nav_debug(state, agent, debug_state, step_completed=None, reason="summary"):
    # 调试阶段使用的 waypoint 诊断输出，默认静音保留逻辑。
    return


def _prepare_planner_env(planner_env, common_env, base_targets, active_agents=None):
    active_agents = {str(agent) for agent in (active_agents or base_targets.keys())}
    planner_env.reset(seed=None)
    planner_env.obstacles = copy.deepcopy(list(getattr(common_env, "obstacles", [])))
    planner_env.all_obstacles = copy.deepcopy(list(getattr(common_env, "obstacles", [])))
    planner_env.agents = planner_env.possible_agents.copy()
    planner_env.step_cnt = 0
    for agent in planner_env.possible_agents:
        start = np.asarray(common_env.agent_positions[agent], dtype=np.float32).copy()
        target = np.asarray(base_targets.get(agent, start), dtype=np.float32).copy()
        planner_env.agent_positions[agent] = start
        planner_env.agent_last_motion[agent] = np.zeros(3, dtype=np.float32)
        planner_env.agent_targets[agent] = target
        planner_env.agent_reached_targets[agent] = agent not in active_agents
        planner_env.agent_dones[agent] = agent not in active_agents
        planner_env.agent_obstacle_collisions[agent] = 0
        planner_env.agent_inter_agent_collisions[agent] = 0
    obs = {agent: planner_env.observe(agent) for agent in planner_env.agents}
    return obs


def _deduplicate_route_points(points, tolerance=1e-5):
    deduped = []
    for point in points:
        point = np.asarray(point, dtype=np.float32)
        if point.shape[0] < 3:
            continue
        if not deduped or np.linalg.norm(point - deduped[-1]) > float(tolerance):
            deduped.append(point.copy())
    return deduped


def _select_route_bridge_fallback(env, agent, current, target_point, previous_point=None):
    current = np.asarray(current, dtype=np.float32)
    target_point = np.asarray(target_point, dtype=np.float32)
    previous_motion = np.zeros(3, dtype=np.float32)
    if previous_point is not None:
        previous_point = np.asarray(previous_point, dtype=np.float32)
        if previous_point.shape[0] >= 3:
            previous_motion = (current - previous_point).astype(np.float32)

    original_position = np.asarray(env.agent_positions.get(agent, current), dtype=np.float32).copy()
    original_motion = np.asarray(
        env.agent_last_motion.get(agent, np.zeros(3, dtype=np.float32)),
        dtype=np.float32,
    ).copy()
    original_target = np.asarray(env.agent_targets.get(agent, target_point), dtype=np.float32).copy()

    env.agent_positions[agent] = current.copy()
    env.agent_last_motion[agent] = previous_motion.copy()
    env.agent_targets[agent] = target_point.copy()
    try:
        candidate = _select_fallback_subgoal(env, agent, target_point)
    finally:
        env.agent_positions[agent] = original_position
        env.agent_last_motion[agent] = original_motion
        env.agent_targets[agent] = original_target
    return None if candidate is None else np.asarray(candidate, dtype=np.float32).copy()


def _append_safe_route_segment(env, agent, route, history_points, target_point):
    target_point = np.asarray(target_point, dtype=np.float32)
    if target_point.shape[0] < 3:
        return route
    if not _point_is_valid(env, target_point):
        return route

    if not route:
        route.append(target_point.copy())
        return route

    current = np.asarray(route[-1], dtype=np.float32)
    if np.linalg.norm(target_point - current) <= 1e-5:
        return route
    if _segment_is_valid(env, current, target_point):
        route.append(target_point.copy())
        return route

    history_points = _deduplicate_route_points(history_points)
    candidate_idx = None
    for idx, point in enumerate(history_points):
        if np.linalg.norm(np.asarray(point, dtype=np.float32) - target_point) <= 1e-5:
            candidate_idx = idx
            break
    if candidate_idx is None:
        return route

    start_idx = 0
    for idx, point in enumerate(history_points):
        if np.linalg.norm(np.asarray(point, dtype=np.float32) - current) <= 1e-5:
            start_idx = idx
            break

    probe_idx = candidate_idx
    while probe_idx > start_idx:
        bridge_point = np.asarray(history_points[probe_idx], dtype=np.float32)
        if _segment_is_valid(env, current, bridge_point):
            if np.linalg.norm(bridge_point - current) > 1e-5:
                route.append(bridge_point.copy())
            current = bridge_point
            if _segment_is_valid(env, current, target_point):
                if np.linalg.norm(target_point - current) > 1e-5:
                    route.append(target_point.copy())
            return route
        probe_idx -= 1

    previous_point = route[-2] if len(route) > 1 else None
    fallback_bridge = _select_route_bridge_fallback(
        env,
        agent,
        current=current,
        target_point=target_point,
        previous_point=previous_point,
    )
    if fallback_bridge is not None and _point_is_valid(env, fallback_bridge):
        if _segment_is_valid(env, current, fallback_bridge):
            if np.linalg.norm(fallback_bridge - current) > 1e-5:
                route.append(fallback_bridge.copy())
            current = np.asarray(route[-1], dtype=np.float32)
            if _segment_is_valid(env, current, target_point):
                if np.linalg.norm(target_point - current) > 1e-5:
                    route.append(target_point.copy())
    return route


def _compress_global_route(points, final_target, stride, env, agent):
    if not points:
        return []

    history_points = _deduplicate_route_points(points)
    if not history_points:
        return []

    stride = max(int(stride), 1)
    sampled = history_points[1::stride] if len(history_points) > 1 else []
    if len(history_points) > 1:
        should_append_last = True
        if sampled:
            should_append_last = np.linalg.norm(history_points[-1] - sampled[-1]) > 1e-5
        if should_append_last:
            sampled.append(history_points[-1].copy())

    route = [history_points[0].copy()]
    for point in sampled:
        _append_safe_route_segment(env, agent, route, history_points, point)

    final_target = np.asarray(final_target, dtype=np.float32)
    if _point_is_valid(env, final_target):
        _append_safe_route_segment(env, agent, route, history_points, final_target)

    if route and np.linalg.norm(route[0] - history_points[0]) <= 1e-5:
        route = route[1:]
    return _deduplicate_route_points(route)


def _plan_global_routes(env, state):
    ctx = state.get("_global_planner_ctx")
    if ctx is None:
        return
    inactive_agents = {str(agent) for agent in state.get("inactive_agents", set())}
    active_agents = [
        str(agent)
        for agent in state.get("active_agents", env.possible_agents)
        if str(agent) not in inactive_agents
    ]
    active_agent_set = set(active_agents)
    base_targets = {
        str(agent): np.asarray(target, dtype=np.float32).copy()
        for agent, target in state.get("direct_task_targets", {}).items()
        if str(agent) in active_agent_set
    }
    if not base_targets:
        return

    planner_env = ctx["env"]
    planner_model = ctx["model"]
    obs = _prepare_planner_env(planner_env, env, base_targets, active_agents=active_agent_set)
    _append_global_nav_event(
        state,
        f"开始全局预规划，active agent 数={len(base_targets)} / model slots={len(planner_env.possible_agents)}",
    )
    history = {
        agent: [np.asarray(planner_env.agent_positions[agent], dtype=np.float32).copy()]
        for agent in planner_env.possible_agents
        if agent in active_agent_set
    }
    planner_policy_time_total = 0.0
    planner_policy_decision_count = 0

    for _ in range(planner_env.max_steps):
        actions = {}
        for agent in planner_env.agents:
            if agent not in active_agent_set:
                actions[agent] = np.zeros(3, dtype=np.float32)
                continue
            predict_started_at = time.perf_counter()
            raw_action, _ = planner_model.predict(obs[agent], deterministic=True)
            planner_policy_time_total += time.perf_counter() - predict_started_at
            planner_policy_decision_count += 1
            actions[agent] = np.asarray(raw_action, dtype=np.float32)
        obs, _, terminations, truncations, _ = planner_env.step(actions)
        for agent in active_agent_set:
            if agent in planner_env.agent_positions:
                history[agent].append(np.asarray(planner_env.agent_positions[agent], dtype=np.float32).copy())
        if all(terminations.get(agent, False) or truncations.get(agent, False) for agent in active_agent_set):
            break

    for agent in inactive_agents:
        _reset_global_nav_state_for_agent(state, agent)

    for agent, final_target in base_targets.items():
        route = _compress_global_route(
            history.get(agent, []),
            final_target,
            ctx["waypoint_stride"],
            planner_env,
            agent,
        )
        state["global_plan_routes"][agent] = route
        state["global_plan_indices"][agent] = 0
        state["global_plan_targets"][agent] = np.asarray(final_target, dtype=np.float32).copy()
        state["global_plan_completed"][agent] = False
        if agent in state.get("global_plan_debug", {}):
            state["global_plan_debug"][agent]["last_index"] = 0
            state["global_plan_debug"][agent]["last_distance"] = None
            state["global_plan_debug"][agent]["stagnation_steps"] = 0
            state["global_plan_debug"][agent]["threshold_hold_steps"] = 0
            state["global_plan_debug"][agent]["last_switch_step"] = None
            state["global_plan_debug"][agent]["cooldown_until_step"] = None
            state["global_plan_debug"][agent]["last_summary_step"] = None
        if route:
            state["global_plan_status"][agent] = "route"
            state["global_plan_subgoals"][agent] = np.asarray(route[0], dtype=np.float32).copy()
            state["global_plan_smoothed_subgoals"][agent] = np.asarray(route[0], dtype=np.float32).copy()
        else:
            state["global_plan_status"][agent] = "direct"
            state["global_plan_subgoals"][agent] = np.asarray(final_target, dtype=np.float32).copy()
            state["global_plan_smoothed_subgoals"][agent] = np.asarray(final_target, dtype=np.float32).copy()
    state["global_plan_last_targets"] = {
        agent: np.asarray(target, dtype=np.float32).copy()
        for agent, target in base_targets.items()
    }
    planner_policy_time_total_ms = 1000.0 * float(planner_policy_time_total)
    planner_policy_avg_ms = (
        planner_policy_time_total_ms / float(planner_policy_decision_count)
        if planner_policy_decision_count > 0
        else 0.0
    )
    state["global_plan_policy_time_total_ms"] = planner_policy_time_total_ms
    state["global_plan_policy_time_avg_ms_per_decision"] = planner_policy_avg_ms
    state["global_plan_policy_decision_count"] = int(planner_policy_decision_count)
    _append_global_nav_event(
        state,
        (
            "全局规划策略决策耗时: "
            f"total={planner_policy_time_total_ms:.3f} ms, "
            f"avg={planner_policy_avg_ms:.3f} ms/decision, "
            f"count={planner_policy_decision_count}"
        ),
    )
    _append_global_nav_event(state, "全局预规划完成")


def _point_is_valid(env, point):
    point = np.asarray(point, dtype=np.float32)
    return not env._is_collision(float(point[0]), float(point[1]), float(point[2]))


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
    previous_ratio = 0.0
    previous_collided = bool(env._is_collision(float(start[0]), float(start[1]), float(start[2])))
    if previous_collided:
        return True

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
        previous_collided = collided
    return False


def _segment_is_valid(env, start, end):
    return not _binary_segment_collision_check(
        env,
        start,
        end,
        coarse_step=max(float(getattr(env, "waypoint_step_size", 1.0)) * 0.5, 1e-3),
        binary_iters=12,
    )


def _candidate_agent_clearance(env, candidate, ignore_agent=None):
    candidate = np.asarray(candidate, dtype=np.float32)
    min_dist = float("inf")
    for agent, pos in env.agent_positions.items():
        if agent == ignore_agent:
            continue
        min_dist = min(min_dist, float(np.linalg.norm(candidate - np.asarray(pos, dtype=np.float32))))
    return min_dist if np.isfinite(min_dist) else env.workspace_diagonal


def _candidate_clearance_score(env, candidate, ignore_agent=None):
    candidate = np.asarray(candidate, dtype=np.float32)
    terrain_clearance = float(candidate[2] - env._terrain_surface_height(candidate[0], candidate[1]))
    obstacle_clearance = float("inf")
    for obstacle in getattr(env, "obstacles", []):
        ox = float(getattr(obstacle, "x", 0.0))
        oy = float(getattr(obstacle, "y", 0.0))
        oz = float(getattr(obstacle, "z", 0.0))
        radius = float(getattr(obstacle, "radius", 0.0))
        height = float(getattr(obstacle, "height", 0.0))
        if getattr(obstacle, "kind", "") == "cylinder":
            horizontal = np.hypot(candidate[0] - ox, candidate[1] - oy) - radius
            vertical = 0.0 if 0.0 <= candidate[2] <= height else min(abs(candidate[2]), abs(candidate[2] - height))
            obstacle_clearance = min(obstacle_clearance, float(max(horizontal, vertical)))
        else:
            obstacle_clearance = min(
                obstacle_clearance,
                float(np.linalg.norm(candidate - np.array([ox, oy, oz], dtype=np.float32)) - max(radius, height)),
            )
    agent_clearance = _candidate_agent_clearance(env, candidate, ignore_agent=ignore_agent) - float(getattr(env, "collision_radius", 0.0))
    min_clearance = min(terrain_clearance, obstacle_clearance, agent_clearance)
    scale = max(float(getattr(env, "waypoint_step_size", 1.0)) * 2.0, 1e-6)
    return 1.0 - float(np.clip(min_clearance / scale, 0.0, 1.0))


def _sample_fallback_candidates(env, current, primary_target):
    current = np.asarray(current, dtype=np.float32)
    primary_target = np.asarray(primary_target, dtype=np.float32)
    delta = primary_target - current
    base_heading = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta[:2]) > 1e-8 else 0.0
    radii = [
        float(getattr(env, "waypoint_step_size", 1.0) * 0.75),
        float(getattr(env, "waypoint_step_size", 1.0) * 1.25),
    ]
    vertical_bias = 0.0
    if current[2] > 0.75 * float(env.Z):
        vertical_bias = -0.25 * float(getattr(env, "waypoint_step_size", 1.0))
    elif current[2] - env._terrain_surface_height(current[0], current[1]) < 0.5 * float(getattr(env, "waypoint_step_size", 1.0)):
        vertical_bias = 0.25 * float(getattr(env, "waypoint_step_size", 1.0))

    candidates = []
    for radius in radii:
        for offset_deg in range(0, 360, 45):
            theta = base_heading + np.deg2rad(offset_deg)
            candidate = current.copy()
            candidate[0] += radius * np.cos(theta)
            candidate[1] += radius * np.sin(theta)
            candidate[2] += vertical_bias
            candidate = np.clip(candidate, [0.0, 0.0, 0.0], [env.X, env.Y, env.Z]).astype(np.float32)
            candidates.append(candidate)
    candidates.append(np.clip(primary_target.astype(np.float32), [0.0, 0.0, 0.0], [env.X, env.Y, env.Z]).astype(np.float32))
    return candidates


def _score_fallback_candidate(env, agent, candidate, primary_target):
    candidate = np.asarray(candidate, dtype=np.float32)
    primary_target = np.asarray(primary_target, dtype=np.float32)
    workspace = max(float(getattr(env, "workspace_diagonal", 1.0)), 1e-6)
    goal_cost = float(np.linalg.norm(candidate - primary_target) / workspace)
    risk_cost = _candidate_clearance_score(env, candidate, ignore_agent=agent)
    last_motion = np.asarray(env.agent_last_motion.get(agent, np.zeros(3, dtype=np.float32)), dtype=np.float32)
    move = candidate - np.asarray(env.agent_positions[agent], dtype=np.float32)
    if np.linalg.norm(last_motion) > 1e-8 and np.linalg.norm(move) > 1e-8:
        cos_value = float(np.clip(np.dot(last_motion, move) / (np.linalg.norm(last_motion) * np.linalg.norm(move)), -1.0, 1.0))
        turn_cost = float(np.arccos(cos_value) / np.pi)
    else:
        turn_cost = 0.0
    altitude_limit = 0.75 * float(env.Z)
    altitude_cost = float(max(candidate[2] - altitude_limit, 0.0) / max(float(env.Z), 1e-6))
    return goal_cost + risk_cost + 0.3 * turn_cost + 0.2 * altitude_cost


def _select_fallback_subgoal(env, agent, primary_target):
    current = np.asarray(env.agent_positions[agent], dtype=np.float32)
    best_candidate = None
    best_score = float("inf")
    for candidate in _sample_fallback_candidates(env, current, primary_target):
        if not _point_is_valid(env, candidate):
            continue
        if not _segment_is_valid(env, current, candidate):
            continue
        if _candidate_agent_clearance(env, candidate, ignore_agent=agent) <= float(getattr(env, "collision_radius", 0.0)) * 1.5:
            continue
        score = _score_fallback_candidate(env, agent, candidate, primary_target)
        if score < best_score:
            best_score = score
            best_candidate = candidate.astype(np.float32).copy()
    return best_candidate


def _targets_changed_for_replan(state, direct_targets):
    inactive_agents = {str(agent) for agent in state.get("inactive_agents", set())}
    active_agents = {
        str(agent)
        for agent in state.get("active_agents", direct_targets.keys())
        if str(agent) not in inactive_agents
    }
    direct_targets = {
        str(agent): np.asarray(target, dtype=np.float32)
        for agent, target in direct_targets.items()
        if str(agent) in active_agents
    }
    last_targets = state.get("global_plan_last_targets", {})
    last_targets = {
        str(agent): np.asarray(target, dtype=np.float32)
        for agent, target in last_targets.items()
        if str(agent) in active_agents
    }
    if set(last_targets.keys()) != set(direct_targets.keys()):
        return True
    for agent, target in direct_targets.items():
        previous = last_targets.get(agent)
        if previous is None:
            return True
        if np.linalg.norm(np.asarray(previous, dtype=np.float32) - np.asarray(target, dtype=np.float32)) > 1e-6:
            return True
    return False


def _sync_navigation_targets(env, state, step_completed=None):
    if not bool(state.get("global_planner_enabled", False)):
        return
    if state.get("_global_planner_ctx") is None:
        return

    for agent in env.possible_agents:
        direct_target = state.get("direct_task_targets", {}).get(agent)
        if direct_target is None:
            continue
        direct_target = np.asarray(direct_target, dtype=np.float32).copy()
        route = state.get("global_plan_routes", {}).get(agent, [])
        route_index = int(state.get("global_plan_indices", {}).get(agent, 0))
        debug_state = state.setdefault("global_plan_debug", {}).setdefault(
            agent,
            {
                "last_index": route_index,
                "last_distance": None,
                "stagnation_steps": 0,
                "threshold_hold_steps": 0,
                "last_switch_step": None,
                "cooldown_until_step": None,
                "last_summary_step": None,
            },
        )
        switch_threshold = _get_waypoint_switch_threshold(env, state)
        args = state.get("_runtime_args")
        hold_trigger = int(getattr(args, "global_waypoint_hold_steps", 3)) if args is not None else 3
        hold_trigger = max(hold_trigger, 1)
        cooldown_steps = int(getattr(args, "global_waypoint_cooldown_steps", 4)) if args is not None else 4
        cooldown_steps = max(cooldown_steps, 0)
        current_step = None if step_completed is None else int(step_completed)

        if not route:
            state["global_plan_status"][agent] = "direct"
            state["global_plan_subgoals"][agent] = direct_target.copy()
            _update_smoothed_subgoal(state, agent, direct_target)
            state["global_plan_completed"][agent] = True
            debug_state, switched, previous_index = _update_global_nav_debug(
                state,
                agent,
                route_index=0,
                distance_to_subgoal=float(np.linalg.norm(direct_target - np.asarray(env.agent_positions[agent], dtype=np.float32))),
                step_completed=step_completed,
            )
            continue

        current = np.asarray(env.agent_positions[agent], dtype=np.float32)
        while route_index < len(route):
            waypoint = np.asarray(route[route_index], dtype=np.float32)
            distance_to_waypoint = float(np.linalg.norm(waypoint - current))
            cooldown_until_step = debug_state.get("cooldown_until_step")
            within_cooldown = (
                current_step is not None
                and cooldown_until_step is not None
                and int(current_step) < int(cooldown_until_step)
            )
            if distance_to_waypoint <= switch_threshold:
                debug_state["threshold_hold_steps"] = int(debug_state.get("threshold_hold_steps", 0)) + 1
                if not within_cooldown and int(debug_state["threshold_hold_steps"]) >= hold_trigger:
                    route_index += 1
                    debug_state["threshold_hold_steps"] = 0
                    debug_state["cooldown_until_step"] = (
                        None if current_step is None else int(current_step) + cooldown_steps
                    )
                    continue
            else:
                debug_state["threshold_hold_steps"] = 0
            force_advance, force_reason = _should_force_advance_waypoint(
                env,
                state,
                agent,
                distance_to_waypoint,
                debug_state,
            )
            if force_advance:
                route_index += 1
                debug_state["threshold_hold_steps"] = 0
                debug_state["cooldown_until_step"] = (
                    None if current_step is None else int(current_step) + cooldown_steps
                )
                continue
            break

        if route_index >= len(route):
            state["global_plan_indices"][agent] = route_index
            state["global_plan_status"][agent] = "task"
            state["global_plan_subgoals"][agent] = direct_target.copy()
            _update_smoothed_subgoal(state, agent, direct_target)
            state["global_plan_completed"][agent] = True
            debug_state, switched, previous_index = _update_global_nav_debug(
                state,
                agent,
                route_index=route_index,
                distance_to_subgoal=float(np.linalg.norm(direct_target - current)),
                step_completed=step_completed,
            )
            continue

        current_waypoint = np.asarray(route[route_index], dtype=np.float32)
        state["global_plan_indices"][agent] = route_index
        state["global_plan_status"][agent] = "route"
        state["global_plan_subgoals"][agent] = current_waypoint.copy()
        _update_smoothed_subgoal(state, agent, current_waypoint)
        state["global_plan_completed"][agent] = False
        debug_state, switched, previous_index = _update_global_nav_debug(
            state,
            agent,
            route_index=route_index,
            distance_to_subgoal=float(np.linalg.norm(current_waypoint - current)),
            step_completed=step_completed,
        )


def _get_guidance_target(env, state, agent):
    if not bool(state.get("global_planner_enabled", False)):
        return np.asarray(env.agent_targets[agent], dtype=np.float32).copy()
    guidance_target = _compute_segment_lookahead_target(env, state, agent)
    if guidance_target is not None:
        guidance_target = _get_detour_target(env, state, agent, guidance_target)
        return np.asarray(guidance_target, dtype=np.float32).copy()
    guidance_target = state.get("global_plan_smoothed_subgoals", {}).get(agent)
    if guidance_target is None:
        guidance_target = state.get("global_plan_subgoals", {}).get(agent)
    if guidance_target is None:
        return np.asarray(env.agent_targets[agent], dtype=np.float32).copy()
    guidance_target = _get_detour_target(env, state, agent, guidance_target)
    return np.asarray(guidance_target, dtype=np.float32).copy()


def _should_skip_turn_limit(env, agent):
    eval_cfg = _get_hidden_eval_policy_config()
    if not bool(eval_cfg["emergency_avoidance"]):
        return False
    threshold = float(np.clip(eval_cfg["emergency_lidar_threshold"], 0.0, 1.0))
    if threshold <= 0.0 or agent not in env.agent_positions:
        return False
    lidar_values = env._cast_rays(env.agent_positions[agent]).astype(np.float32)
    min_lidar = float(np.min(lidar_values)) if lidar_values.size > 0 else 1.0
    return min_lidar < threshold


def _limit_eval_turn_rate(action, previous_action, max_turn_deg):
    action = np.asarray(action, dtype=np.float32)
    previous_action = np.asarray(previous_action, dtype=np.float32)
    if max_turn_deg is None or float(max_turn_deg) <= 0.0:
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    prev_norm = float(np.linalg.norm(previous_action))
    curr_norm = float(np.linalg.norm(action))
    if prev_norm <= 1e-6 or curr_norm <= 1e-6:
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    prev_dir = previous_action / prev_norm
    curr_dir = action / curr_norm
    dot_value = float(np.clip(np.dot(prev_dir, curr_dir), -1.0, 1.0))
    current_angle = float(np.degrees(np.arccos(dot_value)))
    if current_angle <= float(max_turn_deg):
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    blend_ratio = float(max_turn_deg) / max(current_angle, 1e-6)
    limited_dir = (1.0 - blend_ratio) * prev_dir + blend_ratio * curr_dir
    limited_norm = float(np.linalg.norm(limited_dir))
    if limited_norm <= 1e-6:
        return np.clip(action, -1.0, 1.0).astype(np.float32)
    limited_dir = limited_dir / limited_norm
    limited_action = limited_dir * curr_norm
    limited_action = np.clip(limited_action, -1.0, 1.0)
    limited_norm = float(np.linalg.norm(limited_action))
    if limited_norm > 1.0:
        limited_action = limited_action / limited_norm
    return limited_action.astype(np.float32)


def _compute_guided_policy_action(env, state, model, agent, previous_action):
    guidance_target = _get_guidance_target(env, state, agent)
    real_target = np.asarray(env.agent_targets[agent], dtype=np.float32).copy()
    policy_predict_time = 0.0
    policy_predict_count = 0

    env.agent_targets[agent] = guidance_target
    try:
        if _should_activate_apf_override(env, state, agent, guidance_target):
            state.setdefault("apf_override_active", {})[agent] = True
            action = _compute_apf_override_action(
                env,
                state,
                agent,
                guidance_target,
                previous_action,
            )
            action = _apply_eval_emergency_avoidance(env, agent, action, state["_runtime_args"])
            if not _should_skip_turn_limit(env, agent):
                turn_limit_deg = float(getattr(state["_runtime_args"], "eval_turn_limit_deg", 35.0))
                action = _limit_eval_turn_rate(action, previous_action, turn_limit_deg)
            action = _enforce_eval_min_agl(
                env,
                agent,
                action,
                min_agl=max(float(_get_hidden_eval_policy_config()["min_safe_agl"]), 0.0),
            )
            _update_apf_override_state(env, state, agent, guidance_target)
            return np.asarray(action, dtype=np.float32), policy_predict_time, policy_predict_count

        predict_started_at = time.perf_counter()
        raw_action, _ = model.predict(env.observe(agent), deterministic=True)
        policy_predict_time += time.perf_counter() - predict_started_at
        policy_predict_count += 1
        eval_cfg = _get_hidden_eval_policy_config()
        action = _smooth_eval_action(
            env,
            agent,
            raw_action,
            previous_action,
            smoothing_factor=float(np.clip(eval_cfg["smoothing"], 0.0, 0.99)),
            target_blend=float(np.clip(eval_cfg["goal_blend"], 0.0, 1.0)),
        )
        action = _apply_eval_emergency_avoidance(env, agent, action, state["_runtime_args"])
        if not _should_skip_turn_limit(env, agent):
            turn_limit_deg = float(getattr(state["_runtime_args"], "eval_turn_limit_deg", 35.0))
            action = _limit_eval_turn_rate(action, previous_action, turn_limit_deg)
        action = _enforce_eval_min_agl(
            env,
            agent,
            action,
            min_agl=max(float(eval_cfg["min_safe_agl"]), 0.0),
        )
        return np.asarray(action, dtype=np.float32), policy_predict_time, policy_predict_count
    finally:
        env.agent_targets[agent] = real_target


def build_env(args):    # 构建路点环境
    env_kwargs = _build_waypoint_env_kwargs(
        args_or_none=args,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
    )
    # 仅在测试链路关闭无人机间碰撞检测，避免干扰当前全局路点联调。
    env_kwargs["collision_radius"] = 0.0
    env_kwargs["agent_collision_penalty"] = 0.0
    return UAVWaypoint3DMAPFEnv(**env_kwargs, log_collisions=True)


def load_assignment_override(path): # 读取当前手动任务指派
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("assignment_json 必须是 JSON 对象。")
    return payload


def load_scenario_payload(path):    # 读取场景
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("scenario_json 必须是 JSON 对象。")
    return payload


def _apply_scenario_runtime_defaults(args, scenario_payload):
    if not isinstance(scenario_payload, dict):
        return

    scenario_active_n_agents = scenario_payload.get("active_n_agents")
    if (
        scenario_active_n_agents is not None
        and int(getattr(args, "active_n_agents", 0)) <= 0
    ):
        args.active_n_agents = int(scenario_active_n_agents)


def _build_geo_origin_from_args(args, scenario_payload=None):
    scenario_payload = scenario_payload or {}
    origin_lat = scenario_payload.get("origin_lat", getattr(args, "origin_lat", None))
    origin_lon = scenario_payload.get("origin_lon", getattr(args, "origin_lon", None))
    if origin_lat is None or origin_lon is None:
        return None
    return {
        "origin_lat": float(origin_lat),
        "origin_lon": float(origin_lon),
    }


def _enu_km_to_geodetic(point, geo_origin):
    if geo_origin is None:
        return None
    point = np.asarray(point, dtype=np.float32)
    if point.shape[0] < 3:
        return None

    origin_lat = float(geo_origin["origin_lat"])
    origin_lon = float(geo_origin["origin_lon"])
    lat_rad = math.radians(origin_lat)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = max(111320.0 * math.cos(lat_rad), 1e-6)

    east_m = float(point[0]) * 1000.0
    north_m = float(point[1]) * 1000.0
    altitude_km = float(point[2])

    latitude = origin_lat + north_m / meters_per_deg_lat
    longitude = origin_lon + east_m / meters_per_deg_lon
    return {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "alt_km": altitude_km,
    }


def _attach_geo_to_point_payload(point, geo_origin):
    payload = {
        "enu_km": np.asarray(point, dtype=np.float32).tolist(),
    }
    geodetic = _enu_km_to_geodetic(point, geo_origin)
    if geodetic is not None:
        payload.update(geodetic)
    return payload


def _clip_point_height(env, point, z_max=3.0):
    clipped = np.asarray(point, dtype=np.float32).copy()
    clipped[2] = float(np.clip(clipped[2], 0.0, min(float(z_max), float(env.Z))))
    return clipped


def sample_task_points(env, count, existing_positions=None):    # 采样任务点
    existing_positions = list(existing_positions or [])
    tasks = []
    for _ in range(count):
        task = env._sample_ground_target_position(
            min_distance=env.spawn_min_separation,
            existing_positions=existing_positions,
        )
        task = _clip_point_height(env, task, z_max=3.0)
        tasks.append(np.asarray(task, dtype=np.float32))
        existing_positions.append(np.asarray(task, dtype=np.float32))
    return tasks


def build_tasks_from_specs(task_points, task_specs):   # 
    tasks = {}
    for idx, spec in enumerate(task_specs):
        task_id = str(spec["task_id"])
        task_position = np.asarray(task_points[idx], dtype=np.float32)
        task_position[2] = float(np.clip(task_position[2], 0.0, 3.0))
        slot_positions = spec.get("slot_positions")
        normalized_slot_positions = None
        if isinstance(slot_positions, dict):
            normalized_slot_positions = {}
            for agent, target in slot_positions.items():
                clipped_target = np.asarray(target, dtype=np.float32).copy()
                clipped_target[2] = float(np.clip(clipped_target[2], 0.0, 3.0))
                normalized_slot_positions[str(agent)] = clipped_target
        tasks[task_id] = {
            "position": task_position,
            "agents": [str(agent) for agent in spec.get("agents", [])],
            "label": spec.get("label", task_id),
            "active": bool(spec.get("active", True)),
            "completed": bool(spec.get("completed", False)),
        }
        if normalized_slot_positions is not None:
            tasks[task_id]["slot_positions"] = normalized_slot_positions
    return tasks


# 获取无人机信息与任务信息的接口函数
def _build_serializable_task_info(task_id, task):   # 构建可直接查询/打印的任务信息
    geo_origin = task.get("geo_origin")
    task_info = {
        "task_id": str(task_id),
        "position": np.asarray(task.get("position", []), dtype=np.float32).tolist(),
        "position_geo": _attach_geo_to_point_payload(task.get("position", []), geo_origin),
        "agents": [str(agent) for agent in task.get("agents", [])],
        "label": task.get("label", str(task_id)),
        "active": bool(task.get("active", True)),
        "completed": bool(task.get("completed", False)),
    }
    slot_positions = task.get("slot_positions")
    if isinstance(slot_positions, dict):
        task_info["slot_positions"] = {
            str(agent): np.asarray(target, dtype=np.float32).tolist()
            for agent, target in slot_positions.items()
        }
        task_info["slot_positions_geo"] = {
            str(agent): _attach_geo_to_point_payload(target, geo_origin)
            for agent, target in slot_positions.items()
        }
    return task_info


def get_agent_task_dict(state):   # 查询每个无人机当前对应的任务字典
    tasks = state.get("tasks", {})
    agent_queues = state.get("agent_queues", {})
    agent_task_dict = {}

    for agent, queue in agent_queues.items():
        normalized_agent = str(agent)
        agent_task_dict[normalized_agent] = {
            "task_ids": [],
            "tasks": [],
        }
        for task_id in queue:
            normalized_task_id = str(task_id)
            task = tasks.get(normalized_task_id)
            if task is None:
                continue
            agent_task_dict[normalized_agent]["task_ids"].append(normalized_task_id)
            agent_task_dict[normalized_agent]["tasks"].append(
                _build_serializable_task_info(normalized_task_id, task)
            )

    return agent_task_dict


def get_task_agent_dict(state):   # 查询每个任务当前对应的无人机列表
    tasks = state.get("tasks", {})
    task_agent_dict = {}

    for task_id, task in tasks.items():
        normalized_task_id = str(task_id)
        task_agent_dict[normalized_task_id] = _build_serializable_task_info(
            normalized_task_id,
            task,
        )

    return task_agent_dict


def _load_scenario_state_from_setup(setup_path):   # 从保存的场景文件中读取 scenario_state
    with open(setup_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("setup 文件内容必须是 JSON 对象。")

    scenario_state = payload.get("scenario_state")
    if not isinstance(scenario_state, dict):
        raise ValueError("setup 文件中缺少 scenario_state 字段。")
    return scenario_state


def get_agent_task_dict_from_setup(setup_path):   # 从 setup.json 查询每个无人机对应的任务字典
    scenario_state = _load_scenario_state_from_setup(setup_path)
    return get_agent_task_dict(scenario_state)


def get_task_agent_dict_from_setup(setup_path):   # 从 setup.json 查询每个任务对应的无人机列表
    scenario_state = _load_scenario_state_from_setup(setup_path)
    return get_task_agent_dict(scenario_state)


def get_task_target_for_agent(task, agent): # 构建agent的任务目标
    slot_positions = task.get("slot_positions")
    if isinstance(slot_positions, dict):
        slot_target = slot_positions.get(str(agent))
        if slot_target is not None:
            clipped = np.asarray(slot_target, dtype=np.float32).copy()
            clipped[2] = float(np.clip(clipped[2], 0.0, 3.0))
            return clipped
    clipped = np.asarray(task["position"], dtype=np.float32).copy()
    clipped[2] = float(np.clip(clipped[2], 0.0, 3.0))
    return clipped


def build_render_task_paths(state): # 构建渲染任务路径
    tasks = state.get("tasks", {})
    task_paths = {}
    for agent, queue in state.get("agent_queues", {}).items():
        path = []
        for task_id in queue:
            task = tasks.get(task_id)
            if task is None or not task.get("active", True) or task.get("completed", False):
                continue
            path.append(get_task_target_for_agent(task, agent))
        if path:
            task_paths[str(agent)] = path
    return task_paths


def build_render_visible_tasks(state):  # 渲染当前已出现但未完成的任务点
    visible_tasks = []
    geo_origin = state.get("geo_origin")
    for task_id, task in state.get("tasks", {}).items():
        if not task.get("active", True):
            continue
        visible_tasks.append(
            {
                "task_id": str(task_id),
                "label": str(task.get("label", task_id)),
                "position": np.asarray(task.get("position", []), dtype=np.float32).copy(),
                "position_geo": _attach_geo_to_point_payload(task.get("position", []), geo_origin),
                "completed": bool(task.get("completed", False)),
            }
        )
    return visible_tasks


def _format_task_status_for_render(state, task_id, queue_index):
    task_id = str(task_id)
    task = state.get("tasks", {}).get(task_id, {})
    if task.get("completed", False):
        return "done"
    if not task.get("active", True):
        return "inactive"
    return "active" if int(queue_index) == 0 else "wait"


def build_render_task_status_panel(state, max_tasks_per_agent=3):
    inactive_agents = {str(agent) for agent in state.get("inactive_agents", set())}
    active_agents = [str(agent) for agent in state.get("active_agents", [])]
    if not active_agents:
        active_agents = [
            str(agent)
            for agent in state.get("agent_queues", {}).keys()
            if str(agent) not in inactive_agents
        ]

    rows = []
    for agent in active_agents:
        if agent in inactive_agents:
            continue
        queue = [str(task_id) for task_id in state.get("agent_queues", {}).get(agent, [])]
        formatted_tasks = []
        for idx, task_id in enumerate(queue[:max_tasks_per_agent]):
            status = _format_task_status_for_render(state, task_id, idx)
            formatted_tasks.append(f"{task_id}({status})")
        if len(queue) > max_tasks_per_agent:
            formatted_tasks.append("...")
        rows.append({
            "agent": agent,
            "tasks": formatted_tasks,
            "text": f"{agent}: {' -> '.join(formatted_tasks) if formatted_tasks else 'idle'}",
        })
    return rows


def get_damaged_agents_for_render(state):   # 渲染受损agent
    if "damaged_agents" in state:
        return [str(agent) for agent in state.get("damaged_agents", [])]
    if state.get("loss_triggered", False):
        return [str(agent) for agent in state.get("lost_agents", [])]
    return []


def build_render_meta(state):   # 构建渲染meta
    global_subgoals = {}
    global_routes = {}
    geo_origin = state.get("geo_origin")
    inactive_agents = {str(agent) for agent in state.get("inactive_agents", set())}
    for agent, subgoal in state.get("global_plan_subgoals", {}).items():
        if str(agent) in inactive_agents:
            continue
        if subgoal is not None:
            global_subgoals[str(agent)] = np.asarray(subgoal, dtype=np.float32).copy()
    for agent, route in state.get("global_plan_routes", {}).items():
        if str(agent) in inactive_agents:
            continue
        if isinstance(route, (list, tuple)) and route:
            global_routes[str(agent)] = [
                np.asarray(point, dtype=np.float32).copy()
                for point in route
            ]
    return {
        "task_paths": build_render_task_paths(state),
        "visible_tasks": build_render_visible_tasks(state),
        "global_subgoals": global_subgoals,
        "global_subgoals_geo": {
            str(agent): _attach_geo_to_point_payload(subgoal, geo_origin)
            for agent, subgoal in global_subgoals.items()
        },
        "global_routes": global_routes,
        "global_routes_geo": {
            str(agent): [_attach_geo_to_point_payload(point, geo_origin) for point in route]
            for agent, route in global_routes.items()
        },
        "damaged_agents": get_damaged_agents_for_render(state),
        "discovered_hemisphere_indices": sorted(
            int(idx) for idx in state.get("discovered_hemisphere_indices", set())
        ),
        "geo_origin": None if geo_origin is None else dict(geo_origin),
        "active_agents": [str(agent) for agent in state.get("active_agents", [])],
        "inactive_agents": sorted(inactive_agents),
        "task_status_panel": build_render_task_status_panel(state, max_tasks_per_agent=3),
        "reassignment_panel": copy.deepcopy(state.get("last_reassignment_panel", None)),
    }

def _hemisphere_detected_by_agent(env, obstacle, agent_pos):
    if getattr(obstacle, "kind", "") != "hemisphere":
        return False

    lidar_range = float(getattr(env, "lidar_range", 0.0))
    if lidar_range <= 0.0:
        return False

    num_steps = max(int(getattr(env, "lidar_num_samples", 1)), 1)
    step_size = lidar_range / num_steps
    for direction in getattr(env, "ray_dirs", []):
        for step in range(1, num_steps + 1):
            travel = min(step * step_size, lidar_range)
            test_pos = agent_pos + direction * travel
            if obstacle.contains(test_pos[0], test_pos[1], test_pos[2]):
                return True
    return False


def update_discovered_hemispheres(env, state):
    discovered = set(state.get("discovered_hemisphere_indices", set()))
    for idx, obstacle in enumerate(getattr(env, "obstacles", [])):
        if getattr(obstacle, "kind", "") != "hemisphere" or idx in discovered:
            continue
        for agent in getattr(env, "agents", []):
            if agent not in getattr(env, "agent_positions", {}):
                continue
            agent_pos = np.asarray(env.agent_positions[agent], dtype=np.float32)
            if _hemisphere_detected_by_agent(env, obstacle, agent_pos):
                discovered.add(idx)
                break
    state["discovered_hemisphere_indices"] = discovered



def apply_fixed_start_positions(env, start_positions):   # 给定初始位置
    if not start_positions:
        return
    if len(start_positions) < len(env.possible_agents):
        raise ValueError("scenario_json 中的 start_positions 数量不足。")

    env.agents = env.possible_agents.copy()
    env.step_cnt = 0
    for agent_idx, agent in enumerate(env.possible_agents):
        start = _clip_point_height(env, start_positions[agent_idx], z_max=3.0)
        env.agent_positions[agent] = start
        env.agent_last_motion[agent] = np.zeros(3, dtype=np.float32)
        env.agent_targets[agent] = start.copy()
        env.agent_initial_distances[agent] = 0.0
        env.agent_reached_targets[agent] = False
        env.agent_dones[agent] = False
        env.agent_timeouts[agent] = False
        env.agent_obstacle_collisions[agent] = 0
        env.agent_inter_agent_collisions[agent] = 0
        env.agent_collision_reasons[agent] = None


def build_task_ids(task_count):
    return [f"task_{idx + 1:02d}" for idx in range(max(int(task_count), 0))]


def normalize_task_specs(task_specs):
    normalized = []
    for idx, spec in enumerate(task_specs or []):
        if not isinstance(spec, dict):
            raise ValueError(f"task_specs[{idx}] 必须是 JSON 对象。")
        task_id = str(spec.get("task_id", f"task_{idx + 1:02d}"))
        normalized_spec = {
            "task_id": task_id,
            "agents": [str(agent) for agent in spec.get("agents", [])],
        }
        for key in ("label", "active", "completed"):
            if key in spec:
                normalized_spec[key] = spec[key]
        normalized.append(normalized_spec)
    return normalized


def build_round_robin_task_specs(agent_names, task_count):
    agent_names = [str(agent) for agent in agent_names]
    if not agent_names:
        return []
    task_specs = []
    for idx, task_id in enumerate(build_task_ids(task_count)):
        task_specs.append({
            "task_id": task_id,
            "agents": [agent_names[idx % len(agent_names)]],
        })
    return task_specs


def build_agent_queues_from_task_specs(agent_names, task_specs, allowed_task_ids=None):
    agent_names = [str(agent) for agent in agent_names]
    allowed = None if allowed_task_ids is None else {str(task_id) for task_id in allowed_task_ids}
    agent_queues = {agent: [] for agent in agent_names}
    for spec in normalize_task_specs(task_specs):
        task_id = str(spec["task_id"])
        if allowed is not None and task_id not in allowed:
            continue
        seen_agents = set()
        for agent in spec.get("agents", []):
            agent = str(agent)
            if agent not in agent_queues or agent in seen_agents:
                continue
            agent_queues[agent].append(task_id)
            seen_agents.add(agent)
    return agent_queues


def resolve_agent_queues_from_provider_payload(agent_names, provider_payload, allowed_task_ids=None):
    if not isinstance(provider_payload, dict):
        return None

    agent_queues = provider_payload.get("agent_queues")
    if agent_queues is None:
        task_specs = provider_payload.get("task_specs")
        if task_specs is not None:
            agent_queues = build_agent_queues_from_task_specs(
                agent_names,
                task_specs,
                allowed_task_ids=allowed_task_ids,
            )

    if agent_queues is None:
        return None

    normalized = normalize_agent_queues(agent_names, agent_queues)
    if allowed_task_ids is not None:
        normalized = filter_agent_queues(agent_names, normalized, allowed_task_ids)
    return normalize_unique_task_queues(agent_names, normalized)


def normalize_agent_queues(agent_names, agent_queues):
    normalized = {str(agent): [] for agent in agent_names}
    if not isinstance(agent_queues, dict):
        return normalized
    for agent in normalized.keys():
        normalized[agent] = [str(task_id) for task_id in agent_queues.get(agent, [])]
    return normalized


def filter_agent_queues(agent_names, agent_queues, allowed_task_ids):
    allowed = {str(task_id) for task_id in allowed_task_ids}
    filtered = normalize_agent_queues(agent_names, agent_queues)
    for agent, queue in filtered.items():
        filtered[agent] = [task_id for task_id in queue if task_id in allowed]
    return filtered


def merge_agent_queues(agent_names, *queue_maps):
    agent_names = [str(agent) for agent in agent_names]
    merged = {agent: [] for agent in agent_names}
    seen = {agent: set() for agent in agent_names}
    for queue_map in queue_maps:
        normalized = normalize_agent_queues(agent_names, queue_map or {})
        for agent, queue in normalized.items():
            for task_id in queue:
                if task_id in seen[agent]:
                    continue
                merged[agent].append(task_id)
                seen[agent].add(task_id)
    return merged


def normalize_unique_task_queues(agent_names, agent_queues):
    agent_names = [str(agent) for agent in agent_names]
    normalized = normalize_agent_queues(agent_names, agent_queues)
    task_owners = {}
    unique = {agent: [] for agent in agent_names}
    for agent in agent_names:
        for task_id in normalized.get(agent, []):
            task_id = str(task_id)
            if task_id in task_owners:
                continue
            task_owners[task_id] = agent
            unique[agent].append(task_id)
    return unique


def remove_task_ids_from_queues(agent_names, agent_queues, task_ids):
    task_ids = {str(task_id) for task_id in task_ids}
    normalized = normalize_agent_queues(agent_names, agent_queues)
    for agent, queue in normalized.items():
        normalized[agent] = [task_id for task_id in queue if task_id not in task_ids]
    return normalized


def sync_task_assignments_from_queues(state, agent_names=None, queue_maps=None):
    tasks = state.get("tasks", {})
    if agent_names is None:
        agent_names = list(state.get("agent_queues", {}).keys())
    agent_names = [str(agent) for agent in agent_names]
    queue_maps = list(queue_maps) if queue_maps is not None else [state.get("agent_queues", {})]
    merged_queues = merge_agent_queues(agent_names, *queue_maps)

    task_to_agents = {}
    for agent, queue in merged_queues.items():
        for task_id in queue:
            task_to_agents.setdefault(str(task_id), []).append(str(agent))

    for task_id, task in tasks.items():
        task["agents"] = task_to_agents.get(str(task_id), [])

    return merged_queues


def resolve_assignment_plan(
    agent_names,
    task_count,
    assignment_override=None,
    default_task_spec_builder=None,
    default_agent_queue_builder=None,
    assignment_provider=None,
    provider_context=None,
):
    agent_names = [str(agent) for agent in agent_names]
    task_count = max(int(task_count), 0)
    assignment_override = assignment_override or {}
    provider_payload = {}

    if callable(assignment_provider):
        provider_payload = call_assignment_provider(
            assignment_provider,
            agent_names=agent_names,
            task_count=task_count,
            assignment_override=assignment_override,
            provider_context=provider_context,
        )

    task_specs = assignment_override.get("task_specs")
    if task_specs is None:
        task_specs = provider_payload.get("task_specs")
    if task_specs is None:
        task_spec_builder = default_task_spec_builder or build_round_robin_task_specs
        task_specs = task_spec_builder(agent_names, task_count)
    task_specs = normalize_task_specs(task_specs)

    agent_queues = assignment_override.get("agent_queues")
    if agent_queues is None:
        agent_queues = provider_payload.get("agent_queues")
    if agent_queues is None:
        if callable(default_agent_queue_builder):
            agent_queues = default_agent_queue_builder(agent_names, task_specs)
        else:
            agent_queues = build_agent_queues_from_task_specs(agent_names, task_specs)
    agent_queues = normalize_agent_queues(agent_names, agent_queues)

    return {
        "task_specs": task_specs,
        "agent_queues": agent_queues,
        "provider_payload": provider_payload,
    }


def resolve_task_points(env, args, scenario_payload, existing_positions=None, count=None):
    count = max(int(args.task_count if count is None else count), 0)
    task_positions = scenario_payload.get("task_positions") if isinstance(scenario_payload, dict) else None
    task_height_mode = str(
        scenario_payload.get("task_height_mode", "absolute")
        if isinstance(scenario_payload, dict)
        else "absolute"
    ).strip().lower()
    if isinstance(task_positions, list) and len(task_positions) >= count:
        resolved_positions = []
        for idx in range(count):
            position = np.asarray(task_positions[idx], dtype=np.float32).copy()
            if task_height_mode == "agl":
                surface_height = float(env._terrain_surface_height(position[0], position[1]))
                position[2] = surface_height + float(position[2])
            resolved_positions.append(position)
        return resolved_positions
    return sample_task_points(env, count, existing_positions=existing_positions)


def _meters_to_env_height(env, height_meters):
    height_meters = float(height_meters)
    terrain = getattr(env, "terrain", None)
    if terrain is not None and hasattr(terrain, "z_scale"):
        return height_meters * float(getattr(terrain, "z_scale", 1.0))
    return height_meters


def _resolve_lowland_height_threshold(env, args):
    return _meters_to_env_height(env, getattr(args, "lowland_height_threshold", 10.0))


def build_obstacles_from_xy(env, obstacle_xy, args):
    built = []
    lowland_threshold = _resolve_lowland_height_threshold(env, args)
    for point in obstacle_xy:
        x = float(point[0])
        y = float(point[1])
        surface_height = float(env._terrain_surface_height(x, y))
        if surface_height < lowland_threshold:
            radius = float(np.random.uniform(
                min(args.lowland_cylinder_radius_min, args.lowland_cylinder_radius_max),
                max(args.lowland_cylinder_radius_min, args.lowland_cylinder_radius_max),
            ))
            height = float(np.random.uniform(
                min(args.lowland_cylinder_height_min, args.lowland_cylinder_height_max),
                max(args.lowland_cylinder_height_min, args.lowland_cylinder_height_max),
            ))
            built.append(CylinderObstacle(x, y, radius, height))
        else:
            radius = float(np.random.uniform(
                min(args.terrain_radius_min, args.terrain_radius_max),
                max(args.terrain_radius_min, args.terrain_radius_max),
            ))
            height = float(np.random.uniform(
                min(args.terrain_height_min, args.terrain_height_max),
                max(args.terrain_height_min, args.terrain_height_max),
            ))
            built.append(EllipsoidalHemisphereObstacle(x, y, radius, height, z=surface_height))
    return built


def _sample_scenario_obstacles(env, tasks, args):
    count = max(int(getattr(args, "scenario_obstacle_count", 0)), 0)
    if count <= 0:
        return []

    cyl_radius_min = float(min(args.lowland_cylinder_radius_min, args.lowland_cylinder_radius_max))
    cyl_radius_max = float(max(args.lowland_cylinder_radius_min, args.lowland_cylinder_radius_max))
    cyl_height_min = float(min(args.lowland_cylinder_height_min, args.lowland_cylinder_height_max))
    cyl_height_max = float(max(args.lowland_cylinder_height_min, args.lowland_cylinder_height_max))
    hemi_radius_min = float(min(args.terrain_radius_min, args.terrain_radius_max))
    hemi_radius_max = float(max(args.terrain_radius_min, args.terrain_radius_max))
    hemi_height_min = float(min(args.terrain_height_min, args.terrain_height_max))
    hemi_height_max = float(max(args.terrain_height_min, args.terrain_height_max))
    lowland_threshold = _resolve_lowland_height_threshold(env, args)
    start_clearance = max(float(getattr(args, "spawn_min_height_buffer", 8.0)), 0.0)
    task_clearance = max(float(getattr(args, "task_min_obstacle_clearance", 10.0)), 0.0)
    obstacle_clearance = max(float(getattr(args, "obstacle_min_spacing", 12.0)), 0.0)

    existing_obstacles = list(getattr(env, "obstacles", []))
    starts = [
        np.asarray(env.agent_positions[agent], dtype=np.float32)
        for agent in env.possible_agents
        if agent in env.agent_positions
    ]
    task_positions = [
        np.asarray(task["position"], dtype=np.float32)
        for task in tasks.values()
    ]

    generated_obstacles = []
    max_attempts = max(400, count * 200)
    for _ in range(max_attempts):
        if len(generated_obstacles) >= count:
            break
        x = float(np.random.uniform(0.18 * env.X, 0.82 * env.X))
        y = float(np.random.uniform(0.12 * env.Y, 0.88 * env.Y))
        surface_height = float(env._terrain_surface_height(x, y))
        use_cylinder = surface_height < lowland_threshold
        if use_cylinder:
            radius = float(np.random.uniform(cyl_radius_min, cyl_radius_max))
            height = float(np.random.uniform(cyl_height_min, cyl_height_max))
        else:
            radius = float(np.random.uniform(hemi_radius_min, hemi_radius_max))
            height = float(np.random.uniform(hemi_height_min, hemi_height_max))

        if any(np.linalg.norm(np.array([x, y], dtype=np.float32) - pos[:2]) <= radius + start_clearance for pos in starts):
            continue
        if any(np.linalg.norm(np.array([x, y], dtype=np.float32) - pos[:2]) <= radius + task_clearance for pos in task_positions):
            continue

        blocked = False
        for obstacle in existing_obstacles + generated_obstacles:
            ox = float(getattr(obstacle, "x", 0.0))
            oy = float(getattr(obstacle, "y", 0.0))
            other_extent = float(getattr(obstacle, "xy_extent", getattr(obstacle, "radius", 0.0)))
            if np.hypot(x - ox, y - oy) <= radius + other_extent + obstacle_clearance:
                blocked = True
                break
        if blocked:
            continue

        if use_cylinder:
            generated_obstacles.append(CylinderObstacle(x, y, radius, height))
        else:
            generated_obstacles.append(
                EllipsoidalHemisphereObstacle(x, y, radius, height, z=surface_height)
            )

    return generated_obstacles


def mark_idle_agent(env, agent):
    env.agent_targets[agent] = np.asarray(env.agent_positions[agent], dtype=np.float32).copy()
    env.agent_reached_targets[agent] = True
    env.agent_dones[agent] = False
    env.agent_initial_distances[agent] = 0.0


def refresh_agent_targets(env, state):
    inactive_agents = set(state.get("inactive_agents", set()))
    direct_targets = {}
    for agent in env.possible_agents:
        if agent in inactive_agents:
            mark_idle_agent(env, agent)
            _reset_global_nav_state_for_agent(state, agent)
            continue

        queue = state["agent_queues"].get(agent, [])
        next_task_id = None
        for task_id in queue:
            task = state["tasks"].get(task_id)
            if task is not None and task.get("active", True) and not task.get("completed", False):
                next_task_id = task_id
                break

        if next_task_id is None:
            mark_idle_agent(env, agent)
            _reset_global_nav_state_for_agent(state, agent)
            continue

        target = get_task_target_for_agent(state["tasks"][next_task_id], agent)
        direct_targets[agent] = np.asarray(target, dtype=np.float32).copy()
        env.agent_targets[agent] = direct_targets[agent].copy()
        env.agent_reached_targets[agent] = False
        env.agent_dones[agent] = False
        env.agent_initial_distances[agent] = float(
            np.linalg.norm(env.agent_targets[agent] - env.agent_positions[agent])
        )
    state["direct_task_targets"] = direct_targets

    if bool(state.get("global_planner_enabled", False)):
        if _targets_changed_for_replan(state, direct_targets):
            _append_global_nav_event(state, "检测到任务目标变化，重新生成全局路点")
            _plan_global_routes(env, state)
        _sync_navigation_targets(env, state)


def promote_completed_tasks(env, state):
    completed_now = []
    inactive_agents = set(state.get("inactive_agents", set()))
    auto_complete_unassigned = bool(state.get("auto_complete_unassigned_tasks", True))
    for task_id, task in state["tasks"].items():
        if task.get("completed", False) or not task.get("active", True):
            continue

        assigned_agents = [
            agent
            for agent in task.get("agents", [])
            if agent not in inactive_agents
        ]
        if not assigned_agents:
            if auto_complete_unassigned:
                task["completed"] = True
                completed_now.append(task_id)
                refresh_agent_targets(env, state)
            continue

        ready = True
        for agent in assigned_agents:
            queue = state["agent_queues"].get(agent, [])
            if not queue or queue[0] != task_id or not env.agent_reached_targets.get(agent, False):
                ready = False
                break

        if not ready:
            continue

        task["completed"] = True
        completed_now.append(task_id)
        for agent in assigned_agents:
            queue = state["agent_queues"].get(agent, [])
            if queue and queue[0] == task_id:
                queue.pop(0)
        # Refresh immediately so later tasks in the same scan do not reuse
        # stale reached/done flags from the previous task assignment.
        refresh_agent_targets(env, state)
    return completed_now


def scenario_is_complete(state):
    tasks = state["tasks"]
    for task in tasks.values():
        if task.get("active", True) and not task.get("completed", False):
            return False
    for agent, queue in state["agent_queues"].items():
        if agent in state.get("inactive_agents", set()):
            continue
        if any(
            task_id in tasks
            and tasks[task_id].get("active", True)
            and not tasks[task_id].get("completed", False)
            for task_id in queue
        ):
            return False
    return True


def serialize_obstacle(obstacle):
    payload = {
        "kind": getattr(obstacle, "kind", obstacle.__class__.__name__),
        "type": obstacle.__class__.__name__,
    }
    for attr in ("x", "y", "z", "radius", "height"):
        if hasattr(obstacle, attr):
            payload[attr] = float(getattr(obstacle, attr))
    return payload


def build_obstacle_from_setup(payload):
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
    raise ValueError(f"不支持的障碍物类型: {payload.get('kind')}")


def _to_json_safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_safe_value(item) for key, item in value.items()}
    if callable(value):
        return None
    return str(value)


def _serialize_args_for_setup(args):
    serialized = {}
    for key, value in vars(args).items():
        if str(key).startswith("_"):
            continue
        serialized[str(key)] = _to_json_safe_value(value)
    return serialized


def restore_scene_from_setup(env, setup_payload):   # 从配置文件恢复场景
    obstacle_payloads = setup_payload.get("obstacles")
    agent_payloads = setup_payload.get("agents")
    if not isinstance(obstacle_payloads, list) or not isinstance(agent_payloads, dict):
        return

    env.obstacles = [build_obstacle_from_setup(item) for item in obstacle_payloads]
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
        target = np.asarray(payload.get("current_target", payload.get("target", start)), dtype=np.float32)
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


def save_task_scenario_setup(env, args, output_dir, run_name, scenario_name, state, model_path):    # 保存场景设置信息
    os.makedirs(output_dir, exist_ok=True)
    terrain = getattr(env, "terrain", None)
    geo_origin = state.get("geo_origin")
    setup = {
        "run_name": run_name,
        "scenario_name": scenario_name,
        "model_path": model_path,
        "geo_origin": None if geo_origin is None else dict(geo_origin),
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
        "space_dim": [float(env.X), float(env.Y), float(env.Z)],
        "obstacle_rule": env.obstacle_rule,
        "obstacles": [serialize_obstacle(obstacle) for obstacle in env.obstacles],
        "args": _serialize_args_for_setup(args),
        "scenario_state": {
            "task_count": len(state["tasks"]),
            "inactive_agents": sorted(list(state.get("inactive_agents", set()))),
            "event_log": list(state.get("event_log", [])),
            "tasks": {
                task_id: {
                    "position": np.asarray(task["position"], dtype=np.float32).tolist(),
                    "position_geo": _attach_geo_to_point_payload(task["position"], geo_origin),
                    "agents": list(task["agents"]),
                    "label": task.get("label", task_id),
                    "active": bool(task.get("active", True)),
                    "completed": bool(task.get("completed", False)),
                    "slot_positions": {
                        str(agent): np.asarray(target, dtype=np.float32).tolist()
                        for agent, target in task.get("slot_positions", {}).items()
                    },
                    "slot_positions_geo": {
                        str(agent): _attach_geo_to_point_payload(target, geo_origin)
                        for agent, target in task.get("slot_positions", {}).items()
                    },
                }
                for task_id, task in state["tasks"].items()
            },
            "agent_queues": {
                agent: list(queue) for agent, queue in state["agent_queues"].items()
            },
        },
        "agents": {},
    }

    for agent in env.possible_agents:
        setup["agents"][agent] = {
            "start": np.asarray(env.agent_positions[agent], dtype=np.float32).tolist(),
            "start_geo": _attach_geo_to_point_payload(env.agent_positions[agent], geo_origin),
            "current_target": np.asarray(env.agent_targets[agent], dtype=np.float32).tolist(),
            "current_target_geo": _attach_geo_to_point_payload(env.agent_targets[agent], geo_origin),
            "initial_distance": float(env.agent_initial_distances.get(agent, 0.0)),
        }

    setup_path = os.path.join(output_dir, f"{run_name}_setup.json")
    with open(setup_path, "w", encoding="utf-8") as f:
        json.dump(setup, f, indent=2, ensure_ascii=False)
    return setup_path


def resolve_output_dir(args, model_path, scenario_name):    # 解算输出路径
    if args.output_dir:
        return args.output_dir
    model_dir = os.path.dirname(model_path)
    if os.path.basename(model_dir).lower() == "best_model":
        model_dir = os.path.dirname(model_dir)
    return os.path.join(model_dir, f"{scenario_name}_artifacts")


def build_base_parser(description): # 构建参数parser
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config_json", type=str, default="")
    parser.add_argument("--model_root", type=str, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--eval_model_path", type=str, default="")
    parser.add_argument("--use_global_planner", type=_parse_bool, default=False)
    parser.add_argument("--global_planner_model_root", type=str, default=DEFAULT_GLOBAL_PLANNER_MODEL_ROOT)
    parser.add_argument("--global_planner_run_id", type=str, default="")
    parser.add_argument("--global_planner_model_path", type=str, default="")
    parser.add_argument("--global_planner_max_steps", type=int, default=24)
    parser.add_argument("--global_planner_waypoint_stride", type=int, default=1)
    parser.add_argument("--global_fallback_max_steps", type=int, default=5)

    parser.add_argument("--assignment_provider_module", type=str, default="assignment_provider.py")
    parser.add_argument("--assignment_provider_callable", type=str, default="assignment_provider")
    parser.add_argument("--assignment_provider_model_path", type=str, default="")
    parser.add_argument("--assignment_provider_device", type=str, default="cpu")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--save_artifacts", type=_parse_bool, default=True)
    parser.add_argument("--render", type=_parse_bool, default=False)
    parser.add_argument("--eval_render_mode", type=_parse_bool, default=False)
    parser.add_argument("--eval_pause", type=float, default=0.03)
    parser.add_argument("--eval_render_every", type=int, default=5)
    parser.add_argument("--eval_capture_every", type=int, default=5)
    parser.add_argument("--eval_gif_fps", type=int, default=10)
    parser.add_argument("--eval_history_stride", type=int, default=3)
    parser.add_argument("--eval_history_smooth_window", type=int, default=5)
    parser.add_argument("--eval_terrain_stride", type=int, default=2)
    parser.add_argument("--n_agents", type=int, default=10)
    parser.add_argument("--active_n_agents", type=int, default=0)
    parser.add_argument("--task_count", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--assignment_json", type=str, default="")
    parser.add_argument("--scenario_json", type=str, default="")
    parser.add_argument("--space_x", type=float, default=100.0)
    parser.add_argument("--space_y", type=float, default=100.0)
    parser.add_argument("--space_z", type=float, default=15.0)
    parser.add_argument("--origin_lat", type=float, default=31.95)
    parser.add_argument("--origin_lon", type=float, default=118.85)
    parser.add_argument("--terrain_enabled", type=_parse_bool, default=True)
    parser.add_argument("--terrain_source", type=str, default="csv")
    parser.add_argument("--terrain_csv_path", type=str, default=DEFAULT_TERRAIN_CSV)
    parser.add_argument("--terrain_csv_target_x", type=float, default=100.0)
    parser.add_argument("--terrain_csv_target_y", type=float, default=100.0)
    parser.add_argument("--terrain_csv_target_z", type=float, default=100.0)
    parser.add_argument("--terrain_csv_z_scale", type=float, default=0.001)
    parser.add_argument("--terrain_grid_resolution", type=float, default=1.0)
    parser.add_argument("--terrain_min_clearance", type=float, default=0.2)
    parser.add_argument("--terrain_max_clearance", type=float, default=1.0)
    parser.add_argument("--terrain_match_space_dim", type=_parse_bool, default=True)
    parser.add_argument("--terrain_spawn_clearance", type=float, default=0.1)
    parser.add_argument("--global_waypoint_switch_threshold", type=float, default=1.5)
    parser.add_argument("--global_waypoint_relaxed_threshold", type=float, default=1.8)
    parser.add_argument("--global_waypoint_stagnation_steps", type=int, default=10)
    parser.add_argument("--global_waypoint_skip_steps", type=int, default=30)
    parser.add_argument("--global_waypoint_progress_reset_threshold", type=float, default=0.02)
    parser.add_argument("--global_waypoint_jitter_tolerance", type=float, default=0.01)
    parser.add_argument("--global_waypoint_hold_steps", type=int, default=3)
    parser.add_argument("--global_waypoint_cooldown_steps", type=int, default=4)
    parser.add_argument("--global_waypoint_smoothing_alpha", type=float, default=0.25)
    parser.add_argument("--global_segment_lookahead_distance", type=float, default=1.5)
    parser.add_argument("--global_segment_min_progress_ratio", type=float, default=0.2)
    parser.add_argument("--eval_turn_limit_deg", type=float, default=35.0)
    parser.add_argument("--spawn_max_height", type=float, default=30.0)
    parser.add_argument("--ground_target_clearance", type=float, default=0.2)
    parser.add_argument("--obstacle_rule", type=str, default="hybrid", choices=["primitives", "terrain", "hybrid"])
    parser.add_argument("--num_obstacles", type=int, default=0)
    parser.add_argument("--scenario_obstacle_count", type=int, default=6)
    parser.add_argument("--terrain_num_hemispheres", type=int, default=0)
    parser.add_argument("--terrain_radius_min", type=float, default=5.0)
    parser.add_argument("--terrain_radius_max", type=float, default=10.0)
    parser.add_argument("--terrain_height_min", type=float, default=5.0)
    parser.add_argument("--terrain_height_max", type=float, default=10.0)
    parser.add_argument("--lowland_height_threshold", type=float, default=10.0)
    parser.add_argument("--lowland_cylinder_radius_min", type=float, default=5.0)
    parser.add_argument("--lowland_cylinder_radius_max", type=float, default=10.0)
    parser.add_argument("--lowland_cylinder_height_min", type=float, default=5.0)
    parser.add_argument("--lowland_cylinder_height_max", type=float, default=10.0)
    parser.add_argument("--spawn_min_height_buffer", type=float, default=8.0)
    parser.add_argument("--task_min_obstacle_clearance", type=float, default=10.0)
    parser.add_argument("--obstacle_min_spacing", type=float, default=12.0)
    parser.add_argument("--uav_speed", type=float, default=0.06)
    parser.add_argument("--decision_period", type=float, default=1.0)
    parser.add_argument("--lidar_rays", type=int, default=24)
    parser.add_argument("--lidar_range", type=float, default=0.6)
    parser.add_argument("--goal_threshold", type=float, default=1.0)
    parser.add_argument("--collab_slot_obstacle_buffer", type=float, default=2.0)
    parser.add_argument("--step_penalty", type=float, default=-0.01)
    parser.add_argument("--progress_reward_scale", type=float, default=2.0)
    parser.add_argument("--obstacle_collision_penalty", type=float, default=80.0)
    parser.add_argument("--agent_collision_penalty", type=float, default=160.0)
    parser.add_argument("--goal_reward", type=float, default=1500.0)
    parser.add_argument("--timeout_penalty_scale", type=float, default=40.0)
    parser.add_argument("--collision_radius", type=float, default=0.02)
    parser.add_argument("--near_goal_collision_free_radius", type=float, default=1.0)
    parser.add_argument("--hemisphere_exclusion_buffer", type=float, default=5.0)
    parser.add_argument("--avoid_mountain_basins", type=_parse_bool, default=True)
    parser.add_argument("--basin_window_radius", type=int, default=4)
    parser.add_argument("--basin_highland_threshold", type=float, default=20.0)
    parser.add_argument("--basin_relief_threshold", type=float, default=3.0)
    parser.add_argument("--basin_high_neighbor_ratio", type=float, default=0.35)
    return parser


def finalize_args(parser):
    args = parser.parse_args()
    args = _merge_args_from_config_json(args, parser, os.sys.argv[1:])
    get_assignment_provider(args)
    return args


_PROGRESS_SPINNER_FRAMES = "|/-\\"


def _count_active_task_progress(state):
    active_tasks = [
        task for task in state.get("tasks", {}).values()
        if task.get("active", True)
    ]
    completed_tasks = [
        task for task in active_tasks
        if task.get("completed", False)
    ]
    return len(completed_tasks), len(active_tasks)


def _count_active_uavs(env, state):
    inactive_agents = set(state.get("inactive_agents", set()))
    count = 0
    for agent in env.possible_agents:
        if agent in inactive_agents:
            continue
        if env.agent_reached_targets.get(agent, False) and not state["agent_queues"].get(agent):
            continue
        count += 1
    return count


def _log_global_nav_debug_summary(env, state, step_completed):
    # 调试阶段使用的 waypoint 摘要输出，默认静音保留逻辑。
    return


def _format_progress_eta(seconds):
    seconds = max(float(seconds), 0.0)
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _create_progress_state():
    enabled = bool(getattr(sys.stdout, "isatty", lambda: False)())
    use_ansi = enabled
    return {
        "enabled": enabled,
        "use_ansi": use_ansi,
        "last_length": 0,
        "frame_index": 0,
    }


def _clear_progress_line(progress_state):
    if not progress_state.get("enabled") or progress_state.get("last_length", 0) <= 0:
        return
    sys.stdout.write("\r" + (" " * progress_state["last_length"]) + "\r")
    sys.stdout.flush()
    progress_state["last_length"] = 0


def _progress_log(progress_state, message):
    _clear_progress_line(progress_state)
    print(message)


def _render_progress_line(progress_state, env, state, step_completed, total_steps, collision_count, avg_decision_ms, started_at):
    if not progress_state.get("enabled"):
        return

    completed_tasks, total_tasks = _count_active_task_progress(state)
    active_uavs = _count_active_uavs(env, state)
    total_steps = max(int(total_steps), 1)
    step_completed = max(min(int(step_completed), total_steps), 0)
    ratio = step_completed / total_steps
    bar_width = 28
    filled = int(ratio * bar_width)
    if filled <= 0:
        bar = "." * bar_width
    elif filled >= bar_width:
        bar = "=" * bar_width
    else:
        bar = ("=" * max(filled - 1, 0)) + ">" + ("." * (bar_width - filled))

    elapsed = max(time.perf_counter() - started_at, 0.0)
    avg_step_time = elapsed / max(step_completed, 1)
    remaining_steps = max(total_steps - step_completed, 0)
    eta = _format_progress_eta(avg_step_time * remaining_steps)
    spinner = _PROGRESS_SPINNER_FRAMES[progress_state["frame_index"] % len(_PROGRESS_SPINNER_FRAMES)]
    progress_state["frame_index"] += 1

    if progress_state.get("use_ansi"):
        prefix = f"\x1b[36m{spinner}\x1b[0m"
        bar_text = f"\x1b[96m[{bar}]\x1b[0m"
    else:
        prefix = spinner
        bar_text = f"[{bar}]"

    line = (
        f"{prefix} {bar_text} "
        f"{step_completed:>4d}/{total_steps:<4d} "
        f"| tasks {completed_tasks:>2d}/{total_tasks:<2d} "
        f"| active {active_uavs:>2d} "
        f"| collisions {collision_count:>3d} "
        f"| avg {avg_decision_ms:>6.2f} ms "
        f"| ETA {eta}"
    )

    pad = max(progress_state["last_length"] - len(line), 0)
    sys.stdout.write("\r" + line + (" " * pad))
    sys.stdout.flush()
    progress_state["last_length"] = len(line)


def _finish_progress_line(progress_state):
    if not progress_state.get("enabled"):
        return
    if progress_state.get("last_length", 0) > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()
        progress_state["last_length"] = 0


def run_task_scenario(args, scenario_name, build_state_fn, on_post_step=None, is_complete_fn=None): # 主流程
    model_path = resolve_model_path(args)
    print(f"正在加载模型: {model_path}")
    model = PPO.load(model_path, device=args.device)

    env = build_env(args)
    env.reset(seed=args.seed)

    loaded_setup = getattr(args, "_loaded_config_json", None)
    if isinstance(loaded_setup, dict) and "obstacles" in loaded_setup and "agents" in loaded_setup:
        restore_scene_from_setup(env, loaded_setup)

    scenario_payload = load_scenario_payload(args.scenario_json)
    _apply_scenario_runtime_defaults(args, scenario_payload)
    if isinstance(scenario_payload.get("start_positions"), list):
        apply_fixed_start_positions(env, scenario_payload.get("start_positions"))

    assignment_override = load_assignment_override(args.assignment_json)
    state = build_state_fn(env, args, assignment_override, scenario_payload)
    state.setdefault("inactive_agents", set())
    state.setdefault("event_log", [])
    state.setdefault("discovered_hemisphere_indices", set())
    state["_runtime_args"] = args
    state["geo_origin"] = _build_geo_origin_from_args(args, scenario_payload)
    _apply_active_agent_mask(env, state, args)
    state["global_planner_enabled"] = bool(getattr(args, "use_global_planner", False))
    _ensure_global_nav_state(state, env.possible_agents)
    obstacle_xy = scenario_payload.get("obstacle_xy") if isinstance(scenario_payload, dict) else None
    if isinstance(obstacle_xy, list) and obstacle_xy:
        scenario_obstacles = build_obstacles_from_xy(env, obstacle_xy, args)
        env.obstacles = list(scenario_obstacles)
    else:
        scenario_obstacles = _sample_scenario_obstacles(env, state["tasks"], args)
        if scenario_obstacles:
            env.obstacles.extend(scenario_obstacles)
    if scenario_obstacles:
        cylinder_count = sum(
            1 for obstacle in scenario_obstacles
            if getattr(obstacle, "kind", "") == "cylinder"
        )
        hemisphere_count = sum(
            1 for obstacle in scenario_obstacles
            if getattr(obstacle, "kind", "") == "hemisphere"
        )
        print(
            f"已生成{len(scenario_obstacles)} 个障碍物"
            f"{cylinder_count} 个圆柱障碍物"
        )
    if state["global_planner_enabled"]:
        _build_global_planner_context(args, env, state)
    refresh_agent_targets(env, state)

    model_dir = os.path.dirname(model_path)
    if os.path.basename(model_dir).lower() == "best_model":
        model_dir = os.path.dirname(model_dir)
    run_name = f"{scenario_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = resolve_output_dir(args, model_path, scenario_name)
    setup_path = save_task_scenario_setup(env, args, output_dir, run_name, scenario_name, state, model_path)
    print(f"已保存场景配置: {setup_path}")

    history = {agent: [np.asarray(env.agent_positions[agent], dtype=np.float32).copy()] for agent in env.possible_agents}
    previous_eval_actions = {
        agent: np.zeros(3, dtype=np.float32)
        for agent in env.possible_agents
    }
    decision_times = []
    policy_predict_time_total = 0.0
    policy_predict_count_total = 0
    collision_count = 0
    progress_state = _create_progress_state()
    progress_started_at = time.perf_counter()

    render_after_sim = bool(args.render or args.save_artifacts or args.eval_render_mode)
    render_snapshots = []
    if render_after_sim:
        render_snapshots.append(_make_eval_render_snapshot(env, 0, history, render_meta=build_render_meta(state)))

    if callable(getattr(state, "print", None)):
        state.print()

    if "print_summary" in state and callable(state["print_summary"]):
        state["print_summary"]()

    _render_progress_line(
        progress_state,
        env,
        state,
        step_completed=0,
        total_steps=args.max_steps,
        collision_count=collision_count,
        avg_decision_ms=0.0,
        started_at=progress_started_at,
    )

    for step in range(args.max_steps):
        actions = {}
        step_policy_predict_time = 0.0
        step_policy_predict_count = 0
        inactive_agents = set(state.get("inactive_agents", set()))
        _sync_navigation_targets(env, state, step_completed=step)

        for agent in env.possible_agents:
            if agent in inactive_agents:
                actions[agent] = np.zeros(3, dtype=np.float32)
                continue
            if env.agent_reached_targets.get(agent, False) and not state["agent_queues"].get(agent):
                actions[agent] = np.zeros(3, dtype=np.float32)
                continue
            action, predict_time, predict_count = _compute_guided_policy_action(
                env,
                state,
                model,
                agent,
                previous_eval_actions.get(agent, np.zeros(3, dtype=np.float32)),
            )
            step_policy_predict_time += float(predict_time)
            step_policy_predict_count += int(predict_count)
            actions[agent] = action.astype(np.float32)
            previous_eval_actions[agent] = actions[agent]

        if step_policy_predict_count > 0:
            decision_times.append(step_policy_predict_time / float(step_policy_predict_count))
            policy_predict_time_total += step_policy_predict_time
            policy_predict_count_total += step_policy_predict_count

        observations, rewards, terminations, truncations, infos = env.step(actions)
        del observations, rewards

        for agent in env.possible_agents:
            history[agent].append(np.asarray(env.agent_positions[agent], dtype=np.float32).copy())

        _sync_navigation_targets(env, state, step_completed=step + 1)
        _log_global_nav_debug_summary(env, state, step_completed=step + 1)

        completed_now = promote_completed_tasks(env, state)
        if completed_now:
            sim_time = format_sim_time(step + 1, state)
            _progress_log(progress_state, f"Step {step + 1} | sim_time {sim_time:.2f}s: 完成任务 -> {completed_now}")

        if callable(on_post_step):
            on_post_step(step + 1, env, state)
            refresh_agent_targets(env, state)
            _sync_navigation_targets(env, state, step_completed=step + 1)
            completed_after_event = promote_completed_tasks(env, state)
            if completed_after_event:
                sim_time = format_sim_time(step + 1, state)
                _progress_log(progress_state, f"Step {step + 1} | sim_time {sim_time:.2f}s: 事件后完成任务 -> {completed_after_event}")

        for agent, info in infos.items():
            reason = info.get("last_collision_reason")
            if reason:
                collision_count += 1
                _progress_log(progress_state, f"[Collision][step={step + 1}] {agent}: {reason}")

        update_discovered_hemispheres(env, state)

        if render_after_sim and (
            step == 0
            or ((step + 1) % max(int(args.eval_render_every), 1) == 0)
        ):
            render_snapshots.append(
                _make_eval_render_snapshot(env, step + 1, history, render_meta=build_render_meta(state))
            )

        avg_decision_ms_live = (
            1000.0 * policy_predict_time_total / float(policy_predict_count_total)
            if policy_predict_count_total > 0
            else 0.0
        )
        _render_progress_line(
            progress_state,
            env,
            state,
            step_completed=step + 1,
            total_steps=args.max_steps,
            collision_count=collision_count,
            avg_decision_ms=avg_decision_ms_live,
            started_at=progress_started_at,
        )

        if callable(is_complete_fn):
            scene_done = bool(is_complete_fn(step + 1, env, state))
        else:
            scene_done = scenario_is_complete(state)
        if scene_done:
            _progress_log(progress_state, "所有任务均已完成")
            if render_after_sim:
                render_snapshots.append(
                    _make_eval_render_snapshot(env, step + 1, history, render_meta=build_render_meta(state))
                )
            break

        if all(truncations.get(agent, False) or terminations.get(agent, False) for agent in env.possible_agents):
            _progress_log(progress_state, "环境已达到最大步数")
            if render_after_sim:
                render_snapshots.append(
                    _make_eval_render_snapshot(env, step + 1, history, render_meta=build_render_meta(state))
                )
            break

    _finish_progress_line(progress_state)
    avg_decision_ms = (
        1000.0 * policy_predict_time_total / float(policy_predict_count_total)
        if policy_predict_count_total > 0
        else 0.0
    )
    print(
        "底层策略网络决策耗时: "
        f"avg={avg_decision_ms:.3f} ms/decision, "
        f"count={policy_predict_count_total}"
    )

    frames = []
    gif_path = ""
    final_image_path = ""
    if render_after_sim and render_snapshots:
        if args.eval_render_mode:
            plt.ion()
        else:
            plt.ioff()
        fig = plt.figure(figsize=(20, 8))
        ax_3d = fig.add_subplot(121, projection="3d")
        ax_top = fig.add_subplot(122)
        fig.subplots_adjust(right=0.78, wspace=0.18)

        replay_history = None
        for idx, snapshot in enumerate(render_snapshots):
            capture_frame = (
                idx == len(render_snapshots) - 1
                or (idx % max(int(args.eval_capture_every), 1) == 0)
            )
            should_draw_frame = bool(args.eval_render_mode) or bool(capture_frame and args.save_artifacts)
            if not should_draw_frame:
                continue
            replay_history = _apply_eval_render_snapshot(env, snapshot)
            _plot_waypoint_env(
                ax_3d,
                ax_top,
                env,
                snapshot["step"],
                history=replay_history,
                history_stride=max(int(args.eval_history_stride), 1),
                history_smooth_window=max(int(args.eval_history_smooth_window), 1),
                terrain_stride=max(int(args.eval_terrain_stride), 1),
            )
            if capture_frame and args.save_artifacts:
                frames.append(_capture_figure_frame(fig))
            if args.eval_render_mode:
                plt.pause(max(float(args.eval_pause), 0.001))

        if args.save_artifacts:
            gif_path, final_image_path = _save_eval_artifacts(
                frames,
                output_dir,
                run_name,
                args.eval_gif_fps,
            )
            if gif_path:
                print(f"评估过程 GIF 已保存到: {gif_path}")
            if final_image_path:
                print(f"评估最后一步图片已保存到: {final_image_path}")

        if args.eval_render_mode:
            plt.show()
        else:
            plt.close(fig)

    return {
        "setup_path": setup_path,
        "gif_path": gif_path,
        "final_image_path": final_image_path,
        "avg_decision_ms": avg_decision_ms,
    }
