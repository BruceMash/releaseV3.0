import numpy as np

import test_waypoint_task_scenarios_common as scenario_common
from test_waypoint_task_scenarios_common import (
    build_base_parser,
    build_tasks_from_specs,
    finalize_args,
    get_task_target_for_agent,
    refresh_agent_targets,
    resolve_assignment_plan,
    resolve_task_points,
    run_task_scenario,
)

# 单元素元组必须带逗号
_COLLAB_ORBIT_MIN_RADIUS = 1.2
_COLLAB_ORBIT_ANGULAR_SPEED = 0.20
_COLLAB_ORBIT_BREATHING_SCALE = 0.18
_COLLAB_ORBIT_VERTICAL_SCALE = 0.18

def _default_task_specs(agent_names, task_count):
    raise ValueError("collaborative_tasks 需要 assignment_provider 返回 task_specs，已禁用 legacy/自动协同分配策略。")


def _slot_spacing(env):
    return max(
        float(getattr(env, "goal_threshold", 1.0)) * 1.5,
        float(getattr(env, "near_goal_collision_free_radius", 1.0)) * 1.5,
        float(getattr(env, "collision_radius", 0.02)) * 20.0,
        1.2,
    )


def _segment_point_distance_2d(start_xy, end_xy, point_xy):
    start_xy = np.asarray(start_xy, dtype=np.float32)
    end_xy = np.asarray(end_xy, dtype=np.float32)
    point_xy = np.asarray(point_xy, dtype=np.float32)
    segment = end_xy - start_xy
    denom = float(np.dot(segment, segment))
    if denom <= 1e-8:
        return float(np.linalg.norm(point_xy - start_xy))
    t = float(np.clip(np.dot(point_xy - start_xy, segment) / denom, 0.0, 1.0))
    projection = start_xy + t * segment
    return float(np.linalg.norm(point_xy - projection))


def _resolve_collaborative_obstacle_centers(env, scenario_payload=None):
    centers = []
    obstacle_xy = scenario_payload.get("obstacle_xy") if isinstance(scenario_payload, dict) else None
    if isinstance(obstacle_xy, list):
        for point in obstacle_xy:
            if len(point) >= 2:
                centers.append(np.array([float(point[0]), float(point[1])], dtype=np.float32))

    if not centers:
        for obs in getattr(env, "obstacles", []):
            if hasattr(obs, "x") and hasattr(obs, "y"):
                centers.append(np.array([float(obs.x), float(obs.y)], dtype=np.float32))

    unique_centers = []
    for center in centers:
        if any(float(np.linalg.norm(center - existing)) <= 1e-6 for existing in unique_centers):
            continue
        unique_centers.append(center)
    return unique_centers


def _resolve_collab_slot_clearance_radius(args):
    obstacle_radius_cap = max(
        float(getattr(args, "terrain_radius_max", 0.0)),
        float(getattr(args, "lowland_cylinder_radius_max", 0.0)),
        0.0,
    )
    obstacle_buffer = max(float(getattr(args, "collab_slot_obstacle_buffer", 2.0)), 0.0)
    return obstacle_radius_cap + obstacle_buffer


def _build_slot_positions_for_anchor(env, task_center, ordered_agents, anchor, angle_step, radius):
    slot_positions = {}
    clipping_penalty = 0.0
    for rank, agent in enumerate(ordered_agents):
        angle = anchor + angle_step * rank
        raw_slot_xy = task_center[:2] + radius * np.array(
            [np.cos(angle), np.sin(angle)],
            dtype=np.float32,
        )
        clipped_slot_xy = raw_slot_xy.copy()
        clipped_slot_xy[0] = float(np.clip(clipped_slot_xy[0], 0.0, env.X))
        clipped_slot_xy[1] = float(np.clip(clipped_slot_xy[1], 0.0, env.Y))
        clipping_penalty += float(np.linalg.norm(raw_slot_xy - clipped_slot_xy))
        slot_positions[agent] = np.array(
            [clipped_slot_xy[0], clipped_slot_xy[1], float(np.clip(task_center[2], 0.0, env.Z))],
            dtype=np.float32,
        )
    return slot_positions, clipping_penalty


def _score_slot_layout(task_center, slot_positions, obstacle_centers, clearance_radius, clipping_penalty):
    if not obstacle_centers:
        return (0.0, 0.0, 0.0, -clipping_penalty)

    center_xy = np.asarray(task_center[:2], dtype=np.float32)
    min_layout_margin = float("inf")
    min_segment_margin = float("inf")
    min_slot_margin = float("inf")
    total_segment_margin = 0.0
    total_slot_margin = 0.0
    pair_count = 0

    for slot in slot_positions.values():
        slot_xy = np.asarray(slot[:2], dtype=np.float32)
        for obstacle_center in obstacle_centers:
            slot_dist = float(np.linalg.norm(slot_xy - obstacle_center))
            segment_dist = _segment_point_distance_2d(center_xy, slot_xy, obstacle_center)
            slot_margin = slot_dist - clearance_radius
            segment_margin = segment_dist - clearance_radius
            min_layout_margin = min(min_layout_margin, slot_margin, segment_margin)
            min_segment_margin = min(min_segment_margin, segment_margin)
            min_slot_margin = min(min_slot_margin, slot_margin)
            total_segment_margin += segment_margin
            total_slot_margin += slot_margin
            pair_count += 1

    avg_segment_margin = total_segment_margin / max(pair_count, 1)
    avg_slot_margin = total_slot_margin / max(pair_count, 1)
    return (
        min_layout_margin,
        min_segment_margin,
        min_slot_margin,
        avg_segment_margin,
        avg_slot_margin,
        -clipping_penalty,
    )


def _build_collaborative_slot_positions(env, task_center, assigned_agents, args=None, scenario_payload=None):
    assigned_agents = [str(agent) for agent in assigned_agents if agent in env.agent_positions]
    task_center = np.asarray(task_center, dtype=np.float32).copy()
    if not assigned_agents:
        return {}
    if len(assigned_agents) == 1:
        return {assigned_agents[0]: task_center}

    desired_spacing = _slot_spacing(env)
    angle_step = 2.0 * np.pi / len(assigned_agents)
    radius = max(
        desired_spacing / max(2.0 * np.sin(np.pi / len(assigned_agents)), 1e-6),
        0.75 * desired_spacing,
    )

    start_angles = []
    for idx, agent in enumerate(assigned_agents):
        delta_xy = np.asarray(env.agent_positions[agent], dtype=np.float32)[:2] - task_center[:2]
        if float(np.linalg.norm(delta_xy)) <= 1e-6:
            angle = angle_step * idx
        else:
            angle = float(np.arctan2(delta_xy[1], delta_xy[0]))
        start_angles.append((angle, idx, agent))
    start_angles.sort(key=lambda item: item[0])
    ordered_agents = [agent for _, _, agent in start_angles]

    base_anchor = float(
        np.arctan2(
            np.mean([np.sin(item[0]) for item in start_angles]),
            np.mean([np.cos(item[0]) for item in start_angles]),
        )
    )

    obstacle_centers = _resolve_collaborative_obstacle_centers(
        env,
        scenario_payload=scenario_payload,
    )
    if not obstacle_centers:
        slot_positions, _ = _build_slot_positions_for_anchor(
            env,
            task_center,
            ordered_agents,
            base_anchor,
            angle_step,
            radius,
        )
        return slot_positions

    clearance_radius = _resolve_collab_slot_clearance_radius(args)
    anchor_sample_count = max(48, len(assigned_agents) * 16)
    best_slot_positions = None
    best_score = None
    for sample_idx in range(anchor_sample_count):
        candidate_anchor = base_anchor + (2.0 * np.pi * sample_idx / anchor_sample_count)
        candidate_slots, clipping_penalty = _build_slot_positions_for_anchor(
            env,
            task_center,
            ordered_agents,
            candidate_anchor,
            angle_step,
            radius,
        )
        score = _score_slot_layout(
            task_center,
            candidate_slots,
            obstacle_centers,
            clearance_radius,
            clipping_penalty,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_slot_positions = candidate_slots

    return best_slot_positions or {}


def _attach_collaborative_slot_targets(env, tasks, args=None, scenario_payload=None):
    for task in tasks.values():
        assigned_agents = list(task.get("agents", []))
        if not assigned_agents:
            continue
        task["slot_positions"] = _build_collaborative_slot_positions(
            env,
            task["position"],
            assigned_agents,
            args=args,
            scenario_payload=scenario_payload,
        )


def _collab_arrived_agents(state, task_id):
    arrived = state.setdefault("collab_arrived_agents", {}).setdefault(str(task_id), [])
    return [str(agent) for agent in arrived]


def _set_collab_arrived_agents(state, task_id, agents):
    deduped = []
    for agent in agents:
        agent = str(agent)
        if agent not in deduped:
            deduped.append(agent)
    state.setdefault("collab_arrived_agents", {})[str(task_id)] = deduped


def _agent_is_at_collab_target(env, state, task_id, agent):
    if env.agent_reached_targets.get(agent, False):
        return True
    task = state["tasks"].get(task_id)
    if task is None or agent not in env.agent_positions:
        return False
    target = get_task_target_for_agent(task, agent)
    distance = float(np.linalg.norm(np.asarray(env.agent_positions[agent], dtype=np.float32) - target))
    threshold = max(float(getattr(env, "goal_threshold", 1.0)) * 1.15, 0.8)
    return distance <= threshold


def _compute_collab_orbit_target(env, state, task_id, agent, step_completed=None):
    task = state["tasks"][task_id]
    assigned_agents = [str(item) for item in task.get("agents", [])]
    agent_count = max(len(assigned_agents), 1)
    rank = assigned_agents.index(agent) if agent in assigned_agents else 0
    center = np.asarray(task["position"], dtype=np.float32).copy()
    slot_target = get_task_target_for_agent(task, agent)

    base_radius = max(
        _COLLAB_ORBIT_MIN_RADIUS,
        float(getattr(env, "goal_threshold", 1.0)) * 1.8,
        _slot_spacing(env) * 0.45,
    )
    step = float(step_completed if step_completed is not None else getattr(env, "step_cnt", 0))
    phase = (
        _COLLAB_ORBIT_ANGULAR_SPEED * step
        + 2.0 * np.pi * rank / agent_count
        + 0.35 * np.sin(0.07 * step + rank)
    )
    breathing = 1.0 + _COLLAB_ORBIT_BREATHING_SCALE * np.sin(0.11 * step + rank * 1.7)
    radius = base_radius * breathing

    orbit_xy = center[:2] + radius * np.array(
        [
            np.cos(phase) + 0.22 * np.cos(2.0 * phase + rank),
            np.sin(phase) + 0.22 * np.sin(3.0 * phase + rank * 0.5),
        ],
        dtype=np.float32,
    )
    orbit_z = float(slot_target[2]) + _COLLAB_ORBIT_VERTICAL_SCALE * base_radius * np.sin(
        0.15 * step + rank * 1.3
    )

    return np.array(
        [
            float(np.clip(orbit_xy[0], 0.0, env.X)),
            float(np.clip(orbit_xy[1], 0.0, env.Y)),
            float(np.clip(orbit_z, 0.0, env.Z)),
        ],
        dtype=np.float32,
    )


def _set_collab_orbit_target(env, state, task_id, agent, step_completed=None):
    orbit_target = _compute_collab_orbit_target(env, state, task_id, agent, step_completed)
    env.agent_targets[agent] = orbit_target.copy()
    env.agent_reached_targets[agent] = False
    env.agent_dones[agent] = False
    env.agent_initial_distances[agent] = float(
        np.linalg.norm(orbit_target - np.asarray(env.agent_positions[agent], dtype=np.float32))
    )

    state.setdefault("direct_task_targets", {})[agent] = orbit_target.copy()
    state.setdefault("global_plan_routes", {})[agent] = []
    state.setdefault("global_plan_indices", {})[agent] = 0
    state.setdefault("global_plan_subgoals", {})[agent] = orbit_target.copy()
    state.setdefault("global_plan_smoothed_subgoals", {})[agent] = orbit_target.copy()
    state.setdefault("global_plan_completed", {})[agent] = True
    state.setdefault("global_plan_status", {})[agent] = "collab_orbit"


def promote_collaborative_completed_tasks(env, state):
    completed_now = []
    inactive_agents = set(state.get("inactive_agents", set()))
    auto_complete_unassigned = bool(state.get("auto_complete_unassigned_tasks", True))
    step_completed = int(getattr(env, "step_cnt", 0))

    for task_id, task in state["tasks"].items():
        if task.get("completed", False) or not task.get("active", True):
            continue

        assigned_agents = [
            str(agent)
            for agent in task.get("agents", [])
            if str(agent) not in inactive_agents
        ]
        if not assigned_agents:
            if auto_complete_unassigned:
                task["completed"] = True
                completed_now.append(task_id)
                refresh_agent_targets(env, state)
            continue

        current_agents = []
        for agent in assigned_agents:
            queue = state["agent_queues"].get(agent, [])
            if queue and queue[0] == task_id:
                current_agents.append(agent)
        if set(current_agents) != set(assigned_agents):
            continue

        arrived_agents = _collab_arrived_agents(state, task_id)
        for agent in assigned_agents:
            if agent in arrived_agents:
                continue
            if _agent_is_at_collab_target(env, state, task_id, agent):
                arrived_agents.append(agent)
        _set_collab_arrived_agents(state, task_id, arrived_agents)

        if len(assigned_agents) > 1 and len(arrived_agents) < len(assigned_agents):
            for agent in arrived_agents:
                _set_collab_orbit_target(env, state, task_id, agent, step_completed=step_completed)
            continue

        if any(agent not in arrived_agents for agent in assigned_agents):
            continue

        task["completed"] = True
        completed_now.append(task_id)
        _set_collab_arrived_agents(state, task_id, [])
        for agent in assigned_agents:
            queue = state["agent_queues"].get(agent, [])
            if queue and queue[0] == task_id:
                queue.pop(0)
        refresh_agent_targets(env, state)

    return completed_now


def build_state(env, args, assignment_override, scenario_payload=None):
    agent_names = [str(agent) for agent in env.possible_agents]
    if not agent_names:
        agent_names = [str(agent) for agent in getattr(env, "agent_positions", {}).keys()]
    if not agent_names:
        agent_names = [f"agent_{idx}" for idx in range(max(int(args.n_agents), 0))]
    existing_positions = [
        np.asarray(env.agent_positions[agent], dtype=np.float32)
        for agent in agent_names
        if agent in env.agent_positions
    ]
    task_points = resolve_task_points(
        env,
        args,
        scenario_payload or {},
        existing_positions=existing_positions,
        count=args.task_count,
    )
    provider_task_positions = {
        f"task_{idx + 1:02d}": np.asarray(point, dtype=np.float32)
        for idx, point in enumerate(task_points)
    }
    provider_assignment_override = dict(assignment_override or {})
    for key in ("task_specs", "agent_queues"):
        provider_assignment_override.pop(key, None)
    plan = resolve_assignment_plan(
        agent_names=agent_names,
        task_count=args.task_count,
        assignment_override=provider_assignment_override,
        default_task_spec_builder=_default_task_specs,
        assignment_provider=getattr(args, "_assignment_provider", None),
        provider_context={
            "env": env,
            "args": args,
            "scenario_payload": scenario_payload,
            "task_positions": provider_task_positions,
            "phase": "initial",
            "state": None,
            "event": {
                "type": "initial",
                "scenario_name": "collaborative_tasks",
            },
            "scenario_name": "collaborative_tasks",
        },
    )
    provider_payload = plan.get("provider_payload", {})
    if not provider_payload or (
        provider_payload.get("task_specs") is None
        and provider_payload.get("agent_queues") is None
    ):
        raise ValueError("assignment_provider 未返回有效分配方案，collaborative_tasks 不再使用 legacy/自动协同分配策略。")
    task_specs = plan["task_specs"]
    task_points = task_points[: len(task_specs)]
    tasks = build_tasks_from_specs(task_points, task_specs)
    _attach_collaborative_slot_targets(
        env,
        tasks,
        args=args,
        scenario_payload=scenario_payload,
    )
    agent_queues = plan["agent_queues"]

    def _print_summary():
        print("协同任务分配结果：")
        for task_id, task in tasks.items():
            slot_summary = {
                agent: np.asarray(target, dtype=np.float32).round(3).tolist()
                for agent, target in task.get("slot_positions", {}).items()
            }
            print(
                f"  {task_id}: agents={task['agents']}, "
                f"center={np.asarray(task['position'], dtype=np.float32).round(3).tolist()}, "
                f"slots={slot_summary}"
            )
        for agent, queue in agent_queues.items():
            print(f"  {agent}: {' -> '.join(queue) if queue else 'finished'}")

    return {
        "tasks": tasks,
        "agent_queues": agent_queues,
        "inactive_agents": set(),
        "collab_arrived_agents": {},
        "event_log": [],
        "debug_task_transitions": True,
        "debug_task_ids": ["task_09", "task_10", "task_15"],
        "print_summary": _print_summary,
    }


def main():
    parser = build_base_parser("协同任务场景测试")
    args = finalize_args(parser)
    scenario_common.promote_completed_tasks = promote_collaborative_completed_tasks
    run_task_scenario(args, "collaborative_tasks", build_state)


if __name__ == "__main__":
    main()
