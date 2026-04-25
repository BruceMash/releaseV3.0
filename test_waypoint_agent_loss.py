import numpy as np

import test_waypoint_task_scenarios_common as scenario_common
from test_waypoint_task_scenarios_common import (
    attach_collaborative_slot_targets,
    build_base_parser,
    build_tasks_from_specs,
    call_assignment_provider,
    finalize_args,
    normalize_agent_queues,
    remove_task_ids_from_queues,
    refresh_agent_targets,
    resolve_agent_queues_from_provider_payload,
    resolve_assignment_plan,
    resolve_task_points,
    run_task_scenario,
    scenario_is_complete,
    sync_task_assignments_from_queues,
)


def _default_task_specs(agent_names, task_count):
    raise ValueError("agent_loss 需要 assignment_provider 返回 task_specs，已禁用默认固定分配策略。")


def _provider_only_assignment_override(assignment_override):
    """Keep event fields, but ignore manual task assignment/reassignment fields."""
    sanitized = dict(assignment_override or {})
    for key in ("task_specs", "agent_queues", "reassignment"):
        sanitized.pop(key, None)
    return sanitized


def _filter_existing_agents(agent_names, candidate_agents):
    valid_agents = set(str(agent) for agent in agent_names)
    filtered = []
    for agent in candidate_agents:
        agent = str(agent)
        if agent in valid_agents and agent not in filtered:
            filtered.append(agent)
    return filtered


def _fallback_lost_agents(agent_names):
    agent_names = [str(agent) for agent in agent_names]
    if not agent_names:
        return []
    if len(agent_names) == 1:
        return [agent_names[0]]
    candidate_indexes = sorted(set([min(1, len(agent_names) - 1), max(len(agent_names) // 2, 0)]))
    return [agent_names[idx] for idx in candidate_indexes]


def build_state(env, args, assignment_override, scenario_payload=None):
    agent_names = [str(agent) for agent in env.possible_agents]
    if not agent_names:
        agent_names = [str(agent) for agent in getattr(env, "agent_positions", {}).keys()]
    if not agent_names:
        agent_names = [f"agent_{idx}" for idx in range(max(int(args.n_agents), 0))]
    scenario_payload = scenario_payload or {}
    provider_assignment_override = _provider_only_assignment_override(assignment_override)
    agent_loss_event = scenario_payload.get("agent_loss", {})
    if not isinstance(agent_loss_event, dict):
        agent_loss_event = {}
    existing_positions = [
        np.asarray(env.agent_positions[agent], dtype=np.float32)
        for agent in agent_names
        if agent in env.agent_positions
    ]
    task_points = resolve_task_points(
        env,
        args,
        scenario_payload,
        existing_positions=existing_positions,
        count=args.task_count,
    )
    provider_task_positions = {
        f"task_{idx + 1:02d}": np.asarray(point, dtype=np.float32)
        for idx, point in enumerate(task_points)
    }
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
                "scenario_name": "agent_loss",
            },
            "scenario_name": "agent_loss",
        },
    )
    provider_payload = plan.get("provider_payload", {})
    if not provider_payload or (
        provider_payload.get("task_specs") is None
        and provider_payload.get("agent_queues") is None
    ):
        raise ValueError("assignment_provider 未返回有效分配方案，agent_loss 不再使用默认/固定分配策略。")
    task_specs = plan["task_specs"]
    task_points = task_points[: len(task_specs)]
    tasks = build_tasks_from_specs(task_points, task_specs)
    agent_queues = normalize_agent_queues(agent_names, plan["agent_queues"])
    sync_task_assignments_from_queues(
        {
            "tasks": tasks,
            "agent_queues": agent_queues,
        },
        agent_names=agent_names,
    )
    attach_collaborative_slot_targets(
        env,
        tasks,
        args=args,
        scenario_payload=scenario_payload,
    )

    lost_agents = assignment_override.get("lost_agents")
    if not lost_agents:
        lost_agents = agent_loss_event.get("lost_agents")
    if not lost_agents:
        lost_agents = [item.strip() for item in args.lost_agents.split(",") if item.strip()]
    lost_agents = _filter_existing_agents(agent_names, lost_agents)
    if not lost_agents:
        lost_agents = _fallback_lost_agents(agent_names)

    reassignment = {}
    loss_step = int(
        assignment_override.get(
            "loss_step",
            agent_loss_event.get("loss_step", args.agent_loss_step),
        )
    )

    def _print_summary():
        print("无人机损失场景初始分配：")
        print(f"  损失步数: {loss_step}")
        print(f"  损失无人机: {lost_agents}")
        for agent, queue in agent_queues.items():
            print(f"  {agent}: {' -> '.join(queue) if queue else 'finished'}")

    return {
        "tasks": tasks,
        "agent_queues": agent_queues,
        "agent_types": dict(provider_payload.get("_platform_types", {})),
        "inactive_agents": set(),
        "event_log": [],
        "loss_triggered": False,
        "loss_step": loss_step,
        "lost_agents": list(lost_agents),
        "reassignment": copy_reassignment(reassignment),
        "auto_complete_unassigned_tasks": False,
        "_args": args,
        "_scenario_payload": dict(scenario_payload or {}),
        "_assignment_override": provider_assignment_override,
        "print_summary": _print_summary,
    }


def copy_reassignment(reassignment):
    return {
        str(agent): [str(task_id) for task_id in task_ids]
        for agent, task_ids in reassignment.items()
    }


def _dedupe_task_ids(task_ids):
    deduped = []
    seen = set()
    for task_id in task_ids:
        task_id = str(task_id)
        if task_id in seen:
            continue
        deduped.append(task_id)
        seen.add(task_id)
    return deduped


def _collect_unassigned_tasks_after_loss(state, future_inactive_agents):
    pending = []
    future_inactive_agents = {str(agent) for agent in future_inactive_agents}
    for task_id, task in state["tasks"].items():
        if task.get("completed", False) or not task.get("active", True):
            continue
        surviving_agents = [
            str(agent)
            for agent in task.get("agents", [])
            if str(agent) not in future_inactive_agents
        ]
        if not surviving_agents:
            pending.append(str(task_id))
    return pending


def _active_task_ids(state):
    return [
        str(task_id)
        for task_id, task in state["tasks"].items()
        if task.get("active", True) and not task.get("completed", False)
    ]


def _apply_explicit_reassignment(env, state, reassignment):
    assigned = set()
    override_task_ids = [
        str(task_id)
        for task_ids in reassignment.values()
        for task_id in task_ids
    ]
    if override_task_ids:
        state["agent_queues"] = remove_task_ids_from_queues(
            env.possible_agents,
            state["agent_queues"],
            override_task_ids,
        )

    for agent, task_ids in reassignment.items():
        if agent in state["inactive_agents"]:
            continue
        queue = state["agent_queues"].setdefault(agent, [])
        for task_id in task_ids:
            task_id = str(task_id)
            task = state["tasks"].get(task_id)
            if task is None or task.get("completed", False) or not task.get("active", True):
                continue
            if task_id not in queue:
                queue.append(task_id)
            assigned.add(task_id)
    return assigned


def _resolve_external_replan_queues(step, env, state, event_type, event_payload):
    args = state.get("_args")
    assignment_provider = getattr(args, "_assignment_provider", None) if args is not None else None
    provider_payload = call_assignment_provider(
        assignment_provider,
        agent_names=env.possible_agents,
        task_count=len(state.get("tasks", {})),
        assignment_override=state.get("_assignment_override", {}),
        provider_context={
            "env": env,
            "args": args,
            "scenario_payload": state.get("_scenario_payload", {}),
            "task_positions": {
                str(task_id): task["position"]
                for task_id, task in state.get("tasks", {}).items()
                if isinstance(task, dict) and "position" in task
            },
            "phase": "replan",
            "state": state,
            "event": {
                "type": str(event_type),
                "step": int(step),
                **dict(event_payload or {}),
            },
            "scenario_name": "agent_loss",
        },
    )
    resolved_queues = resolve_agent_queues_from_provider_payload(
        env.possible_agents,
        provider_payload,
        allowed_task_ids=_active_task_ids(state),
        preserve_collaborative_tasks=True,
    )
    if resolved_queues is None:
        return None

    inactive_agents = {str(agent) for agent in state.get("inactive_agents", set())}
    for agent in env.possible_agents:
        if str(agent) in inactive_agents:
            resolved_queues[str(agent)] = []
    return normalize_agent_queues(env.possible_agents, resolved_queues)


def on_post_step(step, env, state):
    if state["loss_triggered"] or step < state["loss_step"]:
        return

    state["loss_triggered"] = True
    lost_agents = [agent for agent in state["lost_agents"] if agent in env.possible_agents]
    future_inactive_agents = set(state.get("inactive_agents", set())) | set(lost_agents)
    remaining_tasks = []
    for agent in lost_agents:
        queue = state["agent_queues"].get(agent, [])
        for task_id in queue:
            task = state["tasks"].get(task_id)
            if task is not None and not task.get("completed", False):
                remaining_tasks.append(task_id)
        state["agent_queues"][agent] = []
        state["inactive_agents"].add(agent)
    remaining_tasks.extend(_collect_unassigned_tasks_after_loss(state, future_inactive_agents))
    remaining_tasks = _dedupe_task_ids(remaining_tasks)

    provider_agent_queues = _resolve_external_replan_queues(
        step,
        env,
        state,
        "agent_loss",
        {
            "lost_agents": list(lost_agents),
            "remaining_task_ids": list(remaining_tasks),
        },
    )
    if provider_agent_queues is not None:
        state["agent_queues"] = provider_agent_queues
    else:
        raise ValueError("assignment_provider 未返回无人机损失后的重分配方案，已禁用贪心/手动兜底分配。")

    state["agent_queues"] = normalize_agent_queues(env.possible_agents, state["agent_queues"])
    sync_task_assignments_from_queues(state, agent_names=env.possible_agents)
    attach_collaborative_slot_targets(
        env,
        state["tasks"],
        args=state.get("_args"),
        scenario_payload=state.get("_scenario_payload"),
    )
    unassigned_after_replan = _dedupe_task_ids(
        _collect_unassigned_tasks_after_loss(state, state["inactive_agents"])
    )
    if unassigned_after_replan:
        raise ValueError(
            "assignment_provider 未覆盖无人机损失后的剩余任务: "
            + ", ".join(unassigned_after_replan)
        )
    state["agent_queues"] = normalize_agent_queues(env.possible_agents, state["agent_queues"])
    sync_task_assignments_from_queues(state, agent_names=env.possible_agents)
    refresh_agent_targets(env, state)
    state["event_log"].append({
        "type": "agent_loss",
        "step": int(step),
        "lost_agents": list(lost_agents),
        "reassigned_tasks": list(remaining_tasks),
    })
    print(f"Step {step}: 无人机损失 -> {lost_agents}")
    if remaining_tasks:
        print(f"Step {step}: 已重新分配任务 -> {remaining_tasks}")


def is_complete(step, env, state):
    if not state.get("loss_triggered", False):
        return False
    return scenario_is_complete(state)


def main():
    parser = build_base_parser("无人机损失场景测试")
    parser.add_argument("--agent_loss_step", type=int, default=300, help="无人机损失触发步数")
    parser.add_argument(
        "--lost_agents",
        type=str,
        default="agent_2,agent_7",
        help="发生损失的无人机列表，逗号分隔",
    )
    args = finalize_args(parser)
    scenario_common.promote_completed_tasks = scenario_common.promote_collaborative_completed_tasks
    run_task_scenario(
        args,
        "agent_loss",
        build_state,
        on_post_step=on_post_step,
        is_complete_fn=is_complete,
    )


if __name__ == "__main__":
    main()
