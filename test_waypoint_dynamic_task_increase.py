import numpy as np

import test_waypoint_task_scenarios_common as scenario_common
from test_waypoint_task_scenarios_common import (
    attach_collaborative_slot_targets,
    apply_active_agent_queue_mask,
    build_base_parser,
    build_tasks_from_specs,
    call_assignment_provider,
    filter_agent_queues,
    finalize_args,
    format_sim_time,
    merge_agent_queues,
    normalize_agent_queues,
    print_agent_task_sequences,
    remove_task_ids_from_queues,
    resolve_agent_queues_from_provider_payload,
    resolve_assignment_plan,
    resolve_task_points,
    run_task_scenario,
    scenario_is_complete,
    sync_task_assignments_from_queues,
)


def _default_task_specs(agent_names, task_count):
    raise ValueError("dynamic_task_increase 需要 assignment_provider 返回 task_specs，已禁用默认固定分配策略。")


def _provider_only_assignment_override(assignment_override):
    """Keep scenario/event fields, but ignore manual task assignment fields."""
    sanitized = dict(assignment_override or {})
    for key in (
        "task_specs",
        "agent_queues",
        "initial_agent_queues",
        "injected_agent_queues",
    ):
        sanitized.pop(key, None)
    return sanitized


def build_state(env, args, assignment_override, scenario_payload=None):
    agent_names = [str(agent) for agent in env.possible_agents]
    if not agent_names:
        agent_names = [str(agent) for agent in getattr(env, "agent_positions", {}).keys()]
    if not agent_names:
        agent_names = [f"agent_{idx}" for idx in range(max(int(args.n_agents), 0))]
    scenario_payload = scenario_payload or {}
    provider_assignment_override = _provider_only_assignment_override(assignment_override)
    dynamic_event = scenario_payload.get("dynamic_task_increase", {})
    if not isinstance(dynamic_event, dict):
        dynamic_event = {}
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
                "scenario_name": "dynamic_task_increase",
            },
            "scenario_name": "dynamic_task_increase",
        },
    )
    provider_payload = plan.get("provider_payload", {})
    if not provider_payload or (
        provider_payload.get("task_specs") is None
        and provider_payload.get("agent_queues") is None
        and provider_payload.get("initial_agent_queues") is None
        and provider_payload.get("injected_agent_queues") is None
    ):
        raise ValueError("assignment_provider 未返回有效分配方案，dynamic_task_increase 不再使用默认/固定分配策略。")
    task_specs = plan["task_specs"]
    task_points = task_points[: len(task_specs)]
    tasks = build_tasks_from_specs(task_points, task_specs)

    all_task_ids = list(tasks.keys())
    initial_count = max(
        min(int(dynamic_event.get("initial_task_count", args.initial_task_count)), len(all_task_ids)),
        0,
    )
    initial_task_ids = assignment_override.get("initial_task_ids")
    if initial_task_ids is None:
        initial_task_ids = provider_payload.get("initial_task_ids")
    if initial_task_ids is None:
        initial_task_ids = dynamic_event.get("initial_task_ids")
    if initial_task_ids is None:
        initial_task_ids = all_task_ids[:initial_count]
    initial_task_ids = [task_id for task_id in initial_task_ids if task_id in tasks]
    active_task_id_set = set(initial_task_ids)
    for task_id, task in tasks.items():
        task["active"] = task_id in active_task_id_set

    injected_task_ids = assignment_override.get("injected_task_ids")
    if injected_task_ids is None:
        injected_task_ids = provider_payload.get("injected_task_ids")
    if injected_task_ids is None:
        injected_task_ids = dynamic_event.get("injected_task_ids")
    if injected_task_ids is None:
        injected_task_ids = [task_id for task_id in all_task_ids if task_id not in active_task_id_set]
    injected_task_ids = [task_id for task_id in injected_task_ids if task_id in tasks]

    has_explicit_initial_agent_queues = False
    if provider_payload.get("initial_agent_queues") is not None:
        agent_queues = filter_agent_queues(
            agent_names,
            provider_payload.get("initial_agent_queues"),
            initial_task_ids,
        )
    else:
        agent_queues = filter_agent_queues(agent_names, plan["agent_queues"], initial_task_ids)
    agent_queues = normalize_agent_queues(agent_names, agent_queues)

    has_explicit_injected_agent_queues = False
    if provider_payload.get("injected_agent_queues") is not None:
        injected_agent_queues = filter_agent_queues(
            agent_names,
            provider_payload.get("injected_agent_queues"),
            injected_task_ids,
        )
    else:
        injected_agent_queues = filter_agent_queues(agent_names, plan["agent_queues"], injected_task_ids)
    injected_agent_queues = normalize_agent_queues(agent_names, injected_agent_queues)

    projected_agent_queues = normalize_agent_queues(
        agent_names,
        merge_agent_queues(agent_names, agent_queues, injected_agent_queues),
    )
    preview_state = {
        "tasks": tasks,
        "agent_queues": projected_agent_queues,
    }
    sync_task_assignments_from_queues(preview_state, agent_names=agent_names)
    attach_collaborative_slot_targets(
        env,
        tasks,
        args=args,
        scenario_payload=scenario_payload,
    )

    trigger_step = int(
        assignment_override.get(
            "trigger_step",
            dynamic_event.get("trigger_step", args.task_increase_step),
        )
    )

    state_ref = {}

    def _print_summary():
        current_state = state_ref.get("state")
        print("动态任务增加场景初始分配：")
        print(f"  触发步数: {trigger_step}")
        print(f"  初始任务: {initial_task_ids}")
        print(f"  增量任务: {injected_task_ids}")
        if current_state is None:
            for agent, queue in agent_queues.items():
                print(f"  {agent}: {' -> '.join(queue) if queue else 'idle'}")
        else:
            print_agent_task_sequences(
                current_state,
                title="  当前无人机任务序列：",
                agent_names=agent_names,
                include_inactive=True,
            )

    state = {
        "tasks": tasks,
        "agent_queues": agent_queues,
        "inactive_agents": set(),
        "event_log": [],
        "task_increase_triggered": False,
        "trigger_step": trigger_step,
        "injected_task_ids": list(injected_task_ids),
        "injected_agent_queues": injected_agent_queues,
        "has_explicit_initial_agent_queues": bool(has_explicit_initial_agent_queues),
        "has_explicit_injected_agent_queues": bool(has_explicit_injected_agent_queues),
        "auto_complete_unassigned_tasks": False,
        "_args": args,
        "_scenario_payload": dict(scenario_payload or {}),
        "_assignment_override": provider_assignment_override,
        "print_summary": _print_summary,
    }
    state_ref["state"] = state
    return state


def _active_task_ids(state):
    return [
        str(task_id)
        for task_id, task in state["tasks"].items()
        if task.get("active", True) and not task.get("completed", False)
    ]


def _apply_queue_override_for_task_ids(env, state, queue_map, allowed_task_ids):
    if not allowed_task_ids:
        return
    filtered_override = filter_agent_queues(env.possible_agents, queue_map, allowed_task_ids)
    state["agent_queues"] = remove_task_ids_from_queues(
        env.possible_agents,
        state["agent_queues"],
        allowed_task_ids,
    )
    for agent, extra_queue in filtered_override.items():
        state["agent_queues"].setdefault(agent, []).extend(extra_queue)


def _resolve_external_replan_queues(step, env, state):
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
                "type": "task_increase",
                "step": int(step),
                "activated_task_ids": list(state.get("injected_task_ids", [])),
            },
            "scenario_name": "dynamic_task_increase",
        },
    )
    return resolve_agent_queues_from_provider_payload(
        env.possible_agents,
        provider_payload,
        allowed_task_ids=_active_task_ids(state),
        preserve_collaborative_tasks=True,
    )


def on_post_step(step, env, state):
    if state["task_increase_triggered"] or step < state["trigger_step"]:
        return
    state["task_increase_triggered"] = True
    reassignment_before = {
        str(agent): [str(task_id) for task_id in queue]
        for agent, queue in state.get("agent_queues", {}).items()
    }
    for task_id in state["injected_task_ids"]:
        if task_id in state["tasks"]:
            state["tasks"][task_id]["active"] = True
    provider_agent_queues = _resolve_external_replan_queues(step, env, state)
    if provider_agent_queues is not None:
        state["agent_queues"] = provider_agent_queues
    else:
        raise ValueError(
            "assignment_provider 未返回任务突增后的重分配方案，已禁用手动 injected_agent_queues 兜底。"
        )

    state["agent_queues"] = normalize_agent_queues(env.possible_agents, state["agent_queues"])
    apply_active_agent_queue_mask(env.possible_agents, state)
    sync_task_assignments_from_queues(state, agent_names=env.possible_agents)
    attach_collaborative_slot_targets(
        env,
        state["tasks"],
        args=state.get("_args"),
        scenario_payload=state.get("_scenario_payload"),
    )
    unassigned_injected_task_ids = [
        str(task_id)
        for task_id in state["injected_task_ids"]
        if task_id in state["tasks"]
        and state["tasks"][task_id].get("active", True)
        and not state["tasks"][task_id].get("completed", False)
        and not state["tasks"][task_id].get("agents", [])
    ]
    if unassigned_injected_task_ids:
        raise ValueError(
            "assignment_provider 未覆盖新增任务，未分配任务: "
            + ", ".join(unassigned_injected_task_ids)
        )
    state["event_log"].append({
        "type": "task_increase",
        "step": int(step),
        "activated_tasks": list(state["injected_task_ids"]),
    })
    state["last_reassignment_panel"] = {
        "step": int(step),
        "before": reassignment_before,
        "after": {
            str(agent): [str(task_id) for task_id in queue]
            for agent, queue in state.get("agent_queues", {}).items()
        },
    }
    sim_time = format_sim_time(step, state)
    print(f"Step {step} | sim_time {sim_time:.2f}s: 新增任务已激活 -> {state['injected_task_ids']}")
    print_agent_task_sequences(
        state,
        title="场景变化后无人机任务序列：",
        agent_names=env.possible_agents,
        include_inactive=True,
    )


def is_complete(step, env, state):
    if not state.get("task_increase_triggered", False):
        return False
    return scenario_is_complete(state)


def main():
    parser = build_base_parser("动态任务增加场景测试")
    parser.add_argument("--initial_task_count", type=int, default=10, help="初始激活的任务数")
    parser.add_argument("--task_increase_step", type=int, default=300, help="新增任务的触发步数")
    args = finalize_args(parser)
    scenario_common.promote_completed_tasks = scenario_common.promote_collaborative_completed_tasks
    run_task_scenario(
        args,
        "dynamic_task_increase",
        build_state,
        on_post_step=on_post_step,
        is_complete_fn=is_complete,
    )


if __name__ == "__main__":
    main()
