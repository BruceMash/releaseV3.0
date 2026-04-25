"""
Torch 外部任务分配接口示例

这个示例文件演示如何把“外部任务分配算法”接到当前 release 目录下的
测试脚本中。它既支持初次规划，也支持事件触发后的重规划。

推荐调用方式：

python test_waypoint_collaborative_tasks.py ^
  --assignment_provider_module external_assignment_provider_torch_example.py ^
  --assignment_provider_callable plan_assignments ^
  --assignment_provider_model_path your_assignment_model.pt ^
  --assignment_provider_device cpu

说明：
1. 这里的 Torch 网络只是占位骨架，方便你替换成自己的真实模型。
2. 外部接口函数只需要返回 dict 或 None。
3. 最推荐直接返回 agent_queues；如果你更习惯任务到 agent 的映射，
   也可以改成返回 task_specs。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


_MODEL_CACHE: dict[tuple[str, str], nn.Module] = {}


class ExampleAssignmentPolicy(nn.Module):
    """
    这是一个最小占位网络。

    你后续接入自己的分配模型时，通常需要替换：
    1. 网络结构
    2. forward 的输入编码
    3. 输出解释方式
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _resolve_device(device_name: str) -> torch.device:
    device_name = str(device_name or "cpu").strip().lower()
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name or "cpu")


def _load_model(model_path: str, device_name: str) -> nn.Module:
    """
    懒加载并缓存模型，避免每次规划都重复读取权重。
    """

    device = _resolve_device(device_name)
    cache_key = (str(model_path or "").strip(), str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = ExampleAssignmentPolicy()
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _encode_agents(
    agent_names: list[str],
    env: Any | None,
    state: dict[str, Any] | None,
) -> tuple[torch.Tensor, list[str]]:
    """
    把当前 agent 状态编码成一个简单特征。

    这里只是示例：
    - 位置信息
    - 是否失活
    - 当前队列长度

    你后续应替换成适合自己模型训练时使用的编码方式。
    """

    features = []
    filtered_agents = []
    inactive_agents = {str(agent) for agent in (state or {}).get("inactive_agents", set())}

    for agent in agent_names:
        agent = str(agent)
        if agent in inactive_agents:
            continue

        pos = np.zeros(3, dtype=np.float32)
        if env is not None and hasattr(env, "agent_positions") and agent in getattr(env, "agent_positions", {}):
            pos = np.asarray(env.agent_positions[agent], dtype=np.float32)

        queue_len = 0.0
        if isinstance(state, dict):
            queue_len = float(len(state.get("agent_queues", {}).get(agent, [])))

        features.append(
            np.concatenate(
                [
                    pos.astype(np.float32),
                    np.array(
                        [
                            queue_len,
                            float(agent.endswith("0")),
                            float(agent.endswith("1")),
                            1.0,
                        ],
                        dtype=np.float32,
                    ),
                    np.zeros(9, dtype=np.float32),
                ]
            )
        )
        filtered_agents.append(agent)

    if not features:
        return torch.zeros((0, 16), dtype=torch.float32), []
    return torch.as_tensor(np.stack(features, axis=0), dtype=torch.float32), filtered_agents


def _collect_active_task_ids(
    task_count: int,
    state: dict[str, Any] | None,
    event: dict[str, Any] | None,
) -> list[str]:
    """
    获取当前应该参与分配的任务集合。
    """

    if isinstance(state, dict) and state.get("tasks"):
        return [
            str(task_id)
            for task_id, task in state["tasks"].items()
            if task.get("active", True) and not task.get("completed", False)
        ]

    if isinstance(event, dict):
        if event.get("type") == "task_increase":
            activated_task_ids = event.get("activated_task_ids")
            if isinstance(activated_task_ids, list) and activated_task_ids:
                return [str(task_id) for task_id in activated_task_ids]
        if event.get("type") == "agent_loss":
            remaining_task_ids = event.get("remaining_task_ids")
            if isinstance(remaining_task_ids, list) and remaining_task_ids:
                return [str(task_id) for task_id in remaining_task_ids]

    return [f"task_{idx + 1:02d}" for idx in range(max(int(task_count), 0))]


def _decode_agent_queues(
    logits: torch.Tensor,
    agent_names: list[str],
    task_ids: list[str],
) -> dict[str, list[str]]:
    """
    这里给一个最简单的“按分数排序后轮流分配任务”的示例解码方式。

    真实项目中，你可以替换为：
    - Hungarian
    - 贪心
    - 拍卖算法
    - 自定义多轮匹配
    """

    queues = {str(agent): [] for agent in agent_names}
    if logits.numel() == 0 or not agent_names or not task_ids:
        return queues

    # 这里用每个 agent 的特征均值作为一个稳定分数，只是为了演示完整流程。
    scores = logits.mean(dim=1).detach().cpu().numpy()
    ordered_agents = [
        agent for _, agent in sorted(
            zip(scores.tolist(), agent_names),
            key=lambda item: item[0],
            reverse=True,
        )
    ]
    if not ordered_agents:
        return queues

    for task_idx, task_id in enumerate(task_ids):
        agent = ordered_agents[task_idx % len(ordered_agents)]
        queues[str(agent)].append(str(task_id))
    return queues


def plan_assignments(
    agent_names: list[str],
    task_count: int,
    assignment_override: dict[str, Any] | None = None,
    env: Any | None = None,
    args: Any | None = None,
    scenario_payload: dict[str, Any] | None = None,
    phase: str = "initial",
    state: dict[str, Any] | None = None,
    event: dict[str, Any] | None = None,
    scenario_name: str | None = None,
) -> dict[str, Any] | None:
    """
    外部任务分配接口标准入口。

    当前 release 会调用这个函数，并期待你返回 dict 或 None。

    推荐返回：
    {
        "agent_queues": {
            "agent_0": ["task_01", "task_05"],
            "agent_1": ["task_02"],
            ...
        }
    }

    也可以扩展返回：
    - task_specs
    - initial_task_ids
    - injected_task_ids
    - initial_agent_queues
    - injected_agent_queues
    """

     # 当前任务状态
    current_tasks = state.get("tasks", {}) if isinstance(state, dict) else {}

    # 当前可用无人机
    inactive_agents = set(state.get("inactive_agents", set())) if isinstance(state, dict) else set()

    # 当前无人机位置
    current_agent_positions = {}
    if env is not None and hasattr(env, "agent_positions"):
        current_agent_positions = {
            str(agent): pos.tolist()
            for agent, pos in env.agent_positions.items()
            if str(agent) not in inactive_agents
        }

    # 当前有效任务的位置和状态
    current_task_info = {}
    for task_id, task in current_tasks.items():
        if not task.get("active", True):
            continue
        if task.get("completed", False):
            continue
        current_task_info[str(task_id)] = {
            "position": task["position"].tolist() if hasattr(task["position"], "tolist") else task["position"],
            "agents": list(task.get("agents", [])),
            "active": bool(task.get("active", True)),
            "completed": bool(task.get("completed", False)),
        }

    print("phase =", phase)
    print("event =", event)
    print("current_agent_positions =", current_agent_positions)
    print("current_task_info =", current_task_info)

    return None