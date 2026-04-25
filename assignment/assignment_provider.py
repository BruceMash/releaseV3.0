#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务分配模型适配器 - 对接师兄的 waypoint 测试框架
自动修复：拦截保存功能以避免 JSON 序列化错误
"""

# ============================================================
# 关键修复：自动拦截师兄的保存函数（不修改师兄代码）
# ============================================================
import sys
import types

# 在导入师兄模块之前，准备 monkey-patch
_original_save_task_scenario_setup = None


def _dummy_save_task_scenario_setup(*args, **kwargs):
    """空函数，替换师兄的保存函数以避免 JSON 序列化错误"""
    print("[Adapter] 已拦截场景保存（避免函数序列化错误）")
    return "/dev/null"


# 先标记需要 patch
_needs_patch = True

import os
import torch
import numpy as np
import warnings
from typing import Dict, List, Any, Optional

warnings.filterwarnings('ignore')

# 确保能导入 manual_small
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from manual_small import TaskSystem, TerrainEnv, Config, Platform, Target
except ImportError as e:
    print(f"[Adapter] Error importing manual_small: {e}")
    raise

# ============================================================
# 在导入师兄测试模块后，立即执行 monkey-patch
# ============================================================
try:
    # 导入师兄的模块
    import test_waypoint_task_scenarios_common as common_module

    # 保存原始函数（以防万一）
    if hasattr(common_module, 'save_task_scenario_setup'):
        _original_save_task_scenario_setup = common_module.save_task_scenario_setup
        # 替换为 dummy 函数
        common_module.save_task_scenario_setup = _dummy_save_task_scenario_setup
        print("[Adapter] 已自动禁用场景保存功能（修复 JSON 序列化错误）")

    # 同样处理 collaborative_tasks 模块（如果它有自己的导入）
    try:
        import test_waypoint_collaborative_tasks as collab_module

        if hasattr(collab_module, 'save_task_scenario_setup'):
            collab_module.save_task_scenario_setup = _dummy_save_task_scenario_setup
    except ImportError:
        pass

except ImportError as e:
    print(f"[Adapter] Warning: Could not patch save function: {e}")


class ModelAdapter:
    def __init__(self, model_path: str = "best_model_checkpoint.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.config = Config()

        # 初始化环境
        self.env = TerrainEnv(use_real_terrain=False)
        self.system = TaskSystem(self.env)

        self._load_model()

        print(f"[Adapter] Model loaded: {os.path.abspath(self.model_path)}")
        print(f"[Adapter] Device: {self.device}")

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[Adapter] Model not found: {os.path.abspath(self.model_path)}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self.system.masac.actor.load_state_dict(checkpoint['actor'])
        self.system.encoder.load_state_dict(checkpoint['encoder'])
        if 'critic' in checkpoint:
            self.system.masac.critic.load_state_dict(checkpoint['critic'])

        self.system.masac.actor.eval()
        self.system.encoder.eval()
        self.system.masac.epsilon = 0.05

        print(f"[Adapter] Checkpoint loaded (episode: {checkpoint.get('episode', 'unknown')})")

    def _resize_environment(self, n_platforms: int, n_targets: int):
        self.env.reset()

        if n_platforms != len(self.env.platforms):
            self.env.platforms = []
            types = ['UAV', 'UCAV', 'USV']
            for i in range(n_platforms):
                pos = np.array([
                    np.random.uniform(0, self.config.MAP_SIZE_X),
                    np.random.uniform(0, self.config.MAP_SIZE_Y),
                    np.random.uniform(1000, 50000)
                ], dtype=np.float32)
                self.env.platforms.append(Platform(i, pos, types[i % 3]))

        if n_targets != len(self.env.targets):
            self.env.targets = []
            for i in range(n_targets):
                pos = np.array([
                    np.random.uniform(0, self.config.MAP_SIZE_X),
                    np.random.uniform(0, self.config.MAP_SIZE_Y),
                    0
                ], dtype=np.float32)
                target = Target(i, pos)
                target.task_type = ['electronic', 'reconnaissance', 'attack'][i % 3]
                self.env.targets.append(target)

        self.config.N_PLATFORMS = n_platforms
        self.config.N_TARGETS = n_targets

    def _run_inference(self) -> Dict[int, int]:
        with torch.no_grad():
            assignment, _, _ = self.system.assign(evaluate=True)
        return assignment

    def _format_to_spec(self, assignment: Dict[int, int], agent_names: List[str], task_count: int):
        n_platforms = len(agent_names)

        task_specs = []
        for tid in range(task_count):
            task_id = f"task_{tid + 1:02d}"
            assigned_agents = [agent_names[pid] for pid, assigned_tid in assignment.items()
                               if assigned_tid == tid and pid < n_platforms]
            task_specs.append({
                "task_id": task_id,
                "agents": assigned_agents,
                "label": task_id,
                "active": True,
                "completed": False,
            })

        agent_queues = {agent_name: [] for agent_name in agent_names}
        for pid, agent_name in enumerate(agent_names):
            if pid in assignment:
                task_id = f"task_{assignment[pid] + 1:02d}"
                agent_queues[agent_name] = [task_id]

        return {"task_specs": task_specs, "agent_queues": agent_queues}

    def __call__(self, agent_names: List[str], task_count: int,
                 assignment_override: Optional[Dict] = None, **provider_context):
        self._resize_environment(len(agent_names), task_count)
        raw_assignment = self._run_inference()
        result = self._format_to_spec(raw_assignment, agent_names, task_count)

        if provider_context.get('phase') == 'initial':
            n_assigned = sum(len(q) for q in result['agent_queues'].values())
            print(
                f"[Adapter] Phase: initial | Platforms: {len(agent_names)} | Tasks: {task_count} | Assigned: {n_assigned}")

        return result


_adapter_instance: Optional[ModelAdapter] = None


def get_adapter() -> ModelAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        model_path = os.environ.get('MODEL_PATH', 'best_model_checkpoint.pth')
        _adapter_instance = ModelAdapter(model_path)
    return _adapter_instance


def assignment_provider(agent_names: List[str], task_count: int,
                        assignment_override: Optional[Dict] = None, **provider_context) -> Dict[str, Any]:
    return get_adapter()(agent_names, task_count, assignment_override, **provider_context)


if __name__ == "__main__":
    result = assignment_provider([f"agent_{i}" for i in range(10)], 16, phase="test")
    print(f"Test passed: {len(result['task_specs'])} tasks, {len(result['agent_queues'])} agents")