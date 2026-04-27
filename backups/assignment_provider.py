"""
assignment_provider.py
智能空面协同任务博弈对抗规划系统 - 对接程序

功能：
1. 解析路径规划端场景文件(scenario_A.json)，获取平台/目标位置与数量
2. 自动补全缺失属性（平台类型、载弹量、航程、目标价值、时间窗口等）
3. 加载训练好的最优模型(best_model_checkpoint.pth)
4. 输出标准格式的任务分配结果与3D场景图
5. 支持突发事件触发的快速重分配，并生成前后对比表

调用方式：
  - 作为模块被路径规划端程序调用：提供 plan_assignments() 接口
  - 独立运行测试：python assignment_provider.py --standalone
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 无界面后端，避免显示问题
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# ============================================================
# 1. 导入主程序模块（确保 bibi.py 与本文件在同目录）
# ============================================================
try:
    from backups.bibi import (
        Config, TerrainEnv, TaskSystem, Platform, Target, Threat,
        TerrainTrainer, DynamicReallocator, ConvergenceMonitor,
        GATEncoder, Actor, Critic, MASAC, RouteEstimator, RoutePlanner
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backups.bibi import (
        Config, TerrainEnv, TaskSystem, Platform, Target,
        TerrainTrainer, DynamicReallocator, ConvergenceMonitor,
        GATEncoder, Actor, Critic, MASAC, RouteEstimator, RoutePlanner
    )

warnings.filterwarnings('ignore')

# ============================================================
# 2. 可修改的默认配置区（根据实际场景调整）
# ============================================================
DEFAULT_MODEL_PATH = "best_model_checkpoint.pth"
DEFAULT_SCENARIO_PATH = "../scenario_A_collaborate.json"

# 平台类型分配规则（10平台示例；如平台数变化会自动循环）
DEFAULT_PLATFORM_TYPES = ['侦察', '侦察', '侦察',
                          '打击', '打击', '察打一体',
                          '察打一体', '察打一体', '察打一体', '察打一体']

# 目标类型分配规则（16目标示例；侦察在前，打击居中，察打一体在后）
DEFAULT_TARGET_TYPES = ['侦察'] * 6 + ['察打一体'] * 10


# 坐标单位转换：scenario_A.json 中坐标为 km，需转为 m（×1000）
COORD_SCALE = 1000.0
def _norm_position(pos) -> list[float]:
    """统一坐标转换为米（支持 np.ndarray / list / tuple）"""
    # 1. 转成 Python 列表
    if isinstance(pos, np.ndarray):
        pos = pos.tolist()
    elif isinstance(pos, tuple):
        pos = list(pos)
    pos = [float(v) for v in pos]

    # 2. 启发式判断单位：如果 XY 坐标绝对值都 < 1000，认为是 km，需转 m
    #    （100km 地图内，以米为单位时坐标通常是几千到十几万）
    if len(pos) >= 2 and max(abs(pos[0]), abs(pos[1])) < 1000.0:
        return [v * COORD_SCALE for v in pos]
    return pos
# ============================================================
# 3. 全局状态（保持环境上下文，支持快速重分配）
# ============================================================
_global_state = {
    'env': None,           # TerrainEnv 实例
    'system': None,        # TaskSystem 实例
    'model_loaded': False, # 是否已加载模型权重
    'last_n_platforms': 0, # 上次平台数（用于检测变化）
    'last_n_targets': 0,   # 上次目标数
    'last_assignment': None, # 上次分配结果（用于对比）
    'all_agent_names': [], # 完整的原始平台列表（确保损失平台也在对比表中显示）
}


# ============================================================
# 4. 辅助函数：场景解析 / 模型加载 / 格式化 / 可视化
# ============================================================

def _assign_platform_types(n: int) -> List[str]:
    """按百分比分配平台类型：30%%侦察, 20%%打击, 50%%察打一体（取整，总和=n）"""
    if n <= 0:
        return []
    n_recon = round(n * 0.30)
    n_strike = round(n * 0.20)
    n_multi = n - n_recon - n_strike
    while n_recon + n_strike + n_multi != n:
        if n_recon + n_strike + n_multi < n:
            n_multi += 1
        else:
            if n_strike > 0:
                n_strike -= 1
            elif n_recon > 0:
                n_recon -= 1
            else:
                n_multi -= 1
    return ['侦察'] * n_recon + ['打击'] * n_strike + ['察打一体'] * n_multi


def _assign_target_types(n: int) -> List[str]:
    """按百分比分配目标类型：40%%侦察, 60%%察打一体（取整，总和=n）"""
    if n <= 0:
        return []
    n_recon = round(n * 0.40)
    n_multi = n - n_recon
    while n_recon + n_multi != n:
        if n_recon + n_multi < n:
            n_multi += 1
        else:
            if n_recon > 0:
                n_recon -= 1
            else:
                n_multi -= 1
    return ['侦察'] * n_recon + ['察打一体'] * n_multi


def _get_platform_type_by_index(idx: int, total: int) -> str:
    """根据平台在原始场景中的固定索引返回类型，保持与 _assign_platform_types 一致"""
    types = _assign_platform_types(total)
    if 0 <= idx < len(types):
        return types[idx]
    return '察打一体'


def _get_target_type_by_index(idx: int, total: int) -> str:
    """根据目标在原始场景中的固定索引返回类型，保持与 _assign_target_types 一致"""
    types = _assign_target_types(total)
    if 0 <= idx < len(types):
        return types[idx]
    return '察打一体'


def _parse_scenario(scenario_payload: dict) -> Tuple[List[dict], List[dict], list, list]:
    """
    从 scenario_A.json 结构解析为 bibi.py 需要的 platform_configs / target_configs。
    同时返回 agent_names 列表和 task_ids 列表。
    平台/目标类型按百分比动态分配：
      - 平台：30%%侦察, 20%%打击, 50%%察打一体
      - 目标：40%%侦察, 60%%察打一体
    """
    start_positions = scenario_payload.get('start_positions', [])
    task_positions = scenario_payload.get('task_positions', [])

    n_platforms = len(start_positions)
    n_targets = len(task_positions)

    platform_type_list = _assign_platform_types(n_platforms)
    target_type_list = _assign_target_types(n_targets)

    # 构建平台配置
    platform_configs = []
    for i, pos in enumerate(start_positions):
        ptype = platform_type_list[i]
        platform_configs.append({
            'type': ptype,
            'payload': None,
            'max_range': None,
            'speed': None,
            'pos': [pos[0] * COORD_SCALE,
                    pos[1] * COORD_SCALE,
                    pos[2] * COORD_SCALE]
        })

    # 构建目标配置
    target_configs = []
    recon_count = 0
    multi_count = 0
    for i, pos in enumerate(task_positions):
        ttype = target_type_list[i]
        if ttype == '侦察':
            tw_start, tw_end = 0, 1000
            req_payload = 1
            value = 0.5 + recon_count * 0.08
            recon_count += 1
        else:  # 察打一体
            tw_start = 100 + multi_count * 50
            tw_end = tw_start + 700
            req_payload = 2
            value = 0.8 + multi_count * 0.04
            multi_count += 1

        target_configs.append({
            'type': ttype,
            'value': round(value, 2),
            'pos': [pos[0] * COORD_SCALE,
                    pos[1] * COORD_SCALE,
                    pos[2] * COORD_SCALE],
            'req_payload': req_payload,
            'time_win_start': tw_start,
            'time_win_end': tw_end,
            'predecessors': []
        })

    agent_names = [f"agent_{i}" for i in range(n_platforms)]
    task_ids = [f"task_{i + 1:02d}" for i in range(n_targets)]

    return platform_configs, target_configs, agent_names, task_ids


def _load_model(system: TaskSystem, model_path: str = DEFAULT_MODEL_PATH) -> bool:
    """加载训练好的最优模型到 system"""
    if not os.path.exists(model_path):
        print(f"⚠️  警告: 模型文件 {model_path} 不存在，将使用未训练模型（随机分配）")
        return False

    checkpoint = torch.load(model_path, map_location=system.masac.device, weights_only=False)
    system.encoder.load_state_dict(checkpoint['encoder'])
    system.masac.actor.load_state_dict(checkpoint['actor'])

    # Critic 在推理时非必需，但尽量加载
    if 'critic' in checkpoint:
        try:
            system.masac.critic.load_state_dict(checkpoint['critic'])
        except RuntimeError:
            pass  # 维度不匹配时忽略

    print(f"✅ 已加载最优模型: {model_path}")
    return True


def _format_agent_queues(assignment: Dict[str, List[int]],
                         agent_names: List[str],
                         task_ids: List[str]) -> Dict[str, List[str]]:
    """将内部 {agent_name: [target_id, ...]} 转为路径规划端要求的 agent_queues 格式"""
    agent_queues = {agent: [] for agent in agent_names}
    for agent_name, tids in assignment.items():
        for tid in tids:
            task_name = task_ids[tid] if tid < len(task_ids) else f"task_{tid + 1:02d}"
            agent_queues.setdefault(agent_name, []).append(task_name)
    return agent_queues


def _format_assignment_text(geo_assignment: Dict[str, List[int]],
                            env: TerrainEnv,
                            elapsed_ms: float = 0.0,
                            task_ids: List[str] = None,
                            agent_names: List[str] = None) -> str:
    """
    生成精简文本输出（基于地理目标）：
      - 每平台仅输出一行任务序列
      - 末尾汇总航程代价、时间代价、任务收益
    """
    lines = []
    lines.append("=" * 60)
    lines.append("分配结果：")
    lines.append("=" * 60)

    total_cost = 0.0          # 总航程(m)
    total_time = 0.0          # 最大完成时间(s)
    total_value = 0.0         # 任务收益
    assigned_geo_targets = set()

    # 构建 agent_name -> 平台对象 的映射（应对损失后环境PID错位）
    name_to_platform = {}
    if agent_names:
        for i, plat in enumerate(env.platforms):
            if i < len(agent_names):
                name_to_platform[agent_names[i]] = plat

    for agent_name, geo_tids in sorted(geo_assignment.items()):
        p = name_to_platform.get(agent_name)
        if p is None:
            continue

        # 使用传入的 task_ids 列表映射地理目标索引到原始 task_id
        if task_ids:
            task_names = [task_ids[tid] if tid < len(task_ids) else f"task_{tid + 1:02d}" for tid in geo_tids]
        else:
            task_names = [f"task_{tid + 1:02d}" for tid in geo_tids]
        lines.append(f"{agent_name} -> {len(geo_tids)} 个任务: {task_names}")

        prev_pos = p.position.copy()
        platform_dist = 0.0
        platform_time = 0.0

        for geo_tid in geo_tids:
            # 通过 geo_targets 映射回内部节点计算距离
            # （同一地理目标下的侦察/打击节点位置相同，取第一个即可）
            if 0 <= geo_tid < len(env.geo_targets):
                node_ids = env.geo_targets[geo_tid]['task_ids']
                rep_node_id = node_ids[0]
                if 0 <= rep_node_id < len(env.targets):
                    t = env.targets[rep_node_id]
                    dist = float(np.linalg.norm(prev_pos - t.position))
                    seg_time = dist / max(p.speed, 1e-6)
                    platform_dist += dist
                    platform_time += seg_time
                    prev_pos = t.position.copy()

                # 累加该地理目标下所有内部节点的价值（侦察+打击）
                geo_value = sum(env.targets[nid].value
                                for nid in node_ids
                                if 0 <= nid < len(env.targets))
                if geo_tid not in assigned_geo_targets:
                    total_value += geo_value
                    assigned_geo_targets.add(geo_tid)

        total_cost += platform_dist
        total_time = max(total_time, platform_time)

    # 未分配地理目标统计（只统计 0~15）
    all_geo_targets = set(range(len(env.geo_targets)))
    unassigned = sorted(all_geo_targets - assigned_geo_targets)

    lines.append("\n" + "=" * 60)
    lines.append(f"平台航程代价: {total_cost / 1000:.1f} km")
    lines.append(f"任务收益: {total_value:.2f}")
    lines.append(f"分配计算耗时: {elapsed_ms:.1f} ms")

    if unassigned:
        if task_ids:
            unassigned_names = [task_ids[t] if t < len(task_ids) else f"task_{t + 1:02d}" for t in unassigned]
        else:
            unassigned_names = [f"task_{t + 1:02d}" for t in unassigned]
        lines.append(f"⚠️ 未完全覆盖！未分配目标: {unassigned_names}")
    else:
        lines.append(f"✅ 完全覆盖！所有 {len(env.geo_targets)} 个目标已分配")

    lines.append("=" * 60)
    return "\n".join(lines)


def _save_3d_assignment(env: TerrainEnv,
                        assignment: Dict[int, List[int]],
                        filename: str = "assignment_3d.png",
                        geo_assignment: Dict[str, List[int]] = None,
                        agent_names: List[str] = None):
    """生成3D任务分配场景图（基于地理目标 task_01~task_16 显示）"""

    # ========== 字体与配色 ==========
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                       'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    CMAP = {
        '侦察': {
            'plat': '#2980b9', 'tgt': '#3498db',
            'line': '#5dade2', 'arr': '#1a5276',
        },
        '打击': {
            'plat': '#d4ac0d', 'tgt': '#f1c40f',
            'line': '#f4d03f', 'arr': '#9a7d0a',
        },
        '察打一体': {
            'plat': '#c0392b', 'tgt': '#e74c3c',
            'line': '#ec7063', 'arr': '#641e16',
        }
    }

    # ========== 如果没有传入 geo_assignment，在函数内部构建 ==========
    if geo_assignment is None:
        node_to_geo = {}
        for geo_id, gt in enumerate(env.geo_targets):
            for nid in gt['task_ids']:
                node_to_geo[nid] = geo_id
        geo_assignment = {}
        for pid, node_ids in assignment.items():
            seen = []
            for nid in node_ids:
                if nid in node_to_geo:
                    gid = node_to_geo[nid]
                    if gid not in seen:
                        seen.append(gid)
            agent_name = agent_names[pid] if agent_names and pid < len(agent_names) else f"agent_{pid}"
            geo_assignment[agent_name] = seen

    # 计算地理目标层面的统计
    total_geo_tasks = sum(len(tids) for tids in geo_assignment.values())
    unique_geo_tasks = len(set(tid for tids in geo_assignment.values() for tid in tids))
    n_platforms_assigned = len([name for name, tids in geo_assignment.items() if len(tids) > 0])

    # 建立节点→地理目标映射（用于3D图和俯视图标注）
    node_to_geo = {}
    for geo_id, gt in enumerate(env.geo_targets):
        for nid in gt['task_ids']:
            node_to_geo[nid] = geo_id

    # ========== 布局：左侧3D大图 | 右上序列 | 右下俯视 ==========
    fig = plt.figure(figsize=(26, 14))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, width_ratios=[3, 1.1],
                  height_ratios=[1, 1], wspace=0.12, hspace=0.22)

    ax = fig.add_subplot(gs[:, 0], projection='3d')      # 左侧占满2行
    ax2 = fig.add_subplot(gs[0, 1])                      # 右上：序列
    ax3 = fig.add_subplot(gs[1, 1])                      # 右下：俯视

    assigned_set = set(tid for tids in assignment.values() for tid in tids)

    # ==================== 左侧：3D场景（保持26个内部节点绘制，不做大改） ====================
    # ---------- 1. 简化地形 ----------
    x_terrain = np.linspace(0, Config.MAP_SIZE_X, 60)
    y_terrain = np.linspace(0, Config.MAP_SIZE_Y, 60)
    X, Y = np.meshgrid(x_terrain, y_terrain)
    Z = np.random.uniform(0, 30, X.shape)
    ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.25,
                    rstride=2, cstride=2, zorder=1, linewidth=0)

    for threat in env.threats:
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi / 2, 20)
        x = threat.center[0] + threat.radius * np.outer(np.cos(u), np.sin(v))
        y = threat.center[1] + threat.radius * np.outer(np.sin(u), np.sin(v))
        z = 0 + threat.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='red', alpha=0.12,
                        rstride=2, cstride=2, zorder=2, linewidth=0)
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(threat.center[0] + threat.radius * np.cos(theta),
                threat.center[1] + threat.radius * np.sin(theta),
                np.zeros_like(theta), color='red', alpha=0.6,
                linewidth=0.5, zorder=3)

    # ---------- 3. 平台 ----------
    for p in env.platforms:
        c = CMAP.get(p.type, CMAP['侦察'])
        agent_name = agent_names[p.id] if agent_names and p.id < len(agent_names) else f"agent_{p.id}"
        n_tasks = len(geo_assignment.get(agent_name, []))
        ax.scatter(*p.position, c=c['plat'], marker='o', s=500,
                   edgecolors='black', linewidth=2, alpha=0.95, zorder=10)
        ax.text(p.position[0], p.position[1], p.position[2] + 8000,
                f'P{p.id}({p.type})\n{n_tasks}tasks',
                fontsize=9, fontweight='bold', zorder=15,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=c['plat'], alpha=0.9, linewidth=2))

    # ---------- 4. 目标（仍绘制26个内部节点，但标注改为地理目标编号，避免重叠） ----------
    annotated_geo_3d = set()
    for t in env.targets:
        if t.id in assigned_set:
            c = CMAP.get(t.task_type, CMAP['侦察'])
            alpha, size = 0.95, 250 + int(t.value * 250)
        else:
            c = {'tgt': '#bbbbbb'}
            alpha, size = 0.35, 150
        marker = Target.MARKERS.get(t.task_type, 'o')
        ax.scatter(*t.position, c=c['tgt'], marker=marker, s=size,
                   edgecolors='black', linewidth=1.5, alpha=alpha, zorder=10)

        # 标注改为地理目标编号（同一位置只标一次）
        if t.id in node_to_geo:
            geo_id = node_to_geo[t.id]
            if geo_id not in annotated_geo_3d:
                ax.text(t.position[0], t.position[1], t.position[2] + 5000,
                        f'Task{geo_id+1:02d}', fontsize=9, fontweight='bold', zorder=15,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor=c['tgt'], alpha=0.8, linewidth=1.5))
                annotated_geo_3d.add(geo_id)

    # ---------- 5. 分配路径（仍按内部节点连线，保持颜色逻辑） ----------
    for pid, tids in assignment.items():
        if pid >= len(env.platforms):
            continue
        p = env.platforms[pid]
        prev_pos = p.position.copy()
        for idx, tid in enumerate(tids):
            if tid >= len(env.targets):
                continue
            t = env.targets[tid]
            c = CMAP.get(t.task_type, CMAP['侦察'])
            lw, ls = (3.5, '-') if idx == 0 else (2.5, '--')
            ax.plot([prev_pos[0], t.position[0]],
                    [prev_pos[1], t.position[1]],
                    [prev_pos[2], t.position[2]],
                    color=c['line'], linewidth=lw, linestyle=ls,
                    alpha=0.85, zorder=5)
            mid = (prev_pos + t.position) / 2
            direction = (t.position - prev_pos)
            norm = np.linalg.norm(direction) + 1e-6
            direction = direction / norm * 6000
            ax.quiver(mid[0], mid[1], mid[2],
                      direction[0], direction[1], direction[2],
                      color=c['arr'], arrow_length_ratio=0.35,
                      linewidth=2.5, zorder=8)
            prev_pos = t.position.copy()

    # ---------- 6. 3D坐标轴与标题（改为地理目标统计） ----------
    ax.set_xlim(0, Config.MAP_SIZE_X)
    ax.set_ylim(0, Config.MAP_SIZE_Y)
    ax.set_zlim(0, Config.MAP_HEIGHT)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)

    ax.set_title(f'智能空面协同任务分配 - 3D场景\n'
                 f'平台:{n_platforms_assigned} | 任务:{total_geo_tasks} | '
                 f'目标:{unique_geo_tasks}/{len(env.geo_targets)}',
                 fontsize=13, fontweight='bold', pad=15)

    # 3D图例（保持不变）
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CMAP['侦察']['plat'], markersize=11,
               label='侦察平台', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CMAP['打击']['plat'], markersize=11,
               label='打击平台', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CMAP['察打一体']['plat'], markersize=11,
               label='察打一体平台', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=CMAP['侦察']['tgt'], markersize=11,
               label='侦察目标', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w',
               markerfacecolor=CMAP['打击']['tgt'], markersize=11,
               label='打击目标', markeredgecolor='black'),
        Line2D([0], [0], color=CMAP['侦察']['line'], lw=3, label='侦察路径'),
        Line2D([0], [0], color=CMAP['打击']['line'], lw=3, label='打击路径'),
        Line2D([0], [0], color=CMAP['察打一体']['line'], lw=3, label='察打一体路径'),
        Line2D([0], [0], color='red', lw=0, marker='o', markersize=10,
               markerfacecolor='red', alpha=0.3, label='威胁区域'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.view_init(elev=30, azim=50)

    # ==================== 右上：任务序列面板（关键修改：基于地理目标） ====================
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    lines = []
    lines.append("【任务分配序列表】")
    lines.append("=" * 58)
    lines.append(f"{'平台':<8} {'类型':<6} {'任务链'}")
    lines.append("-" * 58)

    # 构建 env_pid -> agent_name 映射，方便找平台对象
    env_pid_to_name = {i: agent_names[i] for i in range(len(agent_names))} if agent_names else {}

    for agent_name in sorted(geo_assignment.keys()):
        # 找到该平台对应的环境对象
        p = None
        for plat in env.platforms:
            if env_pid_to_name.get(plat.id) == agent_name:
                p = plat
                break
        if p is None:
            continue

        geo_tids = geo_assignment[agent_name]
        if not geo_tids:
            continue
        task_chain = []
        for geo_tid in geo_tids:
            if 0 <= geo_tid < len(env.geo_targets):
                gt = env.geo_targets[geo_tid]
                task_chain.append(f"task_{geo_tid + 1:02d}({gt['type']})")
        chain_str = " → ".join(task_chain)
        lines.append(f"{agent_name:<10} {p.type:<6} {chain_str}")

    lines.append("=" * 58)
    all_geo = set(range(len(env.geo_targets)))
    assigned_geo = set(tid for tids in geo_assignment.values() for tid in tids)
    unassigned_geo = sorted(all_geo - assigned_geo)
    if unassigned_geo:
        lines.append(f"\n⚠️ 未分配: {[f'task_{t + 1:02d}' for t in unassigned_geo]}")
    else:
        lines.append(f"\n✅ 全部 {len(env.geo_targets)} 个目标已分配")

    ax2.text(0.05, 0.98, "\n".join(lines), transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa',
                       edgecolor='#2c3e50', linewidth=2, alpha=0.95))
    ax2.set_title("任务序列", fontsize=12, fontweight='bold', pad=10)

    # ==================== 右下：Z轴俯视图（X-Y平面） ====================
    # 平台
    for p in env.platforms:
        c = CMAP.get(p.type, CMAP['侦察'])
        ax3.scatter(p.position[0], p.position[1], c=c['plat'], marker='o',
                    s=300, edgecolors='black', linewidth=1.5, alpha=0.9, zorder=10)
        ax3.annotate(f'P{p.id}', (p.position[0], p.position[1]),
                     textcoords="offset points", xytext=(8, 8),
                     fontsize=9, fontweight='bold', color=c['plat'])

    # 目标（标注改为地理目标编号，同一位置只标一次）
    annotated_geo_top = set()
    for t in env.targets:
        if t.id in assigned_set:
            c = CMAP.get(t.task_type, CMAP['侦察'])
            alpha, size = 0.9, 180 + int(t.value * 180)
        else:
            c = {'tgt': '#bbbbbb'}
            alpha, size = 0.35, 100
        marker = Target.MARKERS.get(t.task_type, 'o')
        ax3.scatter(t.position[0], t.position[1], c=c['tgt'], marker=marker,
                    s=size, edgecolors='black', linewidth=1.2, alpha=alpha, zorder=9)

        if t.id in node_to_geo:
            geo_id = node_to_geo[t.id]
            if geo_id not in annotated_geo_top:
                ax3.annotate(f'Task{geo_id+1:02d}', (t.position[0], t.position[1]),
                             textcoords="offset points", xytext=(-12, -12),
                             fontsize=8, color='#333333')
                annotated_geo_top.add(geo_id)

    # 路径（俯视图只画X-Y投影）
    for pid, tids in assignment.items():
        if pid >= len(env.platforms):
            continue
        p = env.platforms[pid]
        prev_pos = p.position.copy()
        for idx, tid in enumerate(tids):
            if tid >= len(env.targets):
                continue
            t = env.targets[tid]
            c = CMAP.get(t.task_type, CMAP['侦察'])
            lw, ls = (2.5, '-') if idx == 0 else (1.8, '--')
            ax3.plot([prev_pos[0], t.position[0]],
                     [prev_pos[1], t.position[1]],
                     color=c['line'], linewidth=lw, linestyle=ls,
                     alpha=0.8, zorder=5)
            mid_x = (prev_pos[0] + t.position[0]) / 2
            mid_y = (prev_pos[1] + t.position[1]) / 2
            dx = t.position[0] - prev_pos[0]
            dy = t.position[1] - prev_pos[1]
            norm = np.linalg.norm([dx, dy]) + 1e-6
            dx, dy = dx / norm * 4000, dy / norm * 4000
            ax3.annotate('', xy=(mid_x + dx * 0.5, mid_y + dy * 0.5),
                         xytext=(mid_x - dx * 0.5, mid_y - dy * 0.5),
                         arrowprops=dict(arrowstyle='->', color=c['arr'],
                                         lw=2), zorder=8)
            prev_pos = t.position.copy()

    # 威胁区域
    for threat in env.threats:
        circle = plt.Circle((threat.center[0], threat.center[1]),
                            threat.radius, color='red', alpha=0.22, zorder=2)
        ax3.add_patch(circle)

    margin = max([t.radius for t in env.threats] + [0]) * 1.5
    ax3.set_xlim(-margin, Config.MAP_SIZE_X + margin)
    ax3.set_ylim(-margin, Config.MAP_SIZE_Y + margin)
    ax3.set_xlabel('X (m)', fontsize=11)
    ax3.set_ylabel('Y (m)', fontsize=11)
    ax3.set_title("俯视图（Z轴方向）", fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)

    # 俯视图简例
    top_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CMAP['侦察']['plat'],
               markersize=9, label='侦察', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CMAP['打击']['plat'],
               markersize=9, label='打击', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CMAP['察打一体']['plat'],
               markersize=9, label='察打一体', markeredgecolor='black'),
        Line2D([0], [0], color='red', lw=0, marker='o', markersize=8,
               markerfacecolor='red', alpha=0.3, label='威胁区'),
    ]
    ax3.legend(handles=top_legend, loc='upper right', fontsize=8)

    # 保存
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)

def _compare_assignments(old: Dict[str, List[int]],
                         new: Dict[str, List[int]],
                         env: TerrainEnv,
                         agent_names: List[str] = None,
                         all_agent_names: List[str] = None) -> str:
    """生成重分配前后对比表格（支持显示损失平台）"""
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("【重分配前后对比结果】")
    lines.append("=" * 90)
    header = f"{'平台':<10} {'重分配前任务序列':<35} {'重分配后任务序列':<35} {'状态变化'}"
    lines.append(header)
    lines.append("-" * 90)

    # 使用完整的原始平台列表，确保损失的平台也显示在表格中
    display_names = all_agent_names if all_agent_names else sorted(set(list(old.keys()) + list(new.keys())))

    for agent_name in display_names:
        old_tasks = old.get(agent_name, [])
        new_tasks = new.get(agent_name, [])
        old_tasks = old_tasks if old_tasks is not None else []
        new_tasks = new_tasks if new_tasks is not None else []

        old_str = str([f"task_{t + 1:02d}" for t in old_tasks]) if old_tasks else "无"
        new_str = str([f"task_{t + 1:02d}" for t in new_tasks]) if new_tasks else "无"

        if not old_tasks and not new_tasks:
            change = "⚪ 无分配"
        elif not old_tasks and new_tasks:
            change = "🟢 新增分配"
        elif old_tasks and not new_tasks:
            change = "🔴 任务清空"
        elif set(old_tasks) == set(new_tasks):
            change = "⚪ 无变化"
        else:
            change = "🟡 任务变更"

        lines.append(f"{agent_name:<10} {old_str:<35} {new_str:<35} {change}")

    # 统计行
    old_targets = set(t for tasks in old.values() for t in tasks)
    new_targets = set(t for tasks in new.values() for t in tasks)
    lines.append("-" * 90)
    lines.append(f"覆盖目标数: {len(old_targets)} → {len(new_targets)} / {len(env.geo_targets)}")
    lines.append(f"新增覆盖: {len(new_targets - old_targets)} 个 | 失去覆盖: {len(old_targets - new_targets)} 个")
    lines.append("=" * 90)
    return "\n".join(lines)

def _apply_nearest_fallback(assignment: Dict[int, List[int]], env: TerrainEnv) -> Dict[int, List[int]]:
    """
    硬编码就近兜底后处理。
    对"第一个任务明显跑远"的平台，强制替换为最近合法目标。
    按"跑得越远越优先"顺序处理，避免近处目标被远处平台霸占。
    """
    if not assignment:
        return assignment

    # 复制一份分配结果，避免污染原始数据
    new_assignment = {pid: tids[:] for pid, tids in assignment.items()}
    assigned_set = set(tid for tids in new_assignment.values() for tid in tids)

    # 1. 先找出所有"第一个任务跑远"的平台（>20km），按距离从大到小排序
    far_platforms = []
    for pid in range(len(env.platforms)):
        tids = new_assignment.get(pid, [])
        if not tids or pid >= len(env.platforms):
            continue
        first_tid = tids[0]
        if first_tid >= len(env.targets):
            continue
        dist = float(np.linalg.norm(env.platforms[pid].position - env.targets[first_tid].position))
        if dist > 20e3:  # 只关心超过20km的
            far_platforms.append((dist, pid, first_tid, tids))

    # 远的先处理！这样跑最远的平台优先释放远处目标、抢占近处目标
    far_platforms.sort(reverse=True, key=lambda x: x[0])

    # 2. 逐个尝试替换
    for cur_dist, pid, first_tid, tids in far_platforms:
        p = env.platforms[pid]
        first_t = env.targets[first_tid]

        best_alt = None
        best_dist = cur_dist

        # 遍历所有目标，找最近且合法的替代
        for alt_tid in range(len(env.targets)):
            if alt_tid == first_tid or alt_tid in assigned_set:
                continue
            alt_t = env.targets[alt_tid]

            # ① 类型约束：平台必须能做这个任务
            if not p.can_do(alt_t.task_type):
                continue

            # ② 时序约束：前序任务必须已被分配（确保打击任务的前序侦察已完成）
            if alt_t.predecessors:
                if not all(p_id in assigned_set for p_id in alt_t.predecessors):
                    continue

            # ③ 航程约束：从平台到目标的直线距离不能超过最大航程
            alt_dist = float(np.linalg.norm(p.position - alt_t.position))
            if alt_dist > p.max_range:
                continue

            # 记录最近的那个
            if alt_dist < best_dist:
                best_dist = alt_dist
                best_alt = alt_tid

        # 必须明显更近才换（至少近一半以上，防止微调引发震荡）
        if best_alt is not None and best_dist < cur_dist * 0.5:
            print(f"🔄 就近兜底: P{pid}({p.type}) T{first_tid}({first_t.task_type},{cur_dist/1000:.1f}km) "
                  f"→ T{best_alt}({env.targets[best_alt].task_type},{best_dist/1000:.1f}km)")
            tids[0] = best_alt
            assigned_set.remove(first_tid)
            assigned_set.add(best_alt)

    return new_assignment

# ============================================================
# 5. 核心接口：plan_assignments（供路径规划端程序调用）
# ============================================================

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
    available_agent_positions: dict[str, Any] | None = None,
    pending_task_positions: dict[str, Any] | None = None,
    active_task_ids: set[str] | None = None,                   # ← 新增：激活目标ID集合
) -> dict[str, Any] | None:
    """
    外部任务分配接口标准入口。

    路径规划端的测试脚本通过命令行指定本模块，并调用此函数。
    返回字典必须包含 {"agent_queues": {...}}。
    """

    global _global_state


    # ------------------------------------------------------------------
    # 5.1 获取场景数据（三种来源，按优先级：外部payload > 外部位置 > 本地文件）
    # ------------------------------------------------------------------
    platform_configs = []
    target_configs = []
    default_agent_names = []
    default_task_ids = []

    # 优先级1：外部直接传入完整的 scenario_payload（JSON 解析后的字典）
    if scenario_payload is not None and isinstance(scenario_payload, dict) and scenario_payload:
        platform_configs, target_configs, default_agent_names, default_task_ids = _parse_scenario(scenario_payload)

    # 优先级2：外部传入平台/目标位置（路径规划端主流程标准方式，完全不依赖文件）
    elif available_agent_positions is not None and pending_task_positions is not None:
        # 平台：按原始编号查类型，不因损失/重分配而错位
        names_to_use = agent_names if agent_names else list(available_agent_positions.keys())
        try:
            original_n = max(int(name.split('_')[1]) for name in names_to_use if '_' in name) + 1
        except ValueError:
            original_n = len(names_to_use)

        for i, name in enumerate(names_to_use):
            pos = available_agent_positions.get(name, [50.0, 50.0, 3.0])
            try:
                agent_idx = int(name.split('_')[1])
            except (ValueError, IndexError):
                agent_idx = i
            ptype = _get_platform_type_by_index(agent_idx, original_n)
            platform_configs.append({
                'type': ptype,
                'payload': None,
                'max_range': None,
                'speed': None,
                'pos': _norm_position(pos)
            })
        default_agent_names = names_to_use

        # 目标：先根据 active_task_ids 过滤 pending_task_positions
        task_ids_from_pos = list(pending_task_positions.keys())
        if active_task_ids is not None:
            task_ids_from_pos = [tid for tid in task_ids_from_pos if tid in active_task_ids]
            print(f"[Adapter] 收到 {len(pending_task_positions)} 个目标，其中 {len(task_ids_from_pos)} 个处于激活状态")

        n_targets_actual = len(task_ids_from_pos)
        # 估算原始目标总数（保持类型不因激活/失活而错位）
        try:
            original_n_targets = max(int(tid.split('_')[1]) for tid in task_ids_from_pos if '_' in tid)
        except ValueError:
            original_n_targets = n_targets_actual

        recon_count = 0
        multi_count = 0
        for i in range(n_targets_actual):
            task_id = task_ids_from_pos[i]
            pos = pending_task_positions[task_id]
            try:
                target_idx = int(task_id.split('_')[1]) - 1   # task_03 → 索引 2
            except (ValueError, IndexError):
                target_idx = i
            ttype = _get_target_type_by_index(target_idx, original_n_targets)
            if ttype == '侦察':
                tw_start, tw_end = 0, 1000
                req_payload = 1
                value = 0.5 + recon_count * 0.08
                recon_count += 1
            else:  # 察打一体
                tw_start = 100 + multi_count * 50
                tw_end = tw_start + 700
                req_payload = 2
                value = 0.8 + multi_count * 0.04
                multi_count += 1

            target_configs.append({
                'type': ttype,
                'value': round(value, 2),
                'pos': _norm_position(pos),
                'req_payload': req_payload,
                'time_win_start': tw_start,
                'time_win_end': tw_end,
                'predecessors': []
            })
        default_task_ids = task_ids_from_pos

    # 优先级3：从默认/指定文件读取（独立运行或测试模式）
    else:
        scenario_path = getattr(args, 'scenario_path', DEFAULT_SCENARIO_PATH) if args else DEFAULT_SCENARIO_PATH
        if os.path.exists(scenario_path):
            with open(scenario_path, 'r', encoding='utf-8') as f:
                scenario_payload = json.load(f)
            platform_configs, target_configs, default_agent_names, default_task_ids = _parse_scenario(scenario_payload)
        else:
            print(f"❌ 错误: 未收到外部场景数据，且默认场景文件 {scenario_path} 不存在")
            print("   可能原因：")
            print("   1. 路径规划端主流程未传入 available_agent_positions / pending_task_positions")
            print("   2. 独立运行时未提供 --scenario 参数或场景文件缺失")
            print("   解决：确保从主流程传入平台/目标位置，或提供有效的场景文件。")
            return None

    # ------------------------------------------------------------------
    # 5.2 统一对齐数量
    # ------------------------------------------------------------------
    if agent_names:
        n_platforms = len(agent_names)
    else:
        n_platforms = len(platform_configs)
        agent_names = default_agent_names[:n_platforms]

    n_targets = len(target_configs)

    # 平台配置对齐：如果原始配置比实际平台数多（如损失后），按 agent_name 编号提取
    if len(platform_configs) > n_platforms:
        aligned_platform_configs = []
        for name in agent_names:
            try:
                idx = int(name.split('_')[1])
            except (ValueError, IndexError):
                idx = len(aligned_platform_configs)
            if idx < len(platform_configs):
                aligned_platform_configs.append(platform_configs[idx])
            else:
                aligned_platform_configs.append({
                    'type': '察打一体', 'payload': 5, 'max_range': 600e3,
                    'pos': [50000.0, 50000.0, 3000.0]
                })
        platform_configs = aligned_platform_configs
    else:
        platform_configs = platform_configs[:n_platforms]
    while len(platform_configs) < n_platforms:
        platform_configs.append({
            'type': '察打一体', 'payload': 5, 'max_range': 600e3,
            'pos': [50000.0, 50000.0, 3000.0]
        })

    # 补齐目标配置（理论上不会走到，因为 n_targets 已精确计算）
    while len(target_configs) < n_targets:
        target_configs.append({
            'type': '察打一体', 'value': 0.7,
            'pos': [np.random.uniform(20e3, 80e3),
                    np.random.uniform(20e3, 80e3), 0],
            'req_payload': 1, 'time_win_start': 0,
            'time_win_end': 1000, 'predecessors': []
        })

    task_ids = default_task_ids if default_task_ids else [f"task_{i + 1:02d}" for i in range(n_targets)]
    task_id_to_index = {
        str(task_id): idx
        for idx, task_id in enumerate(task_ids)
    }
    if available_agent_positions:
        for i, name in enumerate(agent_names):
            if i < len(platform_configs) and name in available_agent_positions:
                platform_configs[i]['pos'] = _norm_position(available_agent_positions[name])


    if pending_task_positions:
        for task_id, raw_pos in pending_task_positions.items():
            task_id = str(task_id)
            target_index = task_id_to_index.get(task_id)
            if target_index is None or target_index >= len(target_configs):
                continue
            target_configs[target_index]['pos'] = _norm_position(raw_pos)


    # ------------------------------------------------------------------
    # 5.3 处理突发事件（更新位置 / 标记损失平台 / 新增目标）
    # ------------------------------------------------------------------
    is_replan = (phase == "replan") or (event is not None)
    old_assignment = None

    if is_replan:
        if isinstance(state, dict) and state.get('last_assignment') is not None:
            old_assignment = state.get('last_assignment')
        else:
            # 师兄没传 state 时，自动从全局缓存里取上次结果
            old_assignment = _global_state.get('last_assignment', None)

    # 位置变更事件：直接修改对应平台位置
    if event and event.get('type') == 'position_change':
        new_positions = event.get('new_positions', {})
        for pid_str, pos in new_positions.items():
            pid = int(pid_str)
            if 0 <= pid < len(platform_configs):
                platform_configs[pid]['pos'] = [
                    pos[0] * COORD_SCALE,
                    pos[1] * COORD_SCALE,
                    pos[2] * COORD_SCALE
                ]
        print(f"🔄 位置变更事件: 更新了 {len(new_positions)} 个平台位置")

    # 平台损失事件：从配置中移除（路径规划端框架通常已更新 agent_names，这里同步）
    if event and event.get('type') == 'agent_loss':
        lost_id = event.get('lost_agent_id') or event.get('platform_id')
        if lost_id is not None:
            print(f"🚨 应急事件: 平台 agent_{lost_id} 损失，启动重分配...")

    # 目标增加事件：在 target_configs 中追加新目标
    if event and event.get('type') == 'task_increase':
        new_positions = event.get('new_task_positions', event.get('activated_task_ids', []))
        # 如果传入的是坐标列表
        if new_positions and isinstance(new_positions[0], (list, tuple, np.ndarray)):
            base_id = len(target_configs)
            for i, pos in enumerate(new_positions):
                target_configs.append({
                    'type': '察打一体',
                    'value': 0.75,
                    'pos': [pos[0] * COORD_SCALE,
                            pos[1] * COORD_SCALE,
                            pos[2] * COORD_SCALE],
                    'req_payload': 1,
                    'time_win_start': 0,
                    'time_win_end': 1000,
                    'predecessors': []
                })
            print(f"🚨 应急事件: 新增 {len(new_positions)} 个目标，当前总目标 {len(target_configs)}")

    # ------------------------------------------------------------------
    # 5.4 初始化 / 更新环境
    # ------------------------------------------------------------------
    env_local = _global_state['env']
    system = _global_state['system']

    if env_local is None:
        env_local = TerrainEnv(use_real_terrain=False)
        _global_state['env'] = env_local

    env_local.setup_scenario(platform_configs, target_configs)

    # 设置固定威胁区域（每次调用都重置，避免重复累积）
    env_local.threats = []
    fixed_threats = [
        {'center': [20000, 20000, 0], 'radius': 13000, 'type': 'radar'},
        {'center': [20000, 80000, 0], 'radius': 9000, 'type': 'no_fly'},
        {'center': [80000, 20000, 0], 'radius': 13000, 'type': 'radar'},
        {'center': [80000, 80000, 0], 'radius': 13000, 'type': 'no_fly'},
        {'center': [50000, 50000, 0], 'radius': 15000, 'type': 'radar'},
        {'center': [50000, 15000, 0], 'radius': 9000, 'type': 'no_fly'},
    ]
    for cfg in fixed_threats:
        env_local.threats.append(Threat(
            center=np.array(cfg['center'], dtype=np.float32),
            radius=cfg['radius'],
            intensity=1.0,
            threat_type=cfg['type']
        ))

    if system is None:
        system = TaskSystem(env_local)
        _global_state['system'] = system
        system.reallocator = DynamicReallocator(env_local, system)
    # 加载模型（仅首次）
    if not _global_state['model_loaded']:
        model_path = getattr(args, 'assignment_provider_model_path', DEFAULT_MODEL_PATH) if args else DEFAULT_MODEL_PATH
        _global_state['model_loaded'] = _load_model(system, model_path)

    # 检测平台/目标数量变化，动态调整网络维度
    current_n_platforms = len(platform_configs)
    current_n_targets = len(env_local.targets)

    if (current_n_platforms != _global_state['last_n_platforms'] or
            current_n_targets != _global_state['last_n_targets']):
        system.reallocator.fine_tuner.adapt_to_new_size(current_n_platforms, current_n_targets)
        _global_state['last_n_platforms'] = current_n_platforms
        _global_state['last_n_targets'] = current_n_targets

    # ------------------------------------------------------------------
    # 5.5 执行分配（手动重置状态，不重新随机化位置）
    # ------------------------------------------------------------------
    for p in env_local.platforms:
        p.reset()
    for t in env_local.targets:
        t.reset()
    env_local.completed_tasks.clear()
    env_local.current_time = 0.0
    start_time = time.time()
    assignment, actions, _ = system.assign(evaluate=True)
    assignment = _apply_nearest_fallback(assignment, env_local)
    node_to_geo = {}
    for geo_id, gt in enumerate(env_local.geo_targets):
        for nid in gt['task_ids']:
            node_to_geo[nid] = geo_id

    # 每个平台分配了哪些地理目标（去重，保持顺序），键直接使用 agent_name 避免 PID 错位
    geo_assignment = {}
    for pid, node_ids in assignment.items():
        if pid >= len(agent_names):
            continue
        agent_name = agent_names[pid]
        seen = []
        for nid in node_ids:
            if nid in node_to_geo:
                gid = node_to_geo[nid]
                if gid not in seen:
                    seen.append(gid)
        geo_assignment[agent_name] = seen

    # 地理目标覆盖率统计
    geo_covered = set()
    for tids in geo_assignment.values():
        geo_covered.update(tids)
    print(f"地理目标覆盖: {len(geo_covered)}/{len(env_local.geo_targets)}")
    # 5.6 格式化输出
    # ------------------------------------------------------------------
    elapsed_ms = (time.time() - start_time) * 1000.0
    output_text = _format_assignment_text(geo_assignment, env_local, elapsed_ms, task_ids, agent_names)
    print(output_text)

    # 生成3D场景图
    suffix = "_replan" if is_replan else "_initial"
    pic_name = f"assignment_3d{suffix}.png"
    _save_3d_assignment(env_local, assignment, pic_name, geo_assignment, agent_names)

    # 重分配对比表（传入完整原始平台列表，确保损失平台也显示）
    comparison_text = None
    if is_replan and old_assignment is not None:
        all_names = _global_state.get('all_agent_names', agent_names)
        comparison_text = _compare_assignments(old_assignment, geo_assignment, env_local, agent_names, all_names)
        print(comparison_text)

    # 更新全局状态中的上次分配结果
    _global_state['last_assignment'] = geo_assignment
    if isinstance(state, dict):
        state['last_assignment'] = geo_assignment

    # 初始分配时保存完整的平台列表，供后续重分配对比使用
    if not is_replan:
        _global_state['all_agent_names'] = agent_names[:]

    # ------------------------------------------------------------------
    # 5.7 构建标准返回结构
    # ------------------------------------------------------------------
    agent_queues = _format_agent_queues(geo_assignment, agent_names, task_ids)

    # 构建平台类型映射，供下游 task_specs 排序使用
    platform_types = {}
    for i, name in enumerate(agent_names):
        if i < len(platform_configs):
            platform_types[name] = platform_configs[i]['type']
        else:
            platform_types[name] = '察打一体'

    # 构建目标类型映射，供下游 task_specs 使用
    target_types = {}
    for i, tid in enumerate(task_ids):
        if i < len(target_configs):
            target_types[tid] = target_configs[i]['type']
        else:
            target_types[tid] = '察打一体'

    result = {
        "agent_queues": agent_queues,
        "_platform_types": platform_types,
        "_target_types": target_types,
        # 以下字段供调试与下游扩展使用，不破坏路径规划端框架兼容性
        "_assignment_detail": assignment,
        "_output_text": output_text,
        "_elapsed_ms": elapsed_ms,
        "_coverage": len(geo_covered),
        "_total_targets": len(env_local.geo_targets),
    }

    if comparison_text:
        result["_comparison_text"] = comparison_text

    return result

# ============================================================
# 5.x 路径规划端框架标准入口包装函数
# ============================================================

def assignment_provider(agent_names, task_count, assignment_override=None, **provider_context):
    """
    路径规划端 Waypoint 框架标准入口。
    功能：把 **provider_context 中的字段转交给 plan_assignments，
          并补充路径规划端要求的 task_specs 返回字段。
    """
    # 1. 从 provider_context 中提取路径规划端框架可能传入的上下文
    env           = provider_context.get('env')
    state         = provider_context.get('state')
    phase         = provider_context.get('phase', 'initial')
    event         = provider_context.get('event')
    scenario_name = provider_context.get('scenario_name')
    available_agent_positions = provider_context.get('available_agent_positions')
    pending_task_positions = provider_context.get('pending_task_positions')
    scenario_payload = provider_context.get('scenario_payload')  # ← 新增：支持外部直接传入
    # 2. 只有当完全没有外部数据且指定了场景名时，才尝试读文件（向后兼容测试）
    if (scenario_payload is None and
        available_agent_positions is None and
        pending_task_positions is None and
        scenario_name):
        candidate_paths = [
            f"{scenario_name}_collaborate.json",
            f"{scenario_name}.json",
            DEFAULT_SCENARIO_PATH,
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    scenario_payload = json.load(f)
                break

    # 3. 提取激活目标ID集合（只有 active=True 的目标才参与分配）
    task_specs_in = provider_context.get('task_specs', [])
    active_task_ids = None
    if task_specs_in:
        active_task_ids = {t['task_id'] for t in task_specs_in if t.get('active', False)}
        print(f"[Adapter] 收到 {len(task_specs_in)} 个目标规格，其中 {len(active_task_ids)} 个已激活")

    # 4. 调用你已有的核心分配逻辑（所有脏活累活都在这里面）
    result = plan_assignments(
        agent_names=agent_names,
        task_count=task_count,
        assignment_override=assignment_override,
        env=env,
        args=None,
        scenario_payload=scenario_payload,
        phase=phase,
        state=state,
        event=event,
        scenario_name=scenario_name,
        available_agent_positions=available_agent_positions,
        pending_task_positions=pending_task_positions,
        active_task_ids=active_task_ids,
    )

    if result is None:
        return None

    # 5. 补充路径规划端框架强制要求的 task_specs（任务归属表）
    agent_queues = result.get("agent_queues", {})
    platform_types = result.get("_platform_types", {})
    target_types = result.get("_target_types", {})
    task_to_agents = {}
    for agent, tasks in agent_queues.items():
        for task_id in tasks:
            task_to_agents.setdefault(task_id, []).append(agent)

    # 基于路径规划端传入的 task_specs 更新 agents，保留 active / completed 状态
    if task_specs_in:
        task_specs = []
        for spec in task_specs_in:
            task_id = spec['task_id']
            agents = task_to_agents.get(task_id, [])

            # 对于察打一体目标，agents 按"先侦察、后打击"排序
            if target_types.get(task_id) == '察打一体':
                def _sort_key(agent_name):
                    ptype = platform_types.get(agent_name, '察打一体')
                    return 0 if ptype in ['侦察', '察打一体'] else 1
                agents = sorted(agents, key=_sort_key)

            new_spec = dict(spec)
            new_spec['agents'] = agents
            new_spec['task_type'] = target_types.get(task_id, new_spec.get('task_type', '察打一体'))
            task_specs.append(new_spec)
    else:
        # 兼容无 task_specs 传入的情况（独立运行测试）
        task_specs = []
        for task_id, agents in task_to_agents.items():
            if target_types.get(task_id) == '察打一体':
                def _sort_key(agent_name):
                    ptype = platform_types.get(agent_name, '察打一体')
                    return 0 if ptype in ['侦察', '察打一体'] else 1
                agents = sorted(agents, key=_sort_key)
            task_specs.append({
                "task_id": task_id,
                "agents": agents,
                "label": task_id,
                "task_type": target_types.get(task_id, '察打一体'),
                "active": True,
                "completed": False,
            })

    result["task_specs"] = task_specs
    return result
# ============================================================
# 6. 独立运行模式（供本地测试）
# ============================================================

def run_standalone(scenario_path: str = DEFAULT_SCENARIO_PATH,
                   model_path: str = DEFAULT_MODEL_PATH):
    """
    独立运行演示：
      1) 读取 scenario_A.json 执行初始分配
      2) 模拟平台 agent_1 损失，执行重分配并输出对比表
    """
    print("=" * 70)
    print("智能空面协同任务分配 - 对接程序独立测试模式")
    print("=" * 70)

    if not os.path.exists(scenario_path):
        print(f"❌ 场景文件不存在: {scenario_path}")
        return

    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario_payload = json.load(f)

    start_positions = scenario_payload.get('start_positions', [])
    task_positions = scenario_payload.get('task_positions', [])
    n_platforms = len(start_positions)
    n_targets = len(task_positions)
    agent_names = [f"agent_{i}" for i in range(n_platforms)]

    # 覆盖默认模型路径
    global DEFAULT_MODEL_PATH
    DEFAULT_MODEL_PATH = model_path

    # ---------- 初始分配 ----------
    print("\n【阶段1】初始分配")
    state = {}
    result_initial = assignment_provider(
        agent_names=agent_names,
        task_count=n_targets,
        scenario_payload=scenario_payload,
        phase="initial",
        state=state
    )
    print("\n【阶段1 task_specs】")
    print(json.dumps(result_initial.get("task_specs", []), ensure_ascii=False, indent=4))

    # ---------- 模拟应急：平台 agent_1 损失 ----------
    print("\n【阶段2】模拟应急重分配（平台 agent_1 损失）")
    new_agent_names = [a for a in agent_names if a != "agent_1"]
    # 构造位置字典走优先级2，避免 JSON 截断导致平台配置错位
    avail_pos = {f"agent_{i}": start_positions[i] for i in range(n_platforms) if f"agent_{i}" in new_agent_names}
    pend_pos = {f"task_{i+1:02d}": task_positions[i] for i in range(n_targets)}
    event = {
        'type': 'agent_loss',
        'lost_agent_id': 1,
        'remaining_task_ids': list(range(n_targets))
    }

    result_replan = assignment_provider(
        agent_names=new_agent_names,
        task_count=n_targets,
        available_agent_positions=avail_pos,
        pending_task_positions=pend_pos,
        phase="replan",
        state=state,
        event=event
    )
    print("\n【阶段2 task_specs】")
    print(json.dumps(result_replan.get("task_specs", []), ensure_ascii=False, indent=4))

    # ---------- 模拟应急：新增2个目标 ----------
    print("\n【阶段3】模拟应急重分配（新增2个临时目标）")
    event2 = {
        'type': 'task_increase',
        'new_task_positions': [[85.0, 85.0, 0.2], [15.0, 15.0, 0.2]]
    }
    result_replan2 = assignment_provider(
        agent_names=new_agent_names,
        task_count=n_targets + 2,
        scenario_payload=scenario_payload,
        phase="replan",
        state=state,
        event=event2
    )
    print("\n【阶段3 task_specs】")
    print(json.dumps(result_replan2.get("task_specs", []), ensure_ascii=False, indent=4))

    print("\n" + "=" * 70)
    print("独立测试完成。输出文件：")
    print("  - assignment_3d_initial.png   (初始分配3D图)")
    print("  - assignment_3d_replan.png    (重分配3D图)")
    print("=" * 70)


# ============================================================
# 7. 主入口
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='智能空面协同任务分配对接程序')
    parser.add_argument('--scenario', default=DEFAULT_SCENARIO_PATH, help='场景JSON文件路径')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='训练好的模型路径')
    parser.add_argument('--standalone', action='store_true', help='独立运行测试模式')
    cli_args = parser.parse_args()

    if cli_args.standalone:
        run_standalone(cli_args.scenario, cli_args.model)
    else:
        # 默认也执行独立测试，方便直接双击运行查看效果
        run_standalone(cli_args.scenario, cli_args.model)
