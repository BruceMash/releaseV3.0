import os
import json
import subprocess

plugin_dir = r"D:\RL_env\Lib\site-packages\PyQt6\Qt6\plugins\platforms"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from stable_baselines3 import PPO

from Envs.continuous_3d_waypoint_env_global_planner import UAVWaypoint3DMAPFEnv


def draw_cylinder(ax, x, y, radius, height, color="gray", alpha=0.5):
    z = np.linspace(0, height, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + x
    y_grid = radius * np.sin(theta_grid) + y
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, shade=True)


def draw_ellipsoid(ax, x, y, z, rx, ry, rz, color="gray", alpha=0.5):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi / 2, 20)
    x_grid = x + rx * np.outer(np.cos(u), np.sin(v))
    y_grid = y + ry * np.outer(np.sin(u), np.sin(v))
    z_grid = z + rz * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, shade=True)


def draw_ground_disc(ax, x, y, radius, color="gray", alpha=0.18):
    theta = np.linspace(0, 2 * np.pi, 60)
    radial = np.linspace(0, 1.0, 20)
    theta_grid, radial_grid = np.meshgrid(theta, radial)
    x_grid = x + radius * radial_grid * np.cos(theta_grid)
    y_grid = y + radius * radial_grid * np.sin(theta_grid)
    z_grid = np.zeros_like(x_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, shade=False)


def draw_terrain_surface(ax3d, ax2d, env):
    terrain = getattr(env, "terrain", None)
    if terrain is None or not getattr(terrain, "enabled", False):
        return
    if getattr(terrain, "grid_x", None) is None or getattr(terrain, "grid_y", None) is None:
        return
    if getattr(terrain, "height_map", None) is None:
        return

    x_grid = terrain.grid_x
    y_grid = terrain.grid_y
    z_grid = terrain.height_map

    ax3d.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap="terrain",
        alpha=0.45,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    ax2d.contourf(
        x_grid,
        y_grid,
        z_grid,
        levels=24,
        cmap="terrain",
        alpha=0.35,
    )


def announce_event(message):
    print(f"[Broadcast] {message}")
    escaped_message = message.replace("'", "''")
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$speaker.Speak('{escaped_message}')"
    )
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def plot_waypoint_env(ax3d, ax2d, env, step, history=None):
    ax3d.clear()
    ax3d.set_xlim(0, env.X)
    ax3d.set_ylim(0, env.Y)
    ax3d.set_zlim(0, max(float(getattr(env, "Z", 0.0)), 15.0))
    ax3d.set_xlabel("X (km)")
    ax3d.set_ylabel("Y (km)")
    ax3d.set_zlabel("Z (km)")
    ax3d.set_title(f"Waypoint 3D View - Step: {step}", fontsize=14)

    ax2d.clear()
    ax2d.set_xlim(0, env.X)
    ax2d.set_ylim(0, env.Y)
    ax2d.set_xlabel("X (km)")
    ax2d.set_ylabel("Y (km)")
    ax2d.set_title(f"Waypoint Top View - Step: {step}", fontsize=14)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(True, alpha=0.2)

    draw_terrain_surface(ax3d, ax2d, env)

    for obs in env.obstacles:
        if getattr(obs, "kind", "cylinder") == "hemisphere":
            draw_ground_disc(ax3d, obs.x, obs.y, obs.radius, color="darkorange", alpha=0.16)
            draw_ellipsoid(
                ax3d,
                obs.x,
                obs.y,
                obs.z,
                obs.radius,
                obs.radius,
                obs.height,
                color="darkorange",
                alpha=0.28,
            )
            ax2d.add_patch(Circle((obs.x, obs.y), obs.radius, color="darkorange", alpha=0.24))
            ax2d.add_patch(Circle((obs.x, obs.y), obs.radius, fill=False, color="saddlebrown", linewidth=1.5, alpha=0.9))
        else:
            draw_cylinder(ax3d, obs.x, obs.y, obs.radius, obs.height, color="steelblue", alpha=0.32)
            ax2d.add_patch(Circle((obs.x, obs.y), obs.radius, color="steelblue", alpha=0.22))
            ax2d.add_patch(Circle((obs.x, obs.y), obs.radius, fill=False, color="navy", linewidth=1.2, alpha=0.85))

    colors = plt.cm.rainbow(np.linspace(0, 1, env.n_agents))
    color_map = {agent: colors[index] for index, agent in enumerate(env.possible_agents)}

    for agent in env.possible_agents:
        if agent not in env.agent_positions or agent not in env.agent_targets:
            continue

        color = color_map[agent]
        is_active = agent in env.agents
        is_done = getattr(env, "agent_dones", {}).get(agent, False)
        position = env.agent_positions[agent]
        target = env.agent_targets[agent]
        last_motion = env.agent_last_motion.get(agent, np.zeros(3, dtype=np.float32))

        marker = "^" if is_done else "o"
        alpha = 0.95 if (is_active or is_done) else 0.45
        ax3d.scatter(
            position[0],
            position[1],
            position[2],
            color=color,
            marker=marker,
            s=100,
            edgecolors="black",
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

        ax2d.scatter(position[0], position[1], color=color, marker=marker, s=80, edgecolors="black", alpha=alpha, label=agent)
        ax2d.scatter(target[0], target[1], color=color, marker="*", s=120, alpha=0.9)
        ax2d.plot([position[0], target[0]], [position[1], target[1]], color=color, linestyle="--", alpha=0.3)

        if history is not None and agent in history:
            path = np.array(history[agent], dtype=np.float32)
            if len(path) > 1:
                ax3d.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=2.0, alpha=0.7)
                ax2d.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, alpha=0.7)

    handles, labels = ax3d.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax3d.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.02, 0.5))


def load_latest_model(model_root="./sb3_models_waypoint"):
    if not model_root or not model_root.strip():
        return None
    if not os.path.exists(model_root):
        return None
    runs = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
    if not runs:
        return None
    runs.sort(reverse=True)
    model_dir = os.path.join(model_root, runs[0])
    best_model = os.path.join(model_dir, "best_model", "best_model.zip")
    final_model = os.path.join(model_dir, "ppo_uav_waypoint_final.zip")
    if os.path.exists(best_model):
        return best_model
    if os.path.exists(final_model):
        return final_model
    return None


def load_training_config(model_path):
    if not model_path:
        return {}

    model_dir = os.path.dirname(model_path)
    if os.path.basename(model_dir).lower() == "best_model":
        model_dir = os.path.dirname(model_dir)

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def run_waypoint_test(
    model_path="",
    n_agents=None,
    max_steps=None,
    scenario_json="",
    terrain_enabled=True,
    terrain_csv_path=os.path.join("Envs", "jiangning_mountain_simplified_smoother.csv"),
):
    print("\n--- Starting Waypoint 3D Rendering ---")

    if not model_path:
        model_path = load_latest_model()

    model = PPO.load(model_path) if model_path else None
    if model is None:
        config = {}
        print("未找到训练模型，将使用朝目标点的启发式动作。")
    else:
        print(f"加载模型: {model_path}")
        config = load_training_config(model_path)

    n_agents = int(config.get("n_agents", 5) if n_agents is None else n_agents)
    max_steps = int(config.get("max_steps", 200) if max_steps is None else max_steps)

    env = UAVWaypoint3DMAPFEnv(
        space_dim=(100.0, 100.0, 50.0),
        n_agents=n_agents,
        max_steps=max_steps,
        num_obstacles=3,
        render_mode=None,
        scenario_json=scenario_json,
        terrain_enabled=terrain_enabled,
        terrain_csv_path=terrain_csv_path,
    )

    obs, infos = env.reset()
    history = {agent: [env.agent_positions[agent].copy()] for agent in env.possible_agents}
    last_obstacle_collisions = {
        agent: env.agent_obstacle_collisions.get(agent, 0)
        for agent in env.possible_agents
    }
    last_agent_collisions = {
        agent: env.agent_inter_agent_collisions.get(agent, 0)
        for agent in env.possible_agents
    }

    plt.ion()
    fig = plt.figure(figsize=(16, 8))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    ax3d.view_init(elev=30, azim=45)

    total_rewards = {agent: 0.0 for agent in env.possible_agents}

    for step in range(max_steps):
        plot_waypoint_env(ax3d, ax2d, env, step, history=history)
        plt.draw()
        plt.pause(0.08)

        actions = {}
        for agent in env.agents:
            if model is not None:
                action, _states = model.predict(obs[agent], deterministic=True)
                actions[agent] = action
                continue

            target_vec = env.agent_targets[agent] - env.agent_positions[agent]
            scaled_action = target_vec / max(env.waypoint_step_size, 1e-8)
            actions[agent] = np.clip(scaled_action, -1.0, 1.0)

        obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent in env.possible_agents:
            if agent in env.agent_positions:
                history[agent].append(env.agent_positions[agent].copy())

            obstacle_collisions = env.agent_obstacle_collisions.get(agent, 0)
            if obstacle_collisions > last_obstacle_collisions[agent]:
                announce_event(f"{agent} collided with an obstacle")
                last_obstacle_collisions[agent] = obstacle_collisions

            agent_collisions = env.agent_inter_agent_collisions.get(agent, 0)
            if agent_collisions > last_agent_collisions[agent]:
                announce_event(f"{agent} collided with another UAV")
                last_agent_collisions[agent] = agent_collisions

        for agent, reward in rewards.items():
            total_rewards[agent] += reward

        if all(terminations.values()) or all(truncations.values()):
            plot_waypoint_env(ax3d, ax2d, env, step + 1, history=history)
            plt.draw()
            if all(terminations.values()):
                print("All UAVs reached their targets.")
            else:
                print("Environment reached max_steps. Task truncated.")
            break

    plt.ioff()
    print("Waypoint visualization finished. Close the window.")
    plt.show()

    print("\nWaypoint test return summary:")
    for agent, reward in total_rewards.items():
        print(f"  {agent}: {reward:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Waypoint PPO evaluation / visualization script")
    parser.add_argument("--model_path", type=str, default="", help="模型路径，不填则自动选择最新模型")
    parser.add_argument("--n_agents", type=int, default=None, help="测试无人机数量，默认读取训练配置")
    parser.add_argument("--max_steps", type=int, default=None, help="每回合最大步数，默认读取训练配置")
    parser.add_argument("--scenario_json", type=str, default=os.path.join("Configs", "fixed_task_scenarios", "scenario_A.json"), help="固定起始点和障碍物配置 JSON")
    parser.add_argument("--terrain_enabled", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True, help="是否启用高度图地形")
    parser.add_argument("--terrain_csv_path", type=str, default=os.path.join("Envs", "jiangning_mountain_simplified_smoother.csv"), help="高度图 CSV 路径")
    args = parser.parse_args()

    run_waypoint_test(
        model_path=args.model_path,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        scenario_json=args.scenario_json,
        terrain_enabled=args.terrain_enabled,
        terrain_csv_path=args.terrain_csv_path,
    )
