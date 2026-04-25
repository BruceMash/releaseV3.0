import argparse
import datetime
import json
import os

# plugin_dir = r"D:\RL_env\Lib\site-packages\PyQt6\Qt6\plugins\platforms"
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir

import matplotlib.pyplot as plt
import numpy as np
import supersuit as ss

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.ppo import MlpPolicy

from Envs.continuous_3d_waypoint_env_global_planner import UAVWaypoint3DMAPFEnv


DEFAULT_TERRAIN_CSV = os.path.join("Envs", "jiangning_mountain_simplified_smoother.csv")
DEFAULT_EVAL_MODEL_PATH = os.path.join("sb3_models_waypoint", "model", "best_model.zip")
DEFAULT_OBSTACLES_PATH = os.path.join("Configs", "fixed_task_scenarios", "scenario_A.json")


class StepRewardCallback(BaseCallback):
    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            self.logger.record("step/reward_mean", float(np.mean(rewards)))
            self.logger.record("step/reward_min", float(np.min(rewards)))
            self.logger.record("step/reward_max", float(np.max(rewards)))
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
        finished_success = []
        finished_obstacle_collisions = []
        finished_agent_collisions = []
        finished_remaining_distance = []

        for idx, done in enumerate(dones):
            if not done:
                continue

            info = infos[idx] if idx < len(infos) else {}
            finished_returns.append(float(self.episode_returns[idx]))
            finished_lengths.append(float(self.episode_lengths[idx]))
            finished_success.append(float(bool(info.get("reached_goal", False))))
            finished_obstacle_collisions.append(float(info.get("collision_with_obstacle", 0)))
            finished_agent_collisions.append(float(info.get("collision_with_agent", 0)))

            remaining_distance = info.get("remaining_distance", None)
            if remaining_distance is not None:
                finished_remaining_distance.append(float(remaining_distance))

            self.episode_returns[idx] = 0.0
            self.episode_lengths[idx] = 0

        if finished_returns:
            self.logger.record("episode/return", float(np.mean(finished_returns)))
            self.logger.record("episode/length", float(np.mean(finished_lengths)))
            self.logger.record("episode/success_rate", float(np.mean(finished_success)))
            self.logger.record("episode/obstacle_collisions", float(np.mean(finished_obstacle_collisions)))
            self.logger.record("episode/agent_collisions", float(np.mean(finished_agent_collisions)))
            if finished_remaining_distance:
                self.logger.record(
                    "episode/final_remaining_distance",
                    float(np.mean(finished_remaining_distance)),
                )

        return True


def _build_global_planner_env_kwargs(args_or_none=None, n_agents=5, max_steps=200):
    args = args_or_none
    return {
        "space_dim": (100.0, 100.0, 50.0),  # km
        "n_agents": int(n_agents),
        "max_steps": int(max_steps),
        "num_obstacles": 3,
        "render_mode": None,
        "scenario_json": getattr(args, "scenario_json", DEFAULT_OBSTACLES_PATH)
        if args is not None
        else DEFAULT_OBSTACLES_PATH,
        "terrain_enabled": getattr(args, "terrain_enabled", True) if args is not None else True,
        "terrain_csv_path": getattr(args, "terrain_csv_path", DEFAULT_TERRAIN_CSV)
        if args is not None
        else DEFAULT_TERRAIN_CSV,
        "terrain_z_scale": 0.001,
    }


def make_env(n_agents=5, max_steps=200, args=None):
    env = UAVWaypoint3DMAPFEnv(
        **_build_global_planner_env_kwargs(
            args_or_none=args,
            n_agents=n_agents,
            max_steps=max_steps,
        )
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    return env


def train(args):
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
    config.update(
        _build_global_planner_env_kwargs(
            args_or_none=args,
            n_agents=args.n_agents,
            max_steps=args.max_steps,
        )
    )
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"正在初始化包含 {args.n_agents} 架无人机的小尺度 waypoint 训练环境...")
    print(f"当前训练 Run ID: {run_id}")
    env = make_env(n_agents=args.n_agents, max_steps=args.max_steps, args=args)
    eval_env = make_env(n_agents=args.n_agents, max_steps=args.max_steps, args=args)
    env = VecMonitor(env, filename=os.path.join(log_dir, "train_monitor.csv"))
    eval_env = VecMonitor(eval_env, filename=os.path.join(log_dir, "eval_monitor.csv"))

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50000 // max(int(args.n_agents), 1)),
        save_path=os.path.join(model_dir, "checkpoints/"),
        name_prefix="uav_waypoint_ppo",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model/"),
        log_path=os.path.join(log_dir, "eval/"),
        eval_freq=max(1, 10000 // max(int(args.n_agents), 1)),
        deterministic=True,
        render=False,
    )

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
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
    )

    model.save(os.path.join(model_dir, "ppo_uav_waypoint_final"))
    print("\n训练完成，最终模型已保存。")


def _resolve_eval_model_path(args):
    explicit_model_path = str(getattr(args, "eval_model_path", "") or "").strip()
    if explicit_model_path and os.path.exists(explicit_model_path):
        return explicit_model_path

    if args.run_id:
        model_dir = f"./sb3_models_waypoint/{args.run_id}/"
        best_model = os.path.join(model_dir, "best_model", "best_model.zip")
        final_model = os.path.join(model_dir, "ppo_uav_waypoint_final.zip")
        if os.path.exists(best_model):
            return best_model
        if os.path.exists(final_model):
            return final_model
        return ""

    if not os.path.exists("./sb3_models_waypoint/"):
        return ""

    runs = [
        d
        for d in os.listdir("./sb3_models_waypoint/")
        if os.path.isdir(os.path.join("./sb3_models_waypoint/", d))
    ]
    if not runs:
        return ""
    runs.sort(reverse=True)
    model_dir = f"./sb3_models_waypoint/{runs[0]}/"
    best_model = os.path.join(model_dir, "best_model", "best_model.zip")
    final_model = os.path.join(model_dir, "ppo_uav_waypoint_final.zip")
    if os.path.exists(best_model):
        return best_model
    if os.path.exists(final_model):
        return final_model
    return ""


def evaluate(args):
    model_path = _resolve_eval_model_path(args)
    if not model_path:
        print("未找到可用模型文件。")
        return

    print(f"正在加载模型用于 waypoint 环境测试: {model_path}")
    model = PPO.load(model_path)

    raw_env = UAVWaypoint3DMAPFEnv(
        **_build_global_planner_env_kwargs(
            args_or_none=args,
            n_agents=args.n_agents,
            max_steps=args.max_steps,
        )
    )

    try:
        from test_sb3_waypoint_global_planner import announce_event, plot_waypoint_env
    except ImportError:
        print("未找到 test_sb3_waypoint_global_planner.py，无法执行可视化评估。")
        return

    obs, infos = raw_env.reset()
    del infos

    plt.ion()
    fig = plt.figure(figsize=(16, 8))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    ax3d.view_init(elev=30, azim=45)

    total_rewards = {agent: 0.0 for agent in raw_env.possible_agents}
    history = {agent: [raw_env.agent_positions[agent].copy()] for agent in raw_env.possible_agents}
    last_obstacle_collisions = {
        agent: raw_env.agent_obstacle_collisions.get(agent, 0)
        for agent in raw_env.possible_agents
    }
    last_agent_collisions = {
        agent: raw_env.agent_inter_agent_collisions.get(agent, 0)
        for agent in raw_env.possible_agents
    }
    final_png_path = ""

    print("开始演示 waypoint 策略表现...")
    for step in range(args.max_steps):
        plot_waypoint_env(ax3d, ax2d, raw_env, step, history=history)
        plt.draw()
        plt.pause(0.08)

        actions = {}
        for agent in raw_env.agents:
            action, _states = model.predict(obs[agent], deterministic=True)
            actions[agent] = action

        obs, rewards, terminations, truncations, infos = raw_env.step(actions)

        for agent in raw_env.possible_agents:
            if agent in raw_env.agent_positions:
                history.setdefault(agent, []).append(raw_env.agent_positions[agent].copy())

            obstacle_collisions = raw_env.agent_obstacle_collisions.get(agent, 0)
            if obstacle_collisions > last_obstacle_collisions[agent]:
                announce_event(f"{agent} collided with an obstacle")
                last_obstacle_collisions[agent] = obstacle_collisions

            agent_collisions = raw_env.agent_inter_agent_collisions.get(agent, 0)
            if agent_collisions > last_agent_collisions[agent]:
                announce_event(f"{agent} collided with another UAV")
                last_agent_collisions[agent] = agent_collisions

        for agent, reward in rewards.items():
            total_rewards[agent] += reward

        if all(truncations.values()) or all(terminations.values()):
            print(f"环境在第 {step} 步结束。")
            break

    plot_waypoint_env(ax3d, ax2d, raw_env, step + 1 if 'step' in locals() else 0, history=history)
    plt.draw()
    output_dir = os.path.dirname(model_path) if model_path else os.getcwd()
    final_png_path = os.path.join(
        output_dir,
        f"global_planner_eval_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    fig.savefig(final_png_path, dpi=160, bbox_inches="tight")

    plt.ioff()
    plt.show()
    print(f"最后一步图片已保存到: {final_png_path}")

    print("\n测试回合结束，各无人机累计回报如下:")
    for agent, reward in total_rewards.items():
        print(f"  {agent}: {reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waypoint 3D UAV PPO 训练与评估脚本")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--n_agents", type=int, default=10, help="训练/测试的无人机数量")
    parser.add_argument("--max_steps", type=int, default=200, help="每回合最大步数")
    parser.add_argument("--scenario_json", type=str, default=DEFAULT_OBSTACLES_PATH, help="读取固定起始点和障碍物配置的 JSON 路径")
    parser.add_argument(
        "--terrain_enabled",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="是否启用 CSV 高度图地形",
    )
    parser.add_argument("--terrain_csv_path", type=str, default=DEFAULT_TERRAIN_CSV, help="高度图 CSV 路径")
    parser.add_argument("--eval_model_path", type=str, default=DEFAULT_EVAL_MODEL_PATH, help="评估时优先加载的模型路径")
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
        help="评估模式下指定训练 run_id，不填则自动选择最新模型目录",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
