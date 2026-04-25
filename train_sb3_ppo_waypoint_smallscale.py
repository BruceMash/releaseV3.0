"""Backward-compatible entry point for the renamed global planner trainer."""

from train_sb3_ppo_waypoint_global_planner import *  # noqa: F401,F403
from train_sb3_ppo_waypoint_global_planner import main


if __name__ == "__main__":
    print("提示: train_sb3_ppo_waypoint_smallscale.py 已更名为 train_sb3_ppo_waypoint_global_planner.py")
    main()
