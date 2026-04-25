"""Backward-compatible entry point for the renamed global planner tester."""

from test_sb3_waypoint_global_planner import *  # noqa: F401,F403
from test_sb3_waypoint_global_planner import main


if __name__ == "__main__":
    print("提示: test_sb3_waypoint_smallscale.py 已更名为 test_sb3_waypoint_global_planner.py")
    main()
