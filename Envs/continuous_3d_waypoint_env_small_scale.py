"""Backward-compatible import wrapper.

The implementation was renamed to ``continuous_3d_waypoint_env_global_planner``
because this environment is used by the global large-step waypoint planner.
"""

from Envs.continuous_3d_waypoint_env_global_planner import *  # noqa: F401,F403
