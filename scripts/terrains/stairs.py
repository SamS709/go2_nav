#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn a configurable double-flight stair set in Isaac Sim."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from labyrinth.grid import Direction


def _yaw_from_direction(direction: Direction) -> float:
	"""Map maze direction to a yaw angle in radians."""
	if direction == Direction.E:
		return 0.0
	if direction == Direction.S:
		return math.pi * 0.5
	if direction == Direction.W:
		return math.pi
	if direction == Direction.N:
		return -math.pi * 0.5
	return 0.0


def _direction_from_str(name: str) -> Direction:
	"""Convert string direction to Direction enum."""
	mapping = {"N": Direction.N, "E": Direction.E, "S": Direction.S, "W": Direction.W}
	return mapping[name]


def _rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
	"""Rotate a point in the XY plane by yaw."""
	c = math.cos(yaw)
	s = math.sin(yaw)
	return (c * x - s * y, s * x + c * y)


def _stairs_cfg(size: tuple[float, float, float]) -> sim_utils.CuboidCfg:
	"""Create a static cuboid config for a stair step."""
	return sim_utils.CuboidCfg(
		size=size,
		rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
		collision_props=sim_utils.CollisionPropertiesCfg(),
		visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.6), roughness=0.7),
	)


def spawn_double_flight_stairs(
	prim_path: str,
	center: tuple[float, float, float],
	direction: Direction,
	step_height: float,
	step_depth: float,
	step_width: float,
	num_steps: int,
	landing_depth: float | None = None,
	start_down: bool = True,
	max_length: float | None = None,
	max_width: float | None = None,
	return_footprint: bool = False,
):
	"""Spawn a down-then-up or up-then-down stair set that returns to base height."""
	if num_steps < 1:
		raise ValueError("num_steps must be >= 1")
	if step_height <= 0.0 or step_depth <= 0.0 or step_width <= 0.0:
		raise ValueError("step_height, step_depth, and step_width must be positive")

	landing_depth = step_depth if landing_depth is None else landing_depth

	flight_length = num_steps * step_depth
	total_length = flight_length * 2 + landing_depth

	if max_length is not None and total_length > max_length:
		scale = max_length / total_length
		step_depth *= scale
		landing_depth *= scale
		flight_length = num_steps * step_depth
		total_length = flight_length * 2 + landing_depth

	if max_width is not None and step_width > max_width:
		step_width = max_width

	yaw = _yaw_from_direction(direction)
	base_z = center[2]
	sign = -1.0 if start_down else 1.0

	sim_utils.create_prim(prim_path, "Xform")

	def spawn_step(step_index: int, local_x: float, top_height: float, step_id: int):
		size = (step_depth, step_width, step_height)
		center_z = top_height - step_height * 0.5
		local_y = 0.0
		world_x, world_y = _rotate_xy(local_x, local_y, yaw)
		pos = (center[0] + world_x, center[1] + world_y, center_z)
		cfg = _stairs_cfg(size)
		cfg.func(f"{prim_path}/Step_{step_id}", cfg, translation=pos, orientation=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))

	def spawn_landing(local_x: float, top_height: float, step_id: int):
		size = (landing_depth, step_width, step_height)
		center_z = top_height - step_height * 0.5
		local_y = 0.0
		world_x, world_y = _rotate_xy(local_x, local_y, yaw)
		pos = (center[0] + world_x, center[1] + world_y, center_z)
		cfg = _stairs_cfg(size)
		cfg.func(f"{prim_path}/Landing_{step_id}", cfg, translation=pos, orientation=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))

	start_x = -0.5 * total_length + 0.5 * step_depth
	landing_x = start_x + flight_length - 0.5 * step_depth + 0.5 * landing_depth
	flight2_x = start_x + flight_length + landing_depth

	step_id = 0
	# First flight
	for i in range(num_steps):
		top_height = base_z + sign * step_height * (i + 1)
		local_x = start_x + step_depth * i
		spawn_step(i, local_x, top_height, step_id)
		step_id += 1

	# Landing
	landing_top = base_z + sign * step_height * num_steps
	spawn_landing(landing_x, landing_top, step_id)
	step_id += 1

	# Second flight (symmetric return)
	sign = -sign
	for i in range(num_steps):
		top_height = landing_top + sign * step_height * (i + 1)
		local_x = flight2_x + step_depth * i
		spawn_step(i, local_x, top_height, step_id)
		step_id += 1

	if return_footprint:
		return {
			"length": total_length,
			"width": step_width,
		}
	return None


def run_simulator(sim: sim_utils.SimulationContext):
	"""Runs the simulation loop."""
	while sim.app.is_running():
		sim.step()


def _create_arg_parser() -> argparse.ArgumentParser:
	"""Create CLI parser for standalone stair demo."""
	parser = argparse.ArgumentParser(description="Spawn a configurable stair set in Isaac Sim.")
	parser.add_argument("--step_height", type=float, default=0.05, help="Step height (m).")
	parser.add_argument("--step_depth", type=float, default=0.2, help="Step depth along travel (m).")
	parser.add_argument("--step_width", type=float, default=0.4, help="Step width across travel (m).")
	parser.add_argument("--num_steps", type=int, default=4, help="Number of steps per flight.")
	parser.add_argument("--landing_depth", type=float, default=None, help="Landing depth between flights (m).")
	parser.add_argument(
		"--direction",
		type=str,
		default="E",
		choices=["N", "E", "S", "W"],
		help="Stair travel direction in the XY plane.",
	)
	parser.add_argument(
		"--start_down",
		action="store_true",
		default=False,
		help="Start with a descending flight first (then ascend).",
	)
	parser.add_argument("--max_length", type=float, default=None, help="Optional max length to fit the stairs (m).")
	parser.add_argument("--max_width", type=float, default=None, help="Optional max width to fit the stairs (m).")
	return parser


def main():
	"""Main function."""
	parser = _create_arg_parser()
	AppLauncher.add_app_launcher_args(parser)
	args_cli = parser.parse_args()

	# launch omniverse app
	app_launcher = AppLauncher(args_cli)
	simulation_app = app_launcher.app

	sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
	sim = sim_utils.SimulationContext(sim_cfg)
	sim.set_camera_view(eye=[2.0, -2.0, 1.5], target=[0.0, 0.0, 0.3])

	# Ground-plane
	ground_cfg = sim_utils.GroundPlaneCfg()
	ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
	# Lights
	light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
	light_cfg.func("/World/Light", light_cfg)

	direction = _direction_from_str(args_cli.direction)
	spawn_double_flight_stairs(
		prim_path="/World/Stairs",
		center=(0.0, 0.0, 0.0),
		direction=direction,
		step_height=args_cli.step_height,
		step_depth=args_cli.step_depth,
		step_width=args_cli.step_width,
		num_steps=args_cli.num_steps,
		landing_depth=args_cli.landing_depth,
		start_down=args_cli.start_down,
		max_length=args_cli.max_length,
		max_width=args_cli.max_width,
	)

	sim.reset()
	print("[INFO]: Setup complete...")
	run_simulator(sim)
	simulation_app.close()


if __name__ == "__main__":
	main()
