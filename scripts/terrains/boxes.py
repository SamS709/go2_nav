#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn random boxes on a square plane in Isaac Sim."""

"""Launch Isaac Sim Simulator first."""

import argparse
import random

from isaaclab.app import AppLauncher

"""Rest everything follows."""

import isaaclab.sim as sim_utils


def _plane_cfg(size: tuple[float, float, float]) -> sim_utils.CuboidCfg:
	"""Create static square plane config."""
	return sim_utils.CuboidCfg(
		size=size,
		rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
		collision_props=sim_utils.CollisionPropertiesCfg(),
		visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25), roughness=0.9),
	)


def _box_cfg(size: tuple[float, float, float]) -> sim_utils.CuboidCfg:
	"""Create static box config."""
	return sim_utils.CuboidCfg(
		size=size,
		rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
		collision_props=sim_utils.CollisionPropertiesCfg(),
		visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35), roughness=0.7),
	)


def spawn_boxes_on_square(
	prim_path: str,
	center: tuple[float, float, float],
	plane_size: float,
	n_boxes: int,
	h_boxes: float,
	plane_thickness: float = 0.02,
	seed: int | None = None,
	min_box_size: float = 0.01,
	spawn_plane: bool = True,
) -> int:
	"""Spawn random boxes in a square patch, with optional support plane.

	The number of boxes is sampled uniformly in [0, n_boxes].
	Each box dimension (height, width, length) is sampled uniformly in [0, h_boxes],
	then clamped by min_box_size for valid collision geometry.
	"""
	if plane_size <= 0.0:
		raise ValueError("plane_size must be positive")
	if n_boxes < 0:
		raise ValueError("n_boxes must be >= 0")
	if h_boxes < 0.0:
		raise ValueError("h_boxes must be >= 0")

	rng = random.Random(seed)
	min_box_size = max(1e-4, min_box_size)
	plane_thickness = max(1e-4, plane_thickness)

	sim_utils.create_prim(prim_path, "Xform")

	# Optional square support plane.
	if spawn_plane:
		plane_cfg = _plane_cfg((plane_size, plane_size, plane_thickness))
		plane_cfg.func(
			f"{prim_path}/Plane",
			plane_cfg,
			translation=(center[0], center[1], center[2] - 0.5 * plane_thickness),
		)

	n_spawn = rng.randint(0, n_boxes)
	half_plane = 0.5 * plane_size

	for i in range(n_spawn):
		lx = max(min_box_size, rng.uniform(0.0, h_boxes))
		ly = max(min_box_size, rng.uniform(0.0, h_boxes))
		lz = max(min_box_size, rng.uniform(0.0, h_boxes))

		# Keep box fully on the square support plane.
		x_margin = max(0.0, half_plane - 0.5 * lx)
		y_margin = max(0.0, half_plane - 0.5 * ly)
		x = center[0] + rng.uniform(-x_margin, x_margin)
		y = center[1] + rng.uniform(-y_margin, y_margin)
		z = center[2] + 0.5 * lz

		box_cfg = _box_cfg((lx, ly, lz))
		box_cfg.func(f"{prim_path}/Box_{i}", box_cfg, translation=(x, y, z))

	return n_spawn


def run_simulator(sim: sim_utils.SimulationContext):
	"""Runs the simulation loop."""
	while sim.app.is_running():
		sim.step()


def _create_arg_parser() -> argparse.ArgumentParser:
	"""Create CLI parser for standalone boxes demo."""
	parser = argparse.ArgumentParser(description="Spawn random boxes on a square plane.")
	parser.add_argument("--plane_size", type=float, default=1.0, help="Square plane size (m).")
	parser.add_argument("--n_boxes", type=int, default=10, help="Max number of boxes (uniformly sampled).")
	parser.add_argument("--h_boxes", type=float, default=0.2, help="Max box dimension for h/w/l sampling (m).")
	parser.add_argument("--plane_thickness", type=float, default=0.02, help="Plane thickness (m).")
	parser.add_argument("--seed", type=int, default=0, help="Seed for box count and dimensions.")
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
	sim.set_camera_view(eye=[2.0, -2.0, 1.6], target=[0.0, 0.0, 0.2])

	# Ground-plane
	ground_cfg = sim_utils.GroundPlaneCfg()
	ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
	# Lights
	light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
	light_cfg.func("/World/Light", light_cfg)

	n_spawned = spawn_boxes_on_square(
		prim_path="/World/BoxesPatch",
		center=(0.0, 0.0, 0.0),
		plane_size=args_cli.plane_size,
		n_boxes=args_cli.n_boxes,
		h_boxes=args_cli.h_boxes,
		plane_thickness=args_cli.plane_thickness,
		seed=args_cli.seed,
		spawn_plane=True,
	)

	print(f"[INFO]: Spawned {n_spawned} boxes.")
	sim.reset()
	print("[INFO]: Setup complete...")
	run_simulator(sim)
	simulation_app.close()


if __name__ == "__main__":
	main()
