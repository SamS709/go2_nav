#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn random roughness elements on a square patch in Isaac Sim."""

"""Launch Isaac Sim Simulator first."""

import argparse
import random

from isaaclab.app import AppLauncher

"""Rest everything follows."""

import isaaclab.sim as sim_utils


def _plane_cfg(size: tuple[float, float, float]) -> sim_utils.CuboidCfg:
	"""Create static square support plane config."""
	return sim_utils.CuboidCfg(
		size=size,
		rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
		collision_props=sim_utils.CollisionPropertiesCfg(),
		visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.28, 0.28), roughness=0.95),
	)


def _rough_cfg(size: tuple[float, float, float]) -> sim_utils.CuboidCfg:
	"""Create static roughness element config."""
	return sim_utils.CuboidCfg(
		size=size,
		rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
		collision_props=sim_utils.CollisionPropertiesCfg(),
		visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.38, 0.38, 0.38), roughness=1.0),
	)


def spawn_roughness_on_square(
	prim_path: str,
	center: tuple[float, float, float],
	patch_size: float,
	n_rough: int,
	h_rough_xy: float,
	h_rough_z: float,
	plane_thickness: float = 0.02,
	seed: int | None = None,
	min_feature_size: float = 0.008,
	spawn_plane: bool = False,
) -> int:
	"""Spawn random roughness elements in a square patch.

	- Number of elements is sampled uniformly in [0, n_rough].
	- Each element width/length is sampled uniformly in [0, h_rough_xy].
	- Each element height is sampled uniformly in [0, h_rough_z].
	  then clamped by min_feature_size.
	- If spawn_plane=False, elements are placed directly on the caller-provided ground.
	"""
	if patch_size <= 0.0:
		raise ValueError("patch_size must be positive")
	if n_rough < 0:
		raise ValueError("n_rough must be >= 0")
	if h_rough_xy < 0.0:
		raise ValueError("h_rough_xy must be >= 0")
	if h_rough_z < 0.0:
		raise ValueError("h_rough_z must be >= 0")

	rng = random.Random(seed)
	plane_thickness = max(1e-4, plane_thickness)
	min_feature_size = max(1e-4, min_feature_size)

	sim_utils.create_prim(prim_path, "Xform")

	if spawn_plane:
		plane_cfg = _plane_cfg((patch_size, patch_size, plane_thickness))
		plane_cfg.func(
			f"{prim_path}/Plane",
			plane_cfg,
			translation=(center[0], center[1], center[2] - 0.5 * plane_thickness),
		)

	n_spawn = rng.randint(0, n_rough)
	half_patch = 0.5 * patch_size

	for i in range(n_spawn):
		lx = max(min_feature_size, rng.uniform(0.0, h_rough_xy))
		ly = max(min_feature_size, rng.uniform(0.0, h_rough_xy))
		lz = max(min_feature_size, rng.uniform(0.0, h_rough_z))

		x_margin = max(0.0, half_patch - 0.5 * lx)
		y_margin = max(0.0, half_patch - 0.5 * ly)
		x = center[0] + rng.uniform(-x_margin, x_margin)
		y = center[1] + rng.uniform(-y_margin, y_margin)
		z = center[2] + 0.5 * lz

		rough_cfg = _rough_cfg((lx, ly, lz))
		rough_cfg.func(f"{prim_path}/Rough_{i}", rough_cfg, translation=(x, y, z))

	return n_spawn


def run_simulator(sim: sim_utils.SimulationContext):
	"""Runs the simulation loop."""
	while sim.app.is_running():
		sim.step()


def _create_arg_parser() -> argparse.ArgumentParser:
	"""Create CLI parser for standalone roughness demo."""
	parser = argparse.ArgumentParser(description="Spawn random roughness on a square patch.")
	parser.add_argument("--patch_size", type=float, default=1.0, help="Square patch size (m).")
	parser.add_argument("--n_rough", type=int, default=20, help="Max number of roughness elements.")
	parser.add_argument("--h_rough_xy", type=float, default=0.08, help="Max roughness width/length sampling value (m).")
	parser.add_argument("--h_rough_z", type=float, default=0.08, help="Max roughness height sampling value (m).")
	parser.add_argument("--plane_thickness", type=float, default=0.02, help="Plane thickness if spawn_plane=True (m).")
	parser.add_argument("--spawn_plane", action="store_true", default=False, help="Spawn local support plane.")
	parser.add_argument("--seed", type=int, default=0, help="Seed for roughness generation.")
	return parser


def main():
	"""Main function."""
	parser = _create_arg_parser()
	AppLauncher.add_app_launcher_args(parser)
	args_cli = parser.parse_args()

	app_launcher = AppLauncher(args_cli)
	simulation_app = app_launcher.app

	sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
	sim = sim_utils.SimulationContext(sim_cfg)
	sim.set_camera_view(eye=[2.0, -2.0, 1.6], target=[0.0, 0.0, 0.15])

	ground_cfg = sim_utils.GroundPlaneCfg()
	ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
	light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
	light_cfg.func("/World/Light", light_cfg)

	n_spawn = spawn_roughness_on_square(
		prim_path="/World/RoughPatch",
		center=(0.0, 0.0, 0.0),
		patch_size=args_cli.patch_size,
		n_rough=args_cli.n_rough,
		h_rough_xy=args_cli.h_rough_xy,
		h_rough_z=args_cli.h_rough_z,
		plane_thickness=args_cli.plane_thickness,
		seed=args_cli.seed,
		spawn_plane=args_cli.spawn_plane,
	)

	print(f"[INFO]: Spawned {n_spawn} roughness elements.")
	sim.reset()
	print("[INFO]: Setup complete...")
	run_simulator(sim)
	simulation_app.close()


if __name__ == "__main__":
	main()
