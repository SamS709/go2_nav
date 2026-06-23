# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Display an overview of all built-in Isaac Lab terrains."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Overview of built-in Isaac Lab terrains.")
parser.add_argument(
	"--rows",
	type=int,
	default=1,
	help="Number of difficulty rows to generate. Each row increases terrain difficulty.",
)
parser.add_argument(
	"--sub_terrain_size",
	type=float,
	default=6.0,
	help="Size (m) of each sub-terrain along x and y.",
)
parser.add_argument(
	"--color_scheme",
	type=str,
	default="height",
	choices=["height", "random", "none"],
	help="Color scheme to use for generated terrains.",
)
parser.add_argument(
	"--difficulty",
	type=float,
	default=0.5,
	help="Difficulty for a single-row overview. Ignored when rows > 1.",
)
parser.add_argument(
	"--debug_vis",
	action="store_true",
	default=False,
	help="Show terrain origin markers.",
)
parser.add_argument(
	"--horizontal_scale",
	type=float,
	default=0.1,
	help="Discretization along x/y for height-field terrains (m).",
)
parser.add_argument(
	"--vertical_scale",
	type=float,
	default=0.005,
	help="Discretization along z for height-field terrains (m).",
)
parser.add_argument(
	"--slope_threshold",
	type=float,
	default=0.75,
	help="Slope threshold above which height-field surfaces become vertical.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg


def _safe_grid_width(sub_terrain_size: float, target_cells: float = 12.0, epsilon: float = 0.25) -> float:
	"""Computes a grid width that guarantees a positive border for random grid terrain."""
	return sub_terrain_size / (target_cells + epsilon)


def _build_sub_terrains(sub_terrain_size: float) -> dict[str, terrain_gen.SubTerrainBaseCfg]:
	"""Builds a deterministic list of all built-in terrain configs."""
	grid_width = _safe_grid_width(sub_terrain_size)
	sub_terrains: dict[str, terrain_gen.SubTerrainBaseCfg] = {
		# Height-field terrains
		"hf_random_uniform": terrain_gen.HfRandomUniformTerrainCfg(
			proportion=1.0, noise_range=(0.02, 0.08), noise_step=0.02
		),
		"hf_pyramid_sloped": terrain_gen.HfPyramidSlopedTerrainCfg(
			proportion=1.0, slope_range=(0.0, 0.4), platform_width=1.5
		),
		"hf_pyramid_sloped_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
			proportion=1.0, slope_range=(0.0, 0.4), platform_width=1.5
		),
		"hf_pyramid_stairs": terrain_gen.HfPyramidStairsTerrainCfg(
			proportion=1.0, step_height_range=(0.05, 0.23), step_width=0.3, platform_width=1.5
		),
		"hf_pyramid_stairs_inv": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
			proportion=1.0, step_height_range=(0.05, 0.23), step_width=0.3, platform_width=1.5
		),
		"hf_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
			proportion=1.0,
			obstacle_width_range=(0.2, 0.6),
			obstacle_height_range=(0.05, 0.2),
			num_obstacles=40,
			platform_width=1.0,
		),
		"hf_wave": terrain_gen.HfWaveTerrainCfg(
			proportion=1.0, amplitude_range=(0.02, 0.08), num_waves=2
		),
		"hf_stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
			proportion=1.0,
			stone_height_max=0.2,
			stone_width_range=(0.2, 0.4),
			stone_distance_range=(0.1, 0.3),
			holes_depth=-1.0,
			platform_width=1.0,
		),
		# Trimesh terrains
		"mesh_plane": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
		"mesh_pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
			proportion=1.0,
			border_width=0.5,
			step_height_range=(0.05, 0.23),
			step_width=0.3,
			platform_width=1.5,
			holes=False,
		),
		"mesh_pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
			proportion=1.0,
			border_width=0.5,
			step_height_range=(0.05, 0.23),
			step_width=0.3,
			platform_width=1.5,
			holes=False,
		),
		"mesh_random_grid": terrain_gen.MeshRandomGridTerrainCfg(
			proportion=1.0,
			grid_width=grid_width,
			grid_height_range=(0.05, 0.2),
			platform_width=1.5,
			holes=False,
		),
		"mesh_rails": terrain_gen.MeshRailsTerrainCfg(
			proportion=1.0, rail_thickness_range=(0.05, 0.15), rail_height_range=(0.05, 0.2), platform_width=1.5
		),
		"mesh_pit": terrain_gen.MeshPitTerrainCfg(
			proportion=1.0, pit_depth_range=(0.1, 0.4), platform_width=1.5, double_pit=False
		),
		"mesh_box": terrain_gen.MeshBoxTerrainCfg(
			proportion=1.0, box_height_range=(0.1, 0.4), platform_width=1.5, double_box=False
		),
		"mesh_gap": terrain_gen.MeshGapTerrainCfg(
			proportion=1.0, gap_width_range=(0.2, 0.6), platform_width=1.5
		),
		"mesh_floating_ring": terrain_gen.MeshFloatingRingTerrainCfg(
			proportion=1.0,
			ring_width_range=(0.2, 0.5),
			ring_height_range=(0.05, 0.2),
			ring_thickness=0.1,
			platform_width=1.5,
		),
		"mesh_star": terrain_gen.MeshStarTerrainCfg(
			proportion=1.0, num_bars=5, bar_width_range=(0.1, 0.3), bar_height_range=(0.05, 0.2), platform_width=1.0
		),
		"mesh_repeated_pyramids": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
			proportion=1.0,
			object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
				num_objects=20, height=0.1, radius=0.15, max_yx_angle=0.0
			),
			object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
				num_objects=40, height=0.25, radius=0.25, max_yx_angle=10.0
			),
			platform_width=1.0,
		),
		"mesh_repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
			proportion=1.0,
			object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
				num_objects=20, height=0.1, size=(0.15, 0.15), max_yx_angle=0.0
			),
			object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
				num_objects=40, height=0.25, size=(0.3, 0.3), max_yx_angle=10.0
			),
			platform_width=1.0,
		),
		"mesh_repeated_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
			proportion=1.0,
			object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
				num_objects=20, height=0.1, radius=0.15, max_yx_angle=0.0
			),
			object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
				num_objects=40, height=0.25, radius=0.25, max_yx_angle=10.0
			),
			platform_width=1.0,
		),
	}
	return sub_terrains


def _compute_camera(num_rows: int, num_cols: int, sub_terrain_size: float) -> tuple[list[float], list[float]]:
	"""Computes a camera pose that frames the entire terrain grid."""
	total_x = sub_terrain_size * max(1, num_rows)
	total_y = sub_terrain_size * max(1, num_cols)
	extent = max(total_x, total_y)
	eye = [0.0, -0.75 * total_y, 0.65 * extent + 2.0]
	target = [0.0, 0.0, 0.0]
	return eye, target


def design_scene() -> TerrainImporter:
	"""Designs the scene with all terrains in a single grid."""
	# Lights
	light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
	light_cfg.func("/World/Light", light_cfg)

	# Terrain configs
	sub_terrains = _build_sub_terrains(args_cli.sub_terrain_size)
	num_cols = len(sub_terrains)
	num_rows = max(1, args_cli.rows)

	# Ensure a deterministic, full overview across columns.
	if num_rows > 1:
		difficulty_range = (0.0, 1.0)
	else:
		difficulty = max(0.0, min(1.0, args_cli.difficulty))
		difficulty_range = (difficulty, difficulty)

	terrain_gen_cfg = TerrainGeneratorCfg(
		size=(args_cli.sub_terrain_size, args_cli.sub_terrain_size),
		border_width=0.0,
		num_rows=num_rows,
		num_cols=num_cols,
		horizontal_scale=args_cli.horizontal_scale,
		vertical_scale=args_cli.vertical_scale,
		slope_threshold=args_cli.slope_threshold,
		curriculum=True,
		difficulty_range=difficulty_range,
		color_scheme=args_cli.color_scheme,
		use_cache=False,
		sub_terrains=sub_terrains,
	)

	terrain_importer_cfg = TerrainImporterCfg(
		prim_path="/World/ground",
		terrain_type="generator",
		terrain_generator=terrain_gen_cfg,
		num_envs=num_rows * num_cols,
		max_init_terrain_level=None,
		debug_vis=args_cli.debug_vis,
	)

	# Remove visual material for height and random color schemes to use terrain vertex colors.
	if args_cli.color_scheme in ["height", "random"]:
		terrain_importer_cfg.visual_material = None

	terrain_importer = TerrainImporter(terrain_importer_cfg)

	# Print ordering so it is easy to map columns to terrain types.
	print("[INFO]: Terrain order (left-to-right columns):")
	for index, name in enumerate(sub_terrains.keys()):
		print(f"  {index:02d} - {name}")

	return terrain_importer


def run_simulator(sim: sim_utils.SimulationContext):
	"""Runs the simulation loop."""
	while simulation_app.is_running():
		sim.step()


def main():
	"""Main function."""
	sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
	sim = sim_utils.SimulationContext(sim_cfg)

	# Set main camera to frame the terrain grid.
	eye, target = _compute_camera(args_cli.rows, len(_build_sub_terrains(args_cli.sub_terrain_size)), args_cli.sub_terrain_size)
	sim.set_camera_view(eye=eye, target=target)

	# Build the scene and reset the simulator.
	design_scene()
	sim.reset()

	print("[INFO]: Setup complete...")
	run_simulator(sim)


if __name__ == "__main__":
	main()
	simulation_app.close()
