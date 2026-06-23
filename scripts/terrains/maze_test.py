# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a maze map and visualize it in Isaac Sim with basic walls."""

"""Launch Isaac Sim Simulator first."""

"""
scripts/terrains/maze_test.py --width 12 --height 8 --cell_size 0.6 --wall_height 0.5 --wall_thickness 0.05 --algorithm dfs --seed 1
"""


import argparse
import random

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Spawn a maze map as walls in Isaac Sim.")
parser.add_argument("--width", type=int, default=10, help="Number of columns in the maze.")
parser.add_argument("--height", type=int, default=10, help="Number of rows in the maze.")
parser.add_argument("--cell_size", type=float, default=0.6, help="Cell size (m).")
parser.add_argument("--wall_height", type=float, default=0.5, help="Wall height (m).")
parser.add_argument("--wall_thickness", type=float, default=0.05, help="Wall thickness (m).")
parser.add_argument(
	"--ground_mode",
	type=str,
	default="tiled",
	choices=["plane", "tiled", "none"],
	help="Ground mode: 'plane' blocks below z=0, 'tiled' skips stair cells, 'none' spawns no ground.",
)
parser.add_argument("--floor_thickness", type=float, default=0.08, help="Thickness for tiled floor mode (m).")
parser.add_argument(
	"--stairs_prob_per_m2",
	type=float,
	default=0.15,
	help="Expected stair probability per square meter of maze area.",
)
parser.add_argument("--stairs_step_height", type=float, default=0.05, help="Stair step height (m).")
parser.add_argument("--stairs_step_depth", type=float, default=0.18, help="Stair step depth (m).")
parser.add_argument("--stairs_step_width", type=float, default=0.4, help="Stair step width (m).")
parser.add_argument("--stairs_num_steps", type=int, default=4, help="Number of steps per flight.")
parser.add_argument("--stairs_landing_depth", type=float, default=None, help="Landing depth between flights (m).")
parser.add_argument(
	"--stairs_start_down_prob",
	type=float,
	default=0.5,
	help="Probability that a stair starts with a down flight.",
)
parser.add_argument(
	"--stairs_max_length_ratio",
	type=float,
	default=0.9,
	help="Max stair length as a fraction of cell size.",
)
parser.add_argument(
	"--stairs_max_width_ratio",
	type=float,
	default=0.8,
	help="Max stair width as a fraction of cell size.",
)
parser.add_argument(
	"--stairs_hole_margin",
	type=float,
	default=0.002,
	help="Extra margin (m) around stair footprint when carving floor to avoid visual gaps.",
)
parser.add_argument(
	"--boxes_prob_per_m2",
	type=float,
	default=0.12,
	help="Expected boxes-patch probability per square meter of maze area.",
)
parser.add_argument("--boxes_n_boxes", type=int, default=8, help="Max number of boxes per selected cell patch.")
parser.add_argument(
	"--boxes_h_boxes",
	type=float,
	default=0.18,
	help="Max sampled value for box height/width/length in a patch (m).",
)
parser.add_argument(
	"--boxes_plane_size_ratio",
	type=float,
	default=0.9,
	help="Square support plane size as ratio of cell size.",
)
parser.add_argument(
	"--boxes_plane_thickness",
	type=float,
	default=0.02,
	help="Support plane thickness for box patches (m).",
)
parser.add_argument(
	"--rough_prob_per_m2",
	type=float,
	default=0.1,
	help="Expected roughness-patch probability per square meter of maze area.",
)
parser.add_argument("--rough_n_rough", type=int, default=32, help="Max number of rough elements per selected cell patch.")
parser.add_argument(
	"--rough_h_rough",
	type=float,
	default=0.15,
	help="Max sampled value for rough element width/length in a patch (m).",
)
parser.add_argument(
	"--rough_h_rough_height",
	type=float,
	default=0.01,
	help="Max sampled value for rough element height in a patch (m).",
)
parser.add_argument(
	"--rough_patch_size_ratio",
	type=float,
	default=0.9,
	help="Square roughness patch size as ratio of cell size.",
)
parser.add_argument(
	"--algorithm",
	type=str,
	default="dfs",
	choices=["dfs", "kruskal", "prims", "wilson"],
	help="Maze generation algorithm.",
)
parser.add_argument("--seed", type=int, default=0, help="Random seed for maze generation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from labyrinth.generate import DepthFirstSearchGenerator, KruskalsGenerator, PrimsGenerator, WilsonsGenerator
from labyrinth.grid import Direction
from labyrinth.maze import Maze
from boxes import spawn_boxes_on_square
from rough import spawn_roughness_on_square
from stairs import spawn_double_flight_stairs

try:
	from go2_nav.maze_terrain_cfg import MAZE_TERRAIN_CFG, make_maze_terrain_cfg
except ModuleNotFoundError:
	# Fallback for running this script directly from the repository root.
	import sys
	from pathlib import Path

	repo_root = Path(__file__).resolve().parents[2]
	source_pkg = repo_root / "source" / "go2_nav"
	if str(source_pkg) not in sys.path:
		sys.path.append(str(source_pkg))
	from go2_nav.maze_terrain_cfg import MAZE_TERRAIN_CFG, make_maze_terrain_cfg


def _make_generator(name: str):
	"""Create the requested maze generator."""
	if name == "dfs":
		return DepthFirstSearchGenerator()
	if name == "kruskal":
		return KruskalsGenerator()
	if name == "prims":
		return PrimsGenerator()
	if name == "wilson":
		return WilsonsGenerator()
	raise ValueError(f"Unknown maze algorithm: {name}")


def _wall_cfg(size: tuple[float, float, float]):
	"""Creates a cuboid config for static walls."""
	return sim_utils.CuboidCfg(
		size=size,
		rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
		collision_props=sim_utils.CollisionPropertiesCfg(),
		visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), roughness=0.6),
	)


def _spawn_wall(prim_path: str, size: tuple[float, float, float], position: tuple[float, float, float]):
	"""Spawn a single wall cuboid."""
	cfg = _wall_cfg(size)
	cfg.func(prim_path, cfg, translation=position)


def _floor_cfg(size: tuple[float, float, float]):
	"""Creates a cuboid config for static floor tiles."""
	return sim_utils.CuboidCfg(
		size=size,
		rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
		collision_props=sim_utils.CollisionPropertiesCfg(),
		visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), roughness=0.8),
	)


def _spawn_floor(prim_path: str, size: tuple[float, float, float], position: tuple[float, float, float]):
	"""Spawn a single floor tile cuboid."""
	cfg = _floor_cfg(size)
	cfg.func(prim_path, cfg, translation=position)


def _spawn_floor_ring_around_hole(
	prim_prefix: str,
	center: tuple[float, float],
	cell_size: float,
	hole_length: float,
	hole_width: float,
	direction: Direction,
	floor_thickness: float,
	id_start: int,
) -> int:
	"""Spawn non-overlapping floor patches around a rectangular hole aligned with stair direction."""
	eps = 1e-5
	id_cur = id_start
	cx, cy = center
	half_cell = 0.5 * cell_size

	if direction in (Direction.E, Direction.W):
		hole_half_x = 0.5 * hole_length
		hole_half_y = 0.5 * hole_width
	else:
		hole_half_x = 0.5 * hole_width
		hole_half_y = 0.5 * hole_length

	x_min = cx - half_cell
	x_max = cx + half_cell
	y_min = cy - half_cell
	y_max = cy + half_cell
	hx_min = cx - hole_half_x
	hx_max = cx + hole_half_x
	hy_min = cy - hole_half_y
	hy_max = cy + hole_half_y

	def spawn_rect(x0: float, x1: float, y0: float, y1: float):
		nonlocal id_cur
		sx = x1 - x0
		sy = y1 - y0
		if sx <= eps or sy <= eps:
			return
		size = (sx, sy, floor_thickness)
		pos = (0.5 * (x0 + x1), 0.5 * (y0 + y1), -0.5 * floor_thickness)
		_spawn_floor(f"{prim_prefix}/Floor_{id_cur}", size, pos)
		id_cur += 1

	# Left and right caps
	spawn_rect(x_min, hx_min, y_min, y_max)
	spawn_rect(hx_max, x_max, y_min, y_max)
	# Bottom and top strips over hole x-range
	spawn_rect(hx_min, hx_max, y_min, hy_min)
	spawn_rect(hx_min, hx_max, hy_max, y_max)

	return id_cur


def _compute_camera(width: int, height: int, cell_size: float) -> tuple[list[float], list[float]]:
	"""Compute a camera view that frames the full maze."""
	span_x = width * cell_size
	span_y = height * cell_size
	extent = max(span_x, span_y)
	eye = [0.0, -1.4 * extent, 0.9 * extent + 0.5]
	target = [0.0, 0.0, 0.0]
	return eye, target


def design_scene():
	"""Design the maze scene with static wall cuboids."""
	# Build a reusable TerrainGeneratorCfg in the same style as ROUGH_TERRAINS_CFG.
	maze_terrain_cfg = make_maze_terrain_cfg(
		maze_rows=args_cli.height,
		maze_cols=args_cli.width,
		cell_size=args_cli.cell_size,
		wall_height=args_cli.wall_height,
		wall_thickness=args_cli.wall_thickness,
		floor_thickness=0.0,
		algorithm=args_cli.algorithm,
		seed=args_cli.seed,
	)
	print("[INFO]: MAZE_TERRAIN_CFG runtime object:")
	print(maze_terrain_cfg)

	# Ground-plane (optional). A global plane at z=0 blocks any descending stair geometry.
	if args_cli.ground_mode == "plane":
		ground_cfg = sim_utils.GroundPlaneCfg()
		ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
	# Lights
	light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
	light_cfg.func("/World/Light", light_cfg)

	# Maze root prim
	sim_utils.create_prim("/World/Maze", "Xform")
	sim_utils.create_prim("/World/Maze/Stairs", "Xform")
	sim_utils.create_prim("/World/Maze/Boxes", "Xform")
	sim_utils.create_prim("/World/Maze/Rough", "Xform")
	if args_cli.ground_mode == "tiled":
		sim_utils.create_prim("/World/Maze/Floor", "Xform")

	# Build the maze
	if args_cli.seed is not None:
		random.seed(args_cli.seed)
	generator = _make_generator(args_cli.algorithm)
	maze = Maze(width=args_cli.width, height=args_cli.height, generator=generator)
	print("[INFO]: Generated maze")
	print(maze)

	cell_size = args_cli.cell_size
	wall_height = args_cli.wall_height
	wall_thickness = args_cli.wall_thickness

	maze_origin_x = -0.5 * args_cli.width * cell_size
	maze_origin_y = -0.5 * args_cli.height * cell_size

	wall_id = 0
	stairs_id = 0
	boxes_id = 0
	rough_id = 0
	stairs_cells: set[tuple[int, int]] = set()
	boxes_cells: set[tuple[int, int]] = set()
	stairs_footprints: dict[tuple[int, int], tuple[Direction, float, float]] = {}
	cell_area = cell_size * cell_size
	stairs_prob = max(0.0, min(1.0, args_cli.stairs_prob_per_m2 * cell_area))
	boxes_prob = max(0.0, min(1.0, args_cli.boxes_prob_per_m2 * cell_area))
	rough_prob = max(0.0, min(1.0, args_cli.rough_prob_per_m2 * cell_area))
	max_stairs_length = cell_size * max(0.1, args_cli.stairs_max_length_ratio)
	max_stairs_width = cell_size * max(0.1, args_cli.stairs_max_width_ratio)
	boxes_plane_size = cell_size * max(0.1, min(1.0, args_cli.boxes_plane_size_ratio))
	rough_patch_size = cell_size * max(0.1, min(1.0, args_cli.rough_patch_size_ratio))
	floor_thickness = max(1e-3, args_cli.floor_thickness)
	hole_margin = max(0.0, args_cli.stairs_hole_margin)

	for row in range(args_cli.height):
		for col in range(args_cli.width):
			cell = maze.get_cell(row, col)
			cx = maze_origin_x + (col + 0.5) * cell_size
			cy = maze_origin_y + (row + 0.5) * cell_size

			# North wall
			if Direction.N not in cell.open_walls:
				wy = maze_origin_y + row * cell_size
				size = (cell_size + wall_thickness, wall_thickness, wall_height)
				pos = (cx, wy, wall_height * 0.5)
				_spawn_wall(f"/World/Maze/Wall_{wall_id}", size, pos)
				wall_id += 1

			# West wall
			if Direction.W not in cell.open_walls:
				wx = maze_origin_x + col * cell_size
				size = (wall_thickness, cell_size + wall_thickness, wall_height)
				pos = (wx, cy, wall_height * 0.5)
				_spawn_wall(f"/World/Maze/Wall_{wall_id}", size, pos)
				wall_id += 1

			# South boundary
			if row == args_cli.height - 1 and Direction.S not in cell.open_walls:
				wy = maze_origin_y + (row + 1) * cell_size
				size = (cell_size + wall_thickness, wall_thickness, wall_height)
				pos = (cx, wy, wall_height * 0.5)
				_spawn_wall(f"/World/Maze/Wall_{wall_id}", size, pos)
				wall_id += 1

			# East boundary
			if col == args_cli.width - 1 and Direction.E not in cell.open_walls:
				wx = maze_origin_x + (col + 1) * cell_size
				size = (wall_thickness, cell_size + wall_thickness, wall_height)
				pos = (wx, cy, wall_height * 0.5)
				_spawn_wall(f"/World/Maze/Wall_{wall_id}", size, pos)
				wall_id += 1

			# Stairs (aligned with an open corridor direction)
			if cell.open_walls and random.random() < stairs_prob:
				stairs_dir = random.choice(list(cell.open_walls))
				start_down = random.random() < args_cli.stairs_start_down_prob
				footprint = spawn_double_flight_stairs(
					prim_path=f"/World/Maze/Stairs/Stairs_{stairs_id}",
					center=(cx, cy, 0.0),
					direction=stairs_dir,
					step_height=args_cli.stairs_step_height,
					step_depth=args_cli.stairs_step_depth,
					step_width=args_cli.cell_size,
					num_steps=args_cli.stairs_num_steps,
					landing_depth=args_cli.stairs_landing_depth,
					start_down=start_down,
					max_length=max_stairs_length,
					max_width=max_stairs_width,
					return_footprint=True,
				)
				stairs_cells.add((row, col))
				if footprint is not None:
					stairs_footprints[(row, col)] = (
						stairs_dir,
						min(cell_size, footprint["length"] + 2.0 * hole_margin),
						min(cell_size, footprint["width"] + 2.0 * hole_margin),
					)
				stairs_id += 1

			# Boxes patch in the cell; maze ground handles support, so no local plane here.
			if (row, col) not in stairs_cells and random.random() < boxes_prob:
				spawn_boxes_on_square(
					prim_path=f"/World/Maze/Boxes/Boxes_{boxes_id}",
					center=(cx, cy, 0.0),
					plane_size=boxes_plane_size,
					n_boxes=args_cli.boxes_n_boxes,
					h_boxes=args_cli.boxes_h_boxes,
					plane_thickness=args_cli.boxes_plane_thickness,
					seed=args_cli.seed * 100_003 + boxes_id,
					spawn_plane=False,
				)
				boxes_cells.add((row, col))
				boxes_id += 1

			# Roughness patch on maze-defined ground (no local plane).
			if (row, col) not in stairs_cells and (row, col) not in boxes_cells and random.random() < rough_prob:
				spawn_roughness_on_square(
					prim_path=f"/World/Maze/Rough/Rough_{rough_id}",
					center=(cx, cy, 0.0),
					patch_size=rough_patch_size,
					n_rough=args_cli.rough_n_rough,
					h_rough_xy=args_cli.rough_h_rough,
					h_rough_z=args_cli.rough_h_rough_height,
					seed=args_cli.seed * 170_003 + rough_id,
					spawn_plane=False,
				)
				rough_id += 1


	# Tile-based floor keeps walkable map height while leaving stair cells open to go below z=0.
	if args_cli.ground_mode == "tiled":
		floor_id = 0
		for row in range(args_cli.height):
			for col in range(args_cli.width):
				cx = maze_origin_x + (col + 0.5) * cell_size
				cy = maze_origin_y + (row + 0.5) * cell_size
				if (row, col) in stairs_cells and (row, col) in stairs_footprints:
					stairs_dir, hole_length, hole_width = stairs_footprints[(row, col)]
					floor_id = _spawn_floor_ring_around_hole(
						prim_prefix="/World/Maze/Floor",
						center=(cx, cy),
						cell_size=cell_size,
						hole_length=hole_length,
						hole_width=hole_width,
						direction=stairs_dir,
						floor_thickness=floor_thickness,
						id_start=floor_id,
					)
				else:
					size = (cell_size, cell_size, floor_thickness)
					pos = (cx, cy, -0.5 * floor_thickness)
					_spawn_floor(f"/World/Maze/Floor/Floor_{floor_id}", size, pos)
					floor_id += 1


def run_simulator(sim: sim_utils.SimulationContext):
	"""Runs the simulation loop."""
	while simulation_app.is_running():
		sim.step()


def main():
	"""Main function."""
	sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
	sim = sim_utils.SimulationContext(sim_cfg)

	eye, target = _compute_camera(args_cli.width, args_cli.height, args_cli.cell_size)
	sim.set_camera_view(eye=eye, target=target)

	design_scene()
	sim.reset()
	print("[INFO]: Setup complete...")
	run_simulator(sim)


if __name__ == "__main__":
	main()
	simulation_app.close()