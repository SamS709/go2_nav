# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Maze terrain generator config usable like Isaac Lab's ROUGH_TERRAINS_CFG."""

from __future__ import annotations

import math
import random
from typing import Literal

import numpy as np
import torch
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg, FlatPatchSamplingCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from labyrinth.generate import DepthFirstSearchGenerator, KruskalsGenerator, PrimsGenerator, WilsonsGenerator
from labyrinth.grid import Direction
from labyrinth.maze import Maze

NUM_TERRAIN_TYPES = 5

class _MazeRegistry:
    """Side-channel storing maze layouts keyed by (terrain_generator_id, row, col)."""
    def __init__(self):
        self._mazes: list[torch.Tensor] = [] 
        self._num_rows: list[int] = []
        self._terrain_num_cols: int
        self._terrain_num_rows: int
        self._maze_max_cols: int
        self._maze_max_rows: int
        self.num_terrain_types = NUM_TERRAIN_TYPES
        
    def set_dims(self, terrain_rows: int, terrain_cols: int, maze_max_rows: int, maze_max_cols: int, device):
        self._terrain_num_rows, self._terrain_num_cols = terrain_rows, terrain_cols
        self._maze_max_rows, self._maze_max_cols = maze_max_rows, maze_max_cols
        self.device = device
        self._build_mazes_tensor()
        
    def get_mazes_terrain_coords(self, terrain_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            terrain_coords: Tensor of shape (num_envs, 2) containing [row, col] for each env.
        Returns:
            Tensor of shape (num_envs, max_maze_rows, max_maze_cols) containing the maze layouts.
        """
        # Extract rows and cols separately
        # terrain_coords[:, 0] gives all rows, terrain_coords[:, 1] gives all cols
        rows = terrain_coords[:, 0].long()
        cols = terrain_coords[:, 1].long()
        
        num_envs = terrain_coords.shape[0]
        
        # Initialize output tensor
        mazes_ordered = torch.zeros(
            [num_envs, self._maze_max_rows, self._maze_max_cols, 5], 
            device=self._mazes_tensor.device,
            dtype=self._mazes_tensor.dtype
        )
        
        # Use advanced indexing: [rows, cols] fetches the specific (row, col) entry for each env
        # This works because 'rows' and 'cols' are both 1D tensors of length num_envs.
        # The result shape is automatically (num_envs, max_maze_rows, max_maze_cols)
        mazes_ordered = self._mazes_tensor[rows, cols]
        
        return mazes_ordered
        
    def _build_mazes_tensor(self):
        self._mazes_tensor = -torch.ones([self._terrain_num_rows, self._terrain_num_cols, self._maze_max_rows, self._maze_max_cols, 5])
        for i in range(self._terrain_num_rows):
            for j in range(self._terrain_num_cols):
                assert self._mazes_tensor[i,j].shape == self.maze_at(i,j).shape
                self._mazes_tensor[i,j] = self.maze_at(i,j)
        self._mazes_tensor = self._mazes_tensor.to(self.device)
            

    def record(self, maze_tensor: torch.Tensor, num_row: int):
        self._mazes.append(maze_tensor)
        self._num_rows.append(num_row)
        
    def maze_at(self, row: int, col: int) -> torch.Tensor:
        """Reassemble into (num_rows, num_cols, ...) using TerrainGenerator's known call order."""
        # TerrainGenerator iterates row-major (sub_terrains call order); see step (c) for the
        # exact assumption here and why it's safe.
        # 0,1 = 1 // 1, 1 = 4 (car terrain num_cols = 3)
        # 5 => 2, 1 = 1 * 3 + 2 OK 
        return self._mazes[col * self._terrain_num_rows + row]
    
    def get_num_rows(self, terrain_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            terrain_coords: Tensor of shape (num_envs, 2) containing [row, col] for each env.
        Returns:
            Tensor of shape (num_envs,) containing the actual maze row count for each env.
        """
        rows = terrain_coords[:, 0].long()
        cols = terrain_coords[:, 1].long()

        # Same flat indexing as maze_at
        flat_indices = cols * self._terrain_num_rows + rows  # (num_envs,)

        num_rows_tensor = torch.tensor(self._num_rows, device=self.device, dtype=torch.long)
        return num_rows_tensor[flat_indices]  # (num_envs,)
        
    
    def as_list(self) -> list:
        return self._mazes
    
    
   

MAZE_REGISTRY = _MazeRegistry()

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


def _clamp01(value: float) -> float:
    """Clamp scalar to [0, 1]."""
    return max(0.0, min(1.0, value))


def _sample_float(
    rng: random.Random,
    default_value: float,
    value_range: tuple[float, float] | None,
    min_value: float = 1e-4,
) -> float:
    """Sample a float from a range, or use default if range is not provided."""
    if value_range is None:
        return max(min_value, float(default_value))
    v_min = min(float(value_range[0]), float(value_range[1]))
    v_max = max(float(value_range[0]), float(value_range[1]))
    return max(min_value, rng.uniform(v_min, v_max))


def _sample_int(
    rng: random.Random,
    default_value: int,
    value_range: tuple[int, int] | None,
    min_value: int = 1,
) -> int:
    """Sample an int from a range, or use default if range is not provided."""
    if value_range is None:
        return max(min_value, int(default_value))
    v_min = min(int(value_range[0]), int(value_range[1]))
    v_max = max(int(value_range[0]), int(value_range[1]))
    v_min = max(min_value, v_min)
    v_max = max(v_min, v_max)
    return rng.randint(v_min, v_max)


def _direction_yaw(direction: Direction) -> float:
    """Map maze direction to yaw in radians."""
    if direction == Direction.E:
        return 0.0
    if direction == Direction.S:
        return math.pi * 0.5
    if direction == Direction.W:
        return math.pi
    if direction == Direction.N:
        return -math.pi * 0.5
    return 0.0


def _rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    """Rotate a local XY point by yaw."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y, s * x + c * y


def _box_mesh(
    size: tuple[float, float, float], position: tuple[float, float, float], yaw: float = 0.0
) -> trimesh.Trimesh:
    """Create a single box mesh with optional yaw rotation around z-axis."""
    transform = trimesh.transformations.translation_matrix(position)
    if abs(yaw) > 1e-9:
        rot = trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0])
        transform = trimesh.transformations.concatenate_matrices(transform, rot)
    return trimesh.creation.box(extents=size, transform=transform)


def _append_stairs_meshes(
    meshes: list[trimesh.Trimesh],
    center: tuple[float, float, float],
    direction: Direction,
    step_height: float,
    step_depth: float,
    step_width: float,
    num_steps: int,
    landing_depth: float | None,
    start_down: bool,
    max_length: float | None,
    max_rows: float | None,
) -> tuple[float, float]:
    """Append a double-flight stair mesh sequence and return (length, width) footprint."""
    if num_steps < 1:
        return 0.0, 0.0

    step_height = max(1e-4, step_height)
    step_depth = max(1e-4, step_depth)

    # Stair footprint is fixed by the cell dimensions when max_length/max_rows are provided.
    if max_rows is not None:
        step_width = max(1e-4, max_rows)
    else:
        step_width = max(1e-4, step_width)

    if max_length is not None:
        total_length = max(1e-4, max_length)
    else:
        base_landing = step_depth if landing_depth is None else max(0.0, landing_depth)
        total_length = max(1e-4, 2.0 * num_steps * step_depth + base_landing)

    # Clamp number of steps to what fits the fixed stair length for sampled step depth.
    max_steps_fit = int(total_length // (2.0 * step_depth))
    if max_steps_fit < 1:
        max_steps_fit = 1
        step_depth = total_length * 0.5
    num_steps = max(1, min(num_steps, max_steps_fit))

    # Fill the remaining length with a landing section.
    landing_depth = max(0.0, total_length - 2.0 * num_steps * step_depth)

    yaw = _direction_yaw(direction)
    base_z = center[2]
    sign = -1.0 if start_down else 1.0

    def append_box(local_x: float, top_height: float, box_x: float):
        cx, cy = _rotate_xy(local_x, 0.0, yaw)
        center_z = top_height - 0.5 * step_height
        meshes.append(
            _box_mesh(
                (box_x, step_width, step_height),
                (center[0] + cx, center[1] + cy, center_z),
                yaw=yaw,
            )
        )

    flight_length = num_steps * step_depth
    start_x = -0.5 * total_length + 0.5 * step_depth
    landing_x = start_x + flight_length - 0.5 * step_depth + 0.5 * landing_depth
    flight2_x = start_x + flight_length + landing_depth

    # First flight
    for i in range(num_steps):
        top_height = base_z + sign * step_height * (i + 1)
        local_x = start_x + step_depth * i
        append_box(local_x, top_height, step_depth)

    # Landing
    landing_top = base_z + sign * step_height * num_steps
    append_box(landing_x, landing_top, landing_depth)

    # Second flight (symmetric return)
    sign = -sign
    for i in range(num_steps):
        top_height = landing_top + sign * step_height * (i + 1)
        local_x = flight2_x + step_depth * i
        append_box(local_x, top_height, step_depth)

    return total_length, step_width


def _append_boxes_meshes(
    meshes: list[trimesh.Trimesh],
    rng: random.Random,
    center: tuple[float, float, float],
    patch_size: float,
    n_boxes: int,
    h_boxes: float,
    min_size: float,
) -> None:
    """Append random box obstacles inside a square patch."""
    if n_boxes <= 0:
        return
    n_spawn = rng.randint(0, n_boxes)
    half_patch = 0.5 * patch_size
    min_size = max(1e-4, min_size)

    for _ in range(n_spawn):
        lx = max(min_size, rng.uniform(0.0, h_boxes))
        ly = max(min_size, rng.uniform(0.0, h_boxes))
        lz = max(min_size, rng.uniform(0.0, h_boxes))

        x_margin = max(0.0, half_patch - 0.5 * lx)
        y_margin = max(0.0, half_patch - 0.5 * ly)
        x = center[0] + rng.uniform(-x_margin, x_margin)
        y = center[1] + rng.uniform(-y_margin, y_margin)
        z = center[2] + 0.5 * lz

        meshes.append(_box_mesh((lx, ly, lz), (x, y, z)))


def _append_rough_meshes(
    meshes: list[trimesh.Trimesh],
    rng: random.Random,
    center: tuple[float, float, float],
    patch_size: float,
    n_rough: int,
    h_rough_xy: float,
    h_rough_z: float,
    min_size: float,
) -> None:
    """Append random roughness elements inside a square patch."""
    if n_rough <= 0:
        return
    n_spawn = rng.randint(0, n_rough)
    half_patch = 0.5 * patch_size
    min_size = max(1e-4, min_size)

    for _ in range(n_spawn):
        lx = max(min_size, rng.uniform(0.0, h_rough_xy))
        ly = max(min_size, rng.uniform(0.0, h_rough_xy))
        lz = max(min_size, rng.uniform(0.0, h_rough_z))

        x_margin = max(0.0, half_patch - 0.5 * lx)
        y_margin = max(0.0, half_patch - 0.5 * ly)
        x = center[0] + rng.uniform(-x_margin, x_margin)
        y = center[1] + rng.uniform(-y_margin, y_margin)
        z = center[2] + 0.5 * lz

        meshes.append(_box_mesh((lx, ly, lz), (x, y, z)))


def _corridor_stairs_direction(open_walls: set[Direction]) -> Direction | None:
    """Return the corridor direction for a straight cell, or None for corners/junctions."""
    if len(open_walls) != 2:
        return None
    if Direction.E in open_walls and Direction.W in open_walls:
        return Direction.E
    if Direction.N in open_walls and Direction.S in open_walls:
        return Direction.N
    return None


def _append_floor_rect(
    meshes: list[trimesh.Trimesh],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    floor_thickness: float,
) -> None:
    """Append one floor rectangle box if dimensions are positive."""
    sx = x1 - x0
    sy = y1 - y0
    if sx <= 1e-6 or sy <= 1e-6:
        return
    meshes.append(
        _box_mesh(
            (sx, sy, floor_thickness),
            (0.5 * (x0 + x1), 0.5 * (y0 + y1), -0.5 * floor_thickness),
        )
    )


def _append_floor_ring_with_hole(
    meshes: list[trimesh.Trimesh],
    cx: float,
    cy: float,
    cell_w: float,
    cell_h: float,
    hole_length: float,
    hole_width: float,
    direction: Direction,
    floor_thickness: float,
) -> None:
    """Append non-overlapping floor pieces around a stair-aligned rectangular hole."""
    half_x = 0.5 * cell_w
    half_y = 0.5 * cell_h

    if direction in (Direction.E, Direction.W):
        hole_half_x = min(half_x, 0.5 * hole_length)
        hole_half_y = min(half_y, 0.5 * hole_width)
    else:
        hole_half_x = min(half_x, 0.5 * hole_width)
        hole_half_y = min(half_y, 0.5 * hole_length)

    x_min, x_max = cx - half_x, cx + half_x
    y_min, y_max = cy - half_y, cy + half_y
    hx_min, hx_max = cx - hole_half_x, cx + hole_half_x
    hy_min, hy_max = cy - hole_half_y, cy + hole_half_y

    # Left and right caps.
    _append_floor_rect(meshes, x_min, hx_min, y_min, y_max, floor_thickness)
    _append_floor_rect(meshes, hx_max, x_max, y_min, y_max, floor_thickness)
    # Bottom and top strips over hole x-range.
    _append_floor_rect(meshes, hx_min, hx_max, y_min, hy_min, floor_thickness)
    _append_floor_rect(meshes, hx_min, hx_max, hy_max, y_max, floor_thickness)


def maze_terrain(
    difficulty: float, cfg: "MeshMazeTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a maze terrain mesh from Labyrinth.

    The curriculum difficulty controls maze width (in cells), while maze height is
    sampled in a fixed range.
    """
    difficulty = _clamp01(float(difficulty))

    if cfg.cell_size <= 0.0:
        raise ValueError("cell_size must be positive")
    if cfg.wall_height <= 0.0:
        raise ValueError("wall_height must be positive")
    if cfg.wall_thickness <= 0.0:
        raise ValueError("wall_thickness must be positive")


    # Sample wall destroy
    p_wall_dest = cfg.p_wall_dest
     

    max_rows = max(1, int(round(cfg.maze_max_rows)))
    maze_rows = int(difficulty * cfg.maze_max_rows)
    maze_rows = max(1, min(max_rows, maze_rows))
    
    maze_cols = max(1, int(round(cfg.maze_max_cols)))
    
    # Keep maze cell size fixed; difficulty changes cell count, not physical cell dimensions.
    cell_w = cfg.cell_size
    cell_h = cfg.cell_size

    maze_size_x = maze_cols * cell_w
    maze_size_y = maze_rows * cell_h
    offset_x = 0.5 * (cfg.size[0] - maze_size_x)
    offset_y = 0.5 * (cfg.size[1] - maze_size_y)

    maze = Maze(width=maze_cols, height=maze_rows, generator=_make_generator(cfg.algorithm))

    # Dedicated RNG for feature placement.
    rng = random.Random(cfg.seed)

    # Sample tile-level parameters from configured ranges.
    wall_height = _sample_float(rng, cfg.wall_height, cfg.wall_height_range)
    wall_thickness = _sample_float(rng, cfg.wall_thickness, cfg.wall_thickness_range)

    sampled_stairs_prob_per_m2 = _sample_float(
        rng, cfg.stairs_prob_per_m2, cfg.stairs_prob_per_m2_range, min_value=0.0
    )
    sampled_boxes_prob_per_m2 = _sample_float(
        rng, cfg.boxes_prob_per_m2, cfg.boxes_prob_per_m2_range, min_value=0.0
    )
    sampled_rough_prob_per_m2 = _sample_float(
        rng, cfg.rough_prob_per_m2, cfg.rough_prob_per_m2_range, min_value=0.0
    )

    sampled_boxes_n_boxes = _sample_int(rng, cfg.boxes_n_boxes, cfg.boxes_n_boxes_range, min_value=0)
    sampled_boxes_h_boxes = _sample_float(rng, cfg.boxes_h_boxes, cfg.boxes_h_boxes_range)
    sampled_boxes_patch_size_ratio = _sample_float(
        rng, cfg.boxes_patch_size_ratio, cfg.boxes_patch_size_ratio_range, min_value=0.0
    )

    sampled_rough_n_rough = _sample_int(rng, cfg.rough_n_rough, cfg.rough_n_rough_range, min_value=0)
    sampled_rough_h_rough_xy = _sample_float(rng, cfg.rough_h_rough_xy, cfg.rough_h_rough_xy_range)
    sampled_rough_h_rough_z = _sample_float(rng, cfg.rough_h_rough_z, cfg.rough_h_rough_z_range)
    sampled_rough_patch_size_ratio = _sample_float(
        rng, cfg.rough_patch_size_ratio, cfg.rough_patch_size_ratio_range, min_value=0.0
    )

    # Per-cell probabilities from density definitions.
    cell_area = cell_w * cell_h
    stairs_prob = _clamp01(sampled_stairs_prob_per_m2 * cell_area)
    boxes_prob = _clamp01(sampled_boxes_prob_per_m2 * cell_area)
    rough_prob = _clamp01(sampled_rough_prob_per_m2 * cell_area)

    # Reserve a guaranteed spawn cell so origin is always at a cell center and unobstructed.
    spawn_cell = (maze_rows // 2, maze_cols // 2)

    meshes: list[trimesh.Trimesh] = []
    stairs_cells: set[tuple[int, int]] = set()
    boxes_cells: set[tuple[int, int]] = set()
    stairs_footprints: dict[tuple[int, int], tuple[Direction, float, float]] = {}

    max_cell_span = min(cell_w, cell_h)
    boxes_patch_size = max_cell_span * max(0.1, min(1.0, sampled_boxes_patch_size_ratio))
    rough_patch_size = max_cell_span * max(0.1, min(1.0, sampled_rough_patch_size_ratio))
    
    # maze_array[row, col, 0] -> terrain_type in [0,1,2,3,4]:
    #   - 0 -> flat
    #   - 1 -> stairs
    #   - 2 -> boxes
    #   - 3 -> grid
    #   - 4 -> stairs
    # maze_array[row, col, 1] -> North wall in [0,1] 1 if exixts else 0
    # maze_array[row, col, 2] -> North wall in [0,1] 1 if exixts else 0
    # maze_array[row, col, 3] -> North wall in [0,1] 1 if exixts else 0
    # maze_array[row, col, 4] -> North wall in [0,1] 1 if exixts else 0
    maze_array = torch.zeros((int(cfg.maze_max_rows), int(cfg.maze_max_cols), 5))
    
    

    for row in range(maze_rows):
        for col in range(maze_cols):
            cell = maze.get_cell(row, col)
            cx = offset_x + (col + 0.5) * cell_w
            cy = offset_y + (row + 0.5) * cell_h
            
            # ===========================================================================
            # WALLS 
            # North wall
            dest_wall_cond = _sample_float(rng, 0.0, (0.0, 1.0)) < p_wall_dest and row != 0 and row != maze_rows - 1 and col != 0 and col != maze_cols - 1
            if Direction.N not in cell.open_walls and not dest_wall_cond:
                meshes.append(
                    _box_mesh(
                        (cell_w + wall_thickness, wall_thickness, wall_height),
                        (cx, offset_y + row * cell_h, 0.5 * wall_height),
                    )
                )
                maze_array[row, col, 1] = 1
                
            # East boundary
            dest_wall_cond = _sample_float(rng, 0.0, (0.0, 1.0)) < p_wall_dest and row != 0 and row != maze_rows - 1 and col != 0 and col != maze_cols - 1            
            if col == maze_cols - 1 and Direction.E not in cell.open_walls and not dest_wall_cond:
                meshes.append(
                    _box_mesh(
                        (wall_thickness, cell_h + wall_thickness, wall_height),
                        (offset_x + (col + 1) * cell_w, cy, 0.5 * wall_height),
                    )
                )
                maze_array[row, col, 2] = 1
                
            # South boundary
            dest_wall_cond = _sample_float(rng, 0.0, (0.0, 1.0)) < p_wall_dest and row != 0 and row != maze_rows - 1 and col != 0 and col != maze_cols - 1            
            if row == maze_rows - 1 and Direction.S not in cell.open_walls and not dest_wall_cond:
                meshes.append(
                    _box_mesh(
                        (cell_w + wall_thickness, wall_thickness, wall_height),
                        (cx, offset_y + (row + 1) * cell_h, 0.5 * wall_height),
                    )
                )
                maze_array[row, col, 3] = 1

            # West wall
            dest_wall_cond = _sample_float(rng, 0.0, (0.0, 1.0)) < p_wall_dest and row != 0 and row != maze_rows - 1 and col != 0 and col != maze_cols - 1
            if Direction.W not in cell.open_walls and not dest_wall_cond:
                meshes.append(
                    _box_mesh(
                        (wall_thickness, cell_h + wall_thickness, wall_height),
                        (offset_x + col * cell_w, cy, 0.5 * wall_height),
                    )
                )
                maze_array[row, col, 4] = 1

            # ===========================================================================
            # STAIRS 
            # only in straight corridor cells with two parallel walls.
            stairs_dir = _corridor_stairs_direction(cell.open_walls)
            if (row, col) != spawn_cell and stairs_dir is not None and rng.random() < stairs_prob:
                start_down = rng.random() < cfg.stairs_start_down_prob

                min_step_height = 0.04 * max_cell_span
                min_step_depth = 0.16 * max_cell_span
                step_height = _sample_float(
                    rng, cfg.stairs_step_height, cfg.stairs_step_height_range, min_value=min_step_height
                )
                step_depth = _sample_float(
                    rng, cfg.stairs_step_depth, cfg.stairs_step_depth_range, min_value=min_step_depth
                )
                num_steps = _sample_int(rng, cfg.stairs_num_steps, cfg.stairs_num_steps_range)

                # Fixed footprint: stairs always fill the current cell in width and length.
                if stairs_dir in (Direction.E, Direction.W):
                    stair_total_length = cell_w
                    stair_total_width = cell_h
                else:
                    stair_total_length = cell_h
                    stair_total_width = cell_w

                footprint = _append_stairs_meshes(
                    meshes=meshes,
                    center=(cx, cy, 0.0),
                    direction=stairs_dir,
                    step_height=step_height,
                    step_depth=step_depth,
                    step_width=stair_total_width,
                    num_steps=num_steps,
                    landing_depth=cfg.stairs_landing_depth,
                    start_down=start_down,
                    max_length=stair_total_length,
                    max_rows=stair_total_width,
                )
                stairs_cells.add((row, col))
                
                hole_margin = max(0.0, cfg.stairs_hole_margin)
                stairs_footprints[(row, col)] = (
                    stairs_dir,
                    footprint[0] + 2.0 * hole_margin,
                    footprint[1] + 2.0 * hole_margin,
                )
                maze_array[row, col, 0] = 1
            # ===========================================================================
            # BOXES
            if (row, col) != spawn_cell and (row, col) not in stairs_cells and rng.random() < boxes_prob:
                _append_boxes_meshes(
                    meshes=meshes,
                    rng=rng,
                    center=(cx, cy, 0.0),
                    patch_size=boxes_patch_size,
                    n_boxes=sampled_boxes_n_boxes,
                    h_boxes=sampled_boxes_h_boxes,
                    min_size=cfg.boxes_min_size,
                )
                boxes_cells.add((row, col))
                maze_array[row, col, 0] = 2

            # GRIDS
            if (
                (row, col) != spawn_cell
                and (row, col) not in stairs_cells
                and (row, col) not in boxes_cells
                and rng.random() < rough_prob
            ):
                _append_rough_meshes(
                    meshes=meshes,
                    rng=rng,
                    center=(cx, cy, 0.0),
                    patch_size=rough_patch_size,
                    n_rough=sampled_rough_n_rough,
                    h_rough_xy=sampled_rough_h_rough_xy,
                    h_rough_z=sampled_rough_h_rough_z,
                    min_size=cfg.rough_min_size,
                )
                maze_array[row, col, 0] = 3
    # ===========================================================================
    # FLOORS
    # no floor if there is stairs
    if cfg.floor_thickness > 0.0:
        for row in range(maze_rows):
            for col in range(maze_cols):
                cx = (col + 0.5) * cell_w
                cy = (row + 0.5) * cell_h
                cx = offset_x + (col + 0.5) * cell_w
                cy = offset_y + (row + 0.5) * cell_h
                if (row, col) in stairs_footprints:
                    stairs_dir, hole_length, hole_width = stairs_footprints[(row, col)]
                    _append_floor_ring_with_hole(
                        meshes=meshes,
                        cx=cx,
                        cy=cy,
                        cell_w=cell_w,
                        cell_h=cell_h,
                        hole_length=hole_length,
                        hole_width=hole_width,
                        direction=stairs_dir,
                        floor_thickness=cfg.floor_thickness,
                    )
                else:
                    _append_floor_rect(
                        meshes,
                        x0=offset_x + col * cell_w,
                        x1=offset_x + (col + 1) * cell_w,
                        y0=offset_y + row * cell_h,
                        y1=offset_y + (row + 1) * cell_h,
                        floor_thickness=cfg.floor_thickness,
                    )

        # Fill the area outside the centered maze so the whole tile remains walkable.
        x_min_maze = offset_x
        x_max_maze = offset_x + maze_size_x
        y_min_maze = offset_y
        y_max_maze = offset_y + maze_size_y

        _append_floor_rect(meshes, 0.0, x_min_maze, 0.0, cfg.size[1], cfg.floor_thickness)
        _append_floor_rect(meshes, x_max_maze, cfg.size[0], 0.0, cfg.size[1], cfg.floor_thickness)
        _append_floor_rect(meshes, x_min_maze, x_max_maze, 0.0, y_min_maze, cfg.floor_thickness)
        _append_floor_rect(meshes, x_min_maze, x_max_maze, y_max_maze, cfg.size[1], cfg.floor_thickness)

    # Robot spawn origin is the center of a reserved maze cell.
    spawn_row, spawn_col = spawn_cell
    origin = np.array([offset_x + (spawn_col + 0.5) * cell_w, offset_y + (spawn_row + 0.5) * cell_h, 0.0])

    # --- IsaacLab axis convention fix -------------------------------------------------
    # Everything above builds the maze with `row` driving the y-extent and `col` driving
    # the x-extent (so curriculum difficulty, which scales maze_rows, grows the maze along
    # y). IsaacLab's terrain convention expects rows -> x, cols -> y. Rather than touching
    # any of the generation logic above (walls, stairs, boxes, roughness, floor holes -
    # all of which are internally self-consistent), rotate the finished tile 90 degrees
    # about its own center so the net effect is row -> x, col -> y, with no change to mesh
    # winding/normals (a rotation, unlike a raw x<->y swap, preserves handedness).
    #
    # This rotation only preserves the tile footprint when the tile is square; a
    # non-square tile would have its footprint rotated too, breaking alignment with
    # neighboring tiles. Fail loudly rather than silently misalign.
    if not np.isclose(cfg.size[0], cfg.size[1]):
        raise ValueError(
            "maze_terrain's axis-convention fix assumes a square tile "
            f"(cfg.size[0] == cfg.size[1]); got size={cfg.size}."
        )

    cx_tile = 0.5 * cfg.size[0]
    cy_tile = 0.5 * cfg.size[1]

    def _rotate90_xy(points: np.ndarray) -> np.ndarray:
        # Rotate (x, y) by -90 degrees about the tile center: (x, y) -> (y, -x), then
        # re-center. This maps "what used to vary with col (x)" to vary with y, and
        # "what used to vary with row (y)" to vary with x.
        out = points.copy()
        x = points[..., 0] - cx_tile
        y = points[..., 1] - cy_tile
        out[..., 0] = y + cx_tile
        out[..., 1] = -x + cy_tile
        return out

    for mesh in meshes:
        mesh.vertices = _rotate90_xy(mesh.vertices)

    origin = _rotate90_xy(origin)
    #flip the rows, so that the maze is oriented correctly when looking at the sim for x and rows increasing
    MAZE_REGISTRY.record(maze_array.flip(0), maze_rows)
    print(maze_rows)

    return meshes, origin

@configclass
class MeshMazeTerrainCfg(SubTerrainBaseCfg):
    """Config for a labyrinth-based maze terrain."""

    function = maze_terrain

    cell_size: float = 0.6
    maze_max_cols: float = 10.0
    maze_max_rows: float = 10.0
    maze_cols: float | None = None # shouldn't be specified: useful to retrieve information during training
    maze_rows: float | None = None # shouldn't be specified: useful to retrieve information during training

    wall_height: float = 0.5
    wall_height_range: tuple[float, float] | None = None
    wall_thickness: float = 0.05
    wall_thickness_range: tuple[float, float] | None = None
    floor_thickness: float = 0.08
    p_wall_dest: float = 0.3

    # Stairs parameters.
    stairs_prob_per_m2: float = 0.15
    stairs_prob_per_m2_range: tuple[float, float] | None = None
    stairs_step_height: float = 0.05
    stairs_step_height_range: tuple[float, float] | None = None
    stairs_step_depth: float = 0.18
    stairs_step_depth_range: tuple[float, float] | None = None
    # Kept for API compatibility; actual stair footprint width is fixed to the cell.
    stairs_step_width: float = 0.0
    stairs_num_steps: int = 4
    stairs_num_steps_range: tuple[int, int] | None = None
    stairs_landing_depth: float | None = None
    stairs_start_down_prob: float = 0.5
    stairs_hole_margin: float = 0.002

    # Boxes parameters.
    boxes_prob_per_m2: float = 0.12
    boxes_prob_per_m2_range: tuple[float, float] | None = None
    boxes_n_boxes: int = 8
    boxes_n_boxes_range: tuple[int, int] | None = None
    boxes_h_boxes: float = 0.18
    boxes_h_boxes_range: tuple[float, float] | None = None
    boxes_patch_size_ratio: float = 0.9
    boxes_patch_size_ratio_range: tuple[float, float] | None = None
    boxes_min_size: float = 0.01

    # Roughness parameters.
    rough_prob_per_m2: float = 0.10
    rough_prob_per_m2_range: tuple[float, float] | None = None
    rough_n_rough: int = 32
    rough_n_rough_range: tuple[int, int] | None = None
    rough_h_rough_xy: float = 0.15
    rough_h_rough_xy_range: tuple[float, float] | None = None
    rough_h_rough_z: float = 0.01
    rough_h_rough_z_range: tuple[float, float] | None = None
    rough_patch_size_ratio: float = 0.9
    rough_patch_size_ratio_range: tuple[float, float] | None = None
    rough_min_size: float = 0.008

    algorithm: Literal["dfs", "kruskal", "prims", "wilson"] = "dfs"
    seed: int | None = 0


MAZE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(6.0, 6.0),
    border_width=0.0,
    num_rows=3,
    num_cols=3,
    use_cache=False,
    sub_terrains={
        "maze": MeshMazeTerrainCfg(
            proportion=1.0,
            cell_size=0.6,
            wall_height=0.5,
            wall_thickness=0.05,
            floor_thickness=0.08,
            algorithm="dfs",
            seed=0,
        )
        
    },
)
"""Base maze terrain configuration (same usage pattern as ROUGH_TERRAINS_CFG)."""


def make_maze_terrain_cfg(
    cell_size: float,
    wall_height_range: tuple[float, float] = (0.5, 0.5),
    wall_thickness_range: tuple[float, float] = (0.05, 0.05),
    floor_thickness: float = 0.08,
    p_wall_dest: float = 0.3,
    algorithm: Literal["dfs", "kruskal", "prims", "wilson"] = "dfs",
    seed: int | None = 0,
    stairs_prob_per_m2_range: tuple[float, float] = (0.15, 0.15),
    boxes_prob_per_m2_range: tuple[float, float] = (0.12, 0.12),
    rough_prob_per_m2_range: tuple[float, float] = (0.10, 0.10),
    stairs_step_height_range: tuple[float, float] = (0.05, 0.05),
    stairs_step_depth_range: tuple[float, float] = (0.18, 0.18),
    stairs_step_width: float = 0.0,
    stairs_num_steps_range: tuple[int, int] = (4, 4),
    stairs_landing_depth: float | None = None,
    stairs_start_down_prob: float = 0.5,
    stairs_hole_margin: float = 0.002,
    boxes_n_boxes_range: tuple[int, int] = (8, 8),
    boxes_h_boxes_range: tuple[float, float] = (0.18, 0.18),
    boxes_patch_size_ratio_range: tuple[float, float] = (0.9, 0.9),
    rough_n_rough_range: tuple[int, int] = (32, 32),
    rough_h_rough_xy_range: tuple[float, float] = (0.15, 0.15),
    rough_h_rough_z_range: tuple[float, float] = (0.01, 0.01),
    rough_patch_size_ratio_range: tuple[float, float] = (0.9, 0.9),
    terrain_num_rows: int = 1,
    terrain_num_cols: int = 1,
    curriculum: bool = False,
    difficulty_range: tuple[float, float] = (0.0, 1.0),
    maze_max_cols: float = 10.0,
    maze_max_rows: float = 10.0,
) -> TerrainGeneratorCfg:
    """Create a MAZE_TERRAIN_CFG instance for runtime values.

    Returns:
        TerrainGeneratorCfg: Can be passed directly to TerrainImporterCfg(terrain_type="generator").
    """

    max_cols = max(1.0, float(maze_max_cols))
    max_rows = max(1.0, float(maze_max_rows))

    wall_height_low = max(1e-4, min(float(wall_height_range[0]), float(wall_height_range[1])))
    wall_height_high = max(wall_height_low, max(float(wall_height_range[0]), float(wall_height_range[1])))
    wall_thickness_low = max(1e-4, min(float(wall_thickness_range[0]), float(wall_thickness_range[1])))
    wall_thickness_high = max(
        wall_thickness_low, max(float(wall_thickness_range[0]), float(wall_thickness_range[1]))
    )

    p_wall_dest = min(1.0, max(0.0, p_wall_dest))
    
    stairs_prob_low = max(0.0, min(float(stairs_prob_per_m2_range[0]), float(stairs_prob_per_m2_range[1])))
    stairs_prob_high = max(stairs_prob_low, max(float(stairs_prob_per_m2_range[0]), float(stairs_prob_per_m2_range[1])))
    boxes_prob_low = max(0.0, min(float(boxes_prob_per_m2_range[0]), float(boxes_prob_per_m2_range[1])))
    boxes_prob_high = max(boxes_prob_low, max(float(boxes_prob_per_m2_range[0]), float(boxes_prob_per_m2_range[1])))
    rough_prob_low = max(0.0, min(float(rough_prob_per_m2_range[0]), float(rough_prob_per_m2_range[1])))
    rough_prob_high = max(rough_prob_low, max(float(rough_prob_per_m2_range[0]), float(rough_prob_per_m2_range[1])))

    stairs_step_height_low = max(1e-4, min(float(stairs_step_height_range[0]), float(stairs_step_height_range[1])))
    stairs_step_height_high = max(
        stairs_step_height_low, max(float(stairs_step_height_range[0]), float(stairs_step_height_range[1]))
    )
    stairs_step_depth_low = max(1e-4, min(float(stairs_step_depth_range[0]), float(stairs_step_depth_range[1])))
    stairs_step_depth_high = max(
        stairs_step_depth_low, max(float(stairs_step_depth_range[0]), float(stairs_step_depth_range[1]))
    )
    stairs_num_steps_low = max(1, min(int(stairs_num_steps_range[0]), int(stairs_num_steps_range[1])))
    stairs_num_steps_high = max(stairs_num_steps_low, max(int(stairs_num_steps_range[0]), int(stairs_num_steps_range[1])))

    boxes_n_low = max(0, min(int(boxes_n_boxes_range[0]), int(boxes_n_boxes_range[1])))
    boxes_n_high = max(boxes_n_low, max(int(boxes_n_boxes_range[0]), int(boxes_n_boxes_range[1])))
    boxes_h_low = max(1e-4, min(float(boxes_h_boxes_range[0]), float(boxes_h_boxes_range[1])))
    boxes_h_high = max(boxes_h_low, max(float(boxes_h_boxes_range[0]), float(boxes_h_boxes_range[1])))
    boxes_patch_low = max(0.0, min(float(boxes_patch_size_ratio_range[0]), float(boxes_patch_size_ratio_range[1])))
    boxes_patch_high = max(
        boxes_patch_low, max(float(boxes_patch_size_ratio_range[0]), float(boxes_patch_size_ratio_range[1]))
    )

    rough_n_low = max(0, min(int(rough_n_rough_range[0]), int(rough_n_rough_range[1])))
    rough_n_high = max(rough_n_low, max(int(rough_n_rough_range[0]), int(rough_n_rough_range[1])))
    rough_xy_low = max(1e-4, min(float(rough_h_rough_xy_range[0]), float(rough_h_rough_xy_range[1])))
    rough_xy_high = max(rough_xy_low, max(float(rough_h_rough_xy_range[0]), float(rough_h_rough_xy_range[1])))
    rough_z_low = max(1e-4, min(float(rough_h_rough_z_range[0]), float(rough_h_rough_z_range[1])))
    rough_z_high = max(rough_z_low, max(float(rough_h_rough_z_range[0]), float(rough_h_rough_z_range[1])))
    rough_patch_low = max(0.0, min(float(rough_patch_size_ratio_range[0]), float(rough_patch_size_ratio_range[1])))
    rough_patch_high = max(
        rough_patch_low, max(float(rough_patch_size_ratio_range[0]), float(rough_patch_size_ratio_range[1]))
    )

    # The sub-terrain tile is sized to the maximum maze extent.
    terrain_size = max(max_rows * cell_size, max_cols * cell_size)
    terrain_size_x = terrain_size
    terrain_size_y = terrain_size
    diff_low = min(float(difficulty_range[0]), float(difficulty_range[1]))
    diff_high = max(float(difficulty_range[0]), float(difficulty_range[1]))

    sub_cfg = MAZE_TERRAIN_CFG.sub_terrains["maze"].replace(
        cell_size=cell_size,
        maze_max_cols=max_cols,
        maze_max_rows=max_rows,
        wall_height=wall_height_low,
        wall_height_range=(wall_height_low, wall_height_high),
        wall_thickness=wall_thickness_low,
        wall_thickness_range=(wall_thickness_low, wall_thickness_high),
        floor_thickness=floor_thickness,
        p_wall_dest=p_wall_dest,
        algorithm=algorithm,
        seed=seed,
        stairs_prob_per_m2=stairs_prob_low,
        stairs_prob_per_m2_range=(stairs_prob_low, stairs_prob_high),
        boxes_prob_per_m2=boxes_prob_low,
        boxes_prob_per_m2_range=(boxes_prob_low, boxes_prob_high),
        rough_prob_per_m2=rough_prob_low,
        rough_prob_per_m2_range=(rough_prob_low, rough_prob_high),
        stairs_step_height=stairs_step_height_low,
        stairs_step_height_range=(stairs_step_height_low, stairs_step_height_high),
        stairs_step_depth=stairs_step_depth_low,
        stairs_step_depth_range=(stairs_step_depth_low, stairs_step_depth_high),
        stairs_step_width=stairs_step_width,
        stairs_num_steps=stairs_num_steps_low,
        stairs_num_steps_range=(stairs_num_steps_low, stairs_num_steps_high),
        stairs_landing_depth=stairs_landing_depth,
        stairs_start_down_prob=stairs_start_down_prob,
        stairs_hole_margin=stairs_hole_margin,
        boxes_n_boxes=boxes_n_low,
        boxes_n_boxes_range=(boxes_n_low, boxes_n_high),
        boxes_h_boxes=boxes_h_low,
        boxes_h_boxes_range=(boxes_h_low, boxes_h_high),
        boxes_patch_size_ratio=boxes_patch_low,
        boxes_patch_size_ratio_range=(boxes_patch_low, boxes_patch_high),
        rough_n_rough=rough_n_low,
        rough_n_rough_range=(rough_n_low, rough_n_high),
        rough_h_rough_xy=rough_xy_low,
        rough_h_rough_xy_range=(rough_xy_low, rough_xy_high),
        rough_h_rough_z=rough_z_low,
        rough_h_rough_z_range=(rough_z_low, rough_z_high),
        rough_patch_size_ratio=rough_patch_low,
        rough_patch_size_ratio_range=(rough_patch_low, rough_patch_high),
    )

    return MAZE_TERRAIN_CFG.replace(
        size=(terrain_size_x, terrain_size_y),
        num_rows=terrain_num_rows,
        num_cols=terrain_num_cols,
        curriculum=curriculum,
        difficulty_range=(diff_low, diff_high),
        sub_terrains={"maze": sub_cfg},
    )
