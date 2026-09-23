# Maze shortest path

This note explains the maze-aware distance implementation in `go2_nav_env.py`.

## What changed

The environment no longer relies only on Euclidean distance for navigation progress. It now uses the generated maze walls to estimate the shortest drivable route from the robot's current cell to the goal cell.

The implementation has two parts:

1. Build a shortest-distance lookup table for every maze.
2. Convert the robot and goal positions into cells and query that table at each policy step.

The path is also reconstructed for the global-map debug plot.

## Maze representation

Each generated cell stores five values:

```text
layout[row, col, 0] = terrain type
layout[row, col, 1] = wall on the row-1 side of the recorded cell
layout[row, col, 2] = wall on the col+1 side
layout[row, col, 3] = wall on the row+1 side
layout[row, col, 4] = wall on the col-1 side
```

A value of `1` means that the wall exists. A value of `0` means that the neighboring cell can be entered through that side.

The terrain generator creates a padded tensor of size `MAX_MAZE_ROWS x MAX_MAZE_COLS`, but only the first `maze_rows` rows are active for a particular curriculum level. The generator stores the active rows flipped vertically:

```python
maze_tensor_recorded = maze_tensor.clone()
maze_tensor_recorded[:maze_rows] = maze_tensor[:maze_rows].flip(0)
MAZE_REGISTRY.record(maze_tensor_recorded, maze_rows)
```

Only active rows are flipped. This is important. Flipping the complete padded tensor would move a small curriculum maze into the inactive rows and make the planner read the wrong topology.

Because the active rows are flipped, moving from recorded row `r` to `r - 1` crosses the current cell's `row+1` wall and the neighbor's `row-1` wall:

```python
(-1, 0, 3, 1)
```

Moving from recorded row `r` to `r + 1` uses:

```python
(1, 0, 1, 3)
```

The column directions are unchanged:

```python
(0, -1, 4, 2)  # move left
(0,  1, 2, 4)  # move right
```

A transition is legal only when both sides have no wall.

## Functions

### `_build_maze_path_tables`

Called once during environment initialization after `MAZE_REGISTRY` has been populated.

For each vectorized environment it:

1. Finds the environment's terrain row and column.
2. Retrieves its recorded maze layout.
3. Reads the active number of rows and fixed number of columns.
4. Computes an all-pairs shortest-distance table.
5. Stores the table on the simulation device.

Identical maze layouts are cached, so environments sharing the same terrain do not repeat the BFS calculation.

The final lookup has this conceptual form:

```text
maze_path_distances[env, start_cell, goal_cell] = number of cell transitions
```

### `_compute_maze_distance_table`

This is the topology solver. It runs a breadth-first search from every cell.

BFS is appropriate because every legal move from one cell to an adjacent cell has the same cost: one cell transition. The result is an all-pairs table. An unreachable pair remains `inf`.

For a cell `(row, col)`, each candidate neighbor is checked against the two wall values shared by the boundary. For example, the move from `(row, col)` to `(row - 1, col)` is allowed only if:

```python
layout[row, col, 3] == 0
layout[row - 1, col, 1] == 0
```

### `_world_to_maze_cells`

This converts world positions into recorded maze coordinates.

The terrain origin is the center of the reserved spawn cell. The conversion therefore:

1. Subtracts the terrain tile origin.
2. Divides by `cell_size` and rounds to the nearest cell offset.
3. Converts the world x/y offset into the generator's row/column convention.
4. Applies the active-row flip used by the registry.
5. Clamps the result to valid active cells.

After the generator rotates the mesh, world `x` follows the original maze rows and world `y` follows the negative original maze columns. For the current terrain convention:

```text
spawn_row = floor(maze_rows / 2)
spawn_col = floor(maze_cols / 2)
original_row = spawn_row + round(local_x / cell_size)
original_col = spawn_col - round(local_y / cell_size)
recorded_row = maze_rows - 1 - original_row
recorded_col = original_col
```

The same mapping is used in reverse when plotting a recorded cell:

```text
original_row = maze_rows - 1 - recorded_row
world_x = tile_origin_x + (original_row - spawn_row) * cell_size
world_y = tile_origin_y - (recorded_col - spawn_col) * cell_size
```

### `_maze_cell_centers_world`

This is the inverse coordinate operation for cell centers. It converts recorded `(row, col)` indices back into world XY coordinates. It is used to calculate the partial distance from the robot or goal to the center of its cell.

### `_compute_maze_path_distance`

This is the function used by the reward.

It computes:

```text
path_distance = cell_distance * cell_size
              + distance(robot, robot_cell_center)
              + distance(goal, goal_cell_center)
```

The cell-to-cell portion accounts for walls. The endpoint terms prevent the metric from jumping by a complete cell when the robot moves within a cell.

If the two cells are disconnected, the function falls back to Euclidean distance. This is a defensive fallback for malformed or inconsistent terrain data; generated maze layouts should normally be connected.

The reward uses this value for:

- `rew_goal_distance`
- `rew_goal_progress`

The existing Euclidean distance is still used for local goal orientation gating and termination checks.

### `_shortest_maze_path_cells`

This is another BFS, used only for visualization. It stores a predecessor for every visited cell and walks backward from the goal to the start to reconstruct one shortest route:

```text
start <- predecessor <- ... <- goal
```

The result is a list such as:

```python
[(2, 1), (1, 1), (1, 2), (1, 3), (0, 3)]
```

The reward does not call this function because it only needs the precomputed distance, not the complete route.

## Concrete example

Assume a `3 x 4` recorded maze with `cell_size = 3.0 m`. Use recorded row/column coordinates and ignore terrain type (`channel 0`) for this example.

Suppose the robot is in `(2, 1)` and the goal is in `(0, 3)`. The relevant open passages are:

```text
(2,1) -> (1,1) -> (1,2) -> (1,3) -> (0,3)
```

The BFS begins at `(2, 1)`:

```text
queue = [(2, 1)]
distance[(2, 1)] = 0
```

To visit `(1, 1)`, the solver checks the row-1 transition:

```python
layout[2, 1, 3] == 0
layout[1, 1, 1] == 0
```

So it assigns:

```text
distance[(1,1)] = distance[(2,1)] + 1 = 1
```

It then visits `(1, 2)` through the right boundary:

```python
layout[1, 1, 2] == 0
layout[1, 2, 4] == 0
```

and assigns distance `2`. Continuing gives:

```text
cell path:       (2,1) -> (1,1) -> (1,2) -> (1,3) -> (0,3)
cell transitions:  0        1        2        3        4
```

The topological part of the distance is therefore:

```text
4 transitions * 3.0 m = 12.0 m
```

If the robot is `0.4 m` from the center of `(2,1)` and the goal is `0.7 m` from the center of `(0,3)`, then:

```text
path_distance = 12.0 + 0.4 + 0.7 = 13.1 m
```

The straight-line distance could be much shorter, but it ignores the walls. The path distance correctly reflects the route the robot must take through the maze.

## Plotting

`_plot_map_s` calls `_shortest_maze_path_cells` for the selected environment. It then:

1. Converts each recorded cell center back to world XY.
2. Converts world XY into global heightfield indices.
3. Draws the route in yellow.
4. Draws the robot as a red `+`.
5. Draws the goal as a blue `*`.

The route is drawn on the global heightfield panel because that panel uses world-terrain coordinates. The heightfield is stored as `[x_index, y_index]` and displayed with `data.T`, producing an image whose horizontal axis is still `x` and vertical axis is `y`. Therefore the overlay uses `(world_x, world_y)` directly; it must not swap the two coordinates.

Plots are generated on reset by the existing call:

```python
self._plot_map_s(env_ids_tensor, save_dir=".../map_debug")
```

The files under `map_debug/envN/` are therefore snapshots from reset time. To see a route after the robot has moved, call `_plot_map_s` at the desired step or add a periodic debug trigger.
