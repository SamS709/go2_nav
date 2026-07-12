import torch
from isaaclab.terrains import TerrainImporter

def env_ids_to_terrain_coords(env_ids: torch.Tensor, terrain: TerrainImporter) -> torch.Tensor:
    rows = terrain.terrain_levels[env_ids].clone()
    cols = terrain.terrain_types[env_ids].clone()
    return torch.stack([rows, cols], dim=1)