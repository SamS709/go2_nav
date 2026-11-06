"""Isaac-free perception stack for the Go2 navigation task.

Everything in this package depends on torch only (plus ``rsl_rl`` for the odometry GRU), so the exact same code can
run in the simulator and on the robot.

    LocomotionController  - frozen low-level policy (TorchScript) + its observation/action bookkeeping
    HeightGrid            - ray hits (lidar frame) -> 2D height grid (policy convention)
    OdometryEstimator     - GRU velocity/attitude network + dead-reckoning + pose histories
    MapPerception         - rolling Kalman elevation map + scan-to-map matching + ego-crop denoiser
"""

from .config import MapParams, MatchParams, OdomParams, PerceptionParams, TrainParams
from .height_grid import HeightGrid
from .locomotion import LocomotionController
from .mapping import ElevationMap, MapDenoiser, MapPerception, ScanBuilder
from .odometry import OdometryEstimator, PositionHistory
from .scan_matching import ScanToMapMatcher

__all__ = [
    "MapParams", "MatchParams", "OdomParams", "PerceptionParams", "TrainParams",
    "HeightGrid", "LocomotionController", "ElevationMap", "MapDenoiser", "MapPerception", "ScanBuilder",
    "OdometryEstimator", "PositionHistory", "ScanToMapMatcher",
]