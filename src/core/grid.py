import numpy as np
from typing import Tuple, Optional, Any

class Grid:
    """Represents the 3D spatial discretization of the reservoir."""
    
    def __init__(self, nx: int, ny: int, nz: int, dx: any, dy: any, dz: any, actnum: Optional[np.ndarray] = None, top_depth: Optional[np.ndarray] = None):
        """
        :param nx, ny, nz: Dimensions
        :param dx, dy, dz: Spacing (scalar or (nx, ny, nz) array)
        :param actnum: (nx, ny, nz) array of 0 (inactive) or 1 (active)
        :param top_depth: Depth of the tops of cells (nx, ny, nz)
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = self._as_array(dx)
        self.dy = self._as_array(dy)
        self.dz = self._as_array(dz)
        self.actnum = actnum if actnum is not None else np.ones((nx, ny, nz), dtype=np.int32)
        self.top_depth = top_depth
        
        if self.top_depth is not None:
            self.z_centers = self.top_depth + self.dz / 2.0
        else:
            self.z_centers = np.cumsum(self.dz, axis=2) - self.dz / 2.0

    def _as_array(self, val):
        if isinstance(val, (int, float)):
            return np.full((self.nx, self.ny, self.nz), float(val))
        return np.array(val).reshape((self.nx, self.ny, self.nz), order='F')
        
    @property
    def total_cells(self) -> int:
        return self.nx * self.ny * self.nz
        
    @property
    def dimensions(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)
        
    def get_cell_volume(self) -> np.ndarray:
        """Volume of grid cells (nx, ny, nz)."""
        return self.dx * self.dy * self.dz

    def get_total_volume(self) -> float:
        """Total volume of the reservoir."""
        return self.total_cells * self.get_cell_volume()
