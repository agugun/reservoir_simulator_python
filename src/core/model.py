import numpy as np
from typing import List, Optional
from .grid import Grid
from .rock import Rock
from .fluid import Fluid
from .well import Well

class ReservoirModel:
    """The complete reservoir simulation model."""
    def __init__(self, grid: Grid, rock: Rock, fluid: Fluid, initial_pressure: np.ndarray, wells: Optional[List[Well]] = None, swat: Optional[np.ndarray] = None, sgas: Optional[np.ndarray] = None, rs: Optional[np.ndarray] = None, start_date: str = "01-JAN-2015"):
        """
        :param grid: Grid object containing spatial discretization
        :param rock: Rock object containing petrophysical properties
        :param fluid: Fluid object containing PVT properties
        :param initial_pressure: 3D array of initial pressure distribution
        :param wells: List of Well objects in the model
        :param swat: 3D array of initial water saturation
        :param sgas: 3D array of initial gas saturation
        :param rs: 3D array of dissolved gas ratio
        :param start_date: Simulation start date (e.g., '01-JAN-2015')
        """
        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.pressure = initial_pressure
        self.wells = wells if wells is not None else []
        self.swat = swat if swat is not None else np.full(grid.dimensions, 0.12, dtype=np.float32)
        self.sgas = sgas if sgas is not None else np.zeros(grid.dimensions, dtype=np.float32)
        self.rs = rs if rs is not None else np.zeros(grid.dimensions, dtype=np.float32)
        self.start_date = start_date
        
        self._validate()

    def _validate(self):
        """Validates that dimensions of properties match the grid dimensions."""
        dims = self.grid.dimensions
        if self.rock.porosity.shape != dims:
            raise ValueError(f"Rock porosity dimensions {self.rock.porosity.shape} do not match grid dimensions {dims}")
        if self.pressure.shape != dims:
            raise ValueError(f"Initial pressure dimensions {self.pressure.shape} do not match grid dimensions {dims}")
