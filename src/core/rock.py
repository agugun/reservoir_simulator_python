import numpy as np
from typing import Tuple

class Rock:
    """Represents the rock properties of the reservoir."""
    def __init__(self, porosity: np.ndarray, perm_x: np.ndarray, perm_y: np.ndarray, perm_z: np.ndarray, compressibility: float):
        """
        :param porosity: 3D array of porosity values
        :param perm_x: 3D array of permeability in x-direction (mD)
        :param perm_y: 3D array of permeability in y-direction (mD)
        :param perm_z: 3D array of permeability in z-direction (mD)
        :param compressibility: Rock compressibility (1/psi)
        """
        self.porosity = porosity
        self.perm_x = perm_x
        self.perm_y = perm_y
        self.perm_z = perm_z
        self.compressibility = compressibility

    @classmethod
    def homogeneous(cls, dimensions: Tuple[int, int, int], porosity: float, perm_x: float, perm_y: float, perm_z: float, compressibility: float):
        """Creates a Rock instance with uniform properties everywhere."""
        return cls(
            porosity=np.full(dimensions, porosity),
            perm_x=np.full(dimensions, perm_x),
            perm_y=np.full(dimensions, perm_y),
            perm_z=np.full(dimensions, perm_z),
            compressibility=compressibility
        )
