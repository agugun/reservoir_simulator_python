from enum import Enum, auto
from typing import Tuple, Dict

class BoundaryType(Enum):
    NO_FLOW = auto()  # Neumann: dp/dn = 0
    CONSTANT_PRESSURE = auto()  # Dirichlet: p = const
    CONSTANT_FLUX = auto()  # Neumann: dp/dn = const (Aq)

class BoundaryCondition:
    """Represents a boundary condition on a specific face or region of the grid."""
    
    def __init__(self, boundary_type: BoundaryType, value: float = 0.0):
        """
        :param boundary_type: The type of boundary condition
        :param value: The specified pressure or flux value
        """
        self.boundary_type = boundary_type
        self.value = value

class GridBoundaries:
    """Manages the boundary conditions for the 6 faces of a 3D grid."""
    
    def __init__(self):
        # Default all faces to no-flow
        self.faces = {
            'x_min': BoundaryCondition(BoundaryType.NO_FLOW),
            'x_max': BoundaryCondition(BoundaryType.NO_FLOW),
            'y_min': BoundaryCondition(BoundaryType.NO_FLOW),
            'y_max': BoundaryCondition(BoundaryType.NO_FLOW),
            'z_min': BoundaryCondition(BoundaryType.NO_FLOW),
            'z_max': BoundaryCondition(BoundaryType.NO_FLOW),
        }
        
    def set_boundary(self, face: str, condition: BoundaryCondition):
        """
        Set a boundary condition for a specific face.
        :param face: One of 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        """
        if face not in self.faces:
            raise ValueError(f"Invalid face '{face}'. Must be one of {list(self.faces.keys())}")
        self.faces[face] = condition
