import pytest
import numpy as np
from src.core import Grid, Rock, Fluid, ReservoirModel, Well, BoundaryCondition, BoundaryType

def test_grid_creation():
    grid = Grid(nx=10, ny=10, nz=5, dx=100.0, dy=100.0, dz=20.0)
    assert grid.total_cells == 500
    assert grid.dimensions == (10, 10, 5)
    assert grid.get_cell_volume() == pytest.approx(200000.0)
    assert grid.get_total_volume() == pytest.approx(100000000.0)

def test_rock_homogeneous():
    dims = (10, 10, 5)
    rock = Rock.homogeneous(dims, porosity=0.2, perm_x=100.0, perm_y=100.0, perm_z=10.0, compressibility=1e-6)
    assert rock.porosity.shape == dims
    assert rock.porosity[0, 0, 0] == 0.2
    assert rock.perm_x[9, 9, 4] == 100.0

def test_fluid_creation():
    fluid = Fluid(viscosity=1.0, density=62.4, compressibility=1e-5, formation_volume_factor=1.1)
    assert fluid.viscosity == 1.0

def test_reservoir_model_validation():
    grid = Grid(nx=10, ny=10, nz=5, dx=100.0, dy=100.0, dz=20.0)
    dims = grid.dimensions
    rock = Rock.homogeneous(dims, porosity=0.2, perm_x=100.0, perm_y=100.0, perm_z=10.0, compressibility=1e-6)
    fluid = Fluid(viscosity=1.0, density=62.4, compressibility=1e-5, formation_volume_factor=1.1)
    
    initial_pressure = np.full(dims, 3000.0)
    
    model = ReservoirModel(grid, rock, fluid, initial_pressure)
    assert model.pressure.shape == dims
    assert len(model.wells) == 0

    # Test failure on dimension mismatch
    bad_pressure = np.full((10, 10, 4), 3000.0)
    with pytest.raises(ValueError):
        ReservoirModel(grid, rock, fluid, bad_pressure)

def test_well_creation():
    well = Well(name="PROD1", location=(5, 5, 2), rate=-1000.0)
    assert well.is_producer
    assert not well.is_injector
    assert well.rate == -1000.0

    well2 = Well(name="INJ1", location=(1, 1, 0), rate=2000.0)
    assert well2.is_injector

    with pytest.raises(ValueError):
        Well(name="BAD", location=(0,0,0), rate=0.0)
