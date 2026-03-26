import os
import sys
import numpy as np
import jax.numpy as jnp
from src.io import EclipseParser
from src.core.simulator import Simulator

def debug_physics():
    print("=== Physics Diagnostic Audit (Step 0) ===")
    data_file = "comparison/opm_run/SPE1CASE1_OPM.DATA"
    parser = EclipseParser(data_file)
    model = parser.build_model()
    sim = Simulator(model)
    
    # Extract initial state
    p = model.pressure.flatten(order='F')
    sg = model.sgas.flatten(order='F')
    rs = model.rs.flatten(order='F')
    is_sat = sg > 0
    y = jnp.where(is_sat, sg, rs)
    
    dt = 1.0
    # Manually trigger residual calculation
    R, cfl = sim._calc_residuals_jax(p, y, is_sat, p, sg, rs, dt)
    
    R_o = R[0::2]
    R_g = R[1::2]
    
    # Analyze Fluxes specifically
    # Recall flow_o = flow_o.at[...].add(flux) logic
    # We can't easily see internal JAX flux variables, but we can check the cell sums.
    print(f"Max T=0 Residuals (Hydrostatic Equilibrium Test):")
    print(f"  Field: SPE1CASE1 (datum 8400, Pdatum 4800)")
    print(f"  Max R_oil: {np.max(np.abs(R_o)):.1e} STB/day")
    print(f"  Max R_gas: {np.max(np.abs(R_g)):.1e} MSCF/day")
    
    # Check if Layer 1 is worse than Layer 3
    for k in range(model.grid.nz):
        R_k = R_o.reshape(model.grid.dimensions, order='F')[:,:,k]
        print(f"  Max R_oil Layer {k+1}: {np.max(np.abs(R_k)):.1e}")

if __name__ == "__main__":
    debug_physics()
